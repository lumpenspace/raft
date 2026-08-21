"""
Document sources for the grounding corpus: RSS/Atom feeds, single web
pages, and PDFs.

Everything lands in data/{name}.jsonl -- one {"title", "link", "date",
"content"} object per line, the input of `raft chunk` -- so sources can
be added one at a time and combined with each other (and with the
substack / tweet / local-file imports). Records whose link is already in
the corpus are skipped, so re-running a feed only adds what is new.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from . import hx

USER_AGENT = "raft-ft (+https://github.com/lumpenspace/raft)"

PYPDF_INSTALL_HINT = (
    "pypdf is not installed. Install it with:\n"
    "  pip install 'raft-ft[pdf]'"
)

# A feed entry shorter than this is probably a teaser; fetch the page.
FULL_PAGE_THRESHOLD = 500


def iso_date(value: Any) -> str:
    """
    Normalise a timestamp to YYYY-MM-DD.

    Handles ISO 8601 and the RFC 2822 format used by RSS pubDate and
    Twitter archive exports ("Mon Jan 01 12:00:00 +0000 2024").
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return ""


def date_num(value: Any) -> int:
    """
    A date as a comparable YYYYMMDD integer, 0 if unparseable.

    Chroma metadata filters ($lt and friends) only work on numbers, so
    this is the form dates take in the embedding store.
    """
    day = iso_date(value)
    return int(day.replace("-", "")) if day else 0


def _http_get(url: str) -> str:
    """Fetch a URL and return its body text."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def html_to_text(html: str) -> str:
    """Strip markup, keeping line structure (chunking splits on newlines)."""
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


def append_corpus_records(name: str, records: List[Dict[str, Any]]) -> int:
    """
    Append records to data/{name}.jsonl, skipping links already there.

    Returns:
        int: Number of records actually appended.
    """
    corpus_path = f"data/{name}.jsonl"
    os.makedirs("data", exist_ok=True)

    seen_links = set()
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    link = json.loads(line).get("link")
                    if link:
                        seen_links.add(link)

    added = 0
    with open(corpus_path, "a") as f:
        for record in records:
            link = record.get("link", "")
            if link and link in seen_links:
                continue
            record.setdefault("title", "untitled")
            record.setdefault("link", "")
            # An unknown date stays empty (date_num 0): under the
            # earlier-writings retrieval filter that means "always
            # retrievable", whereas stamping today would hide the
            # document from every interview that predates the import.
            record.setdefault("date", "")
            f.write(json.dumps(record) + "\n")
            if link:
                seen_links.add(link)
            added += 1
    return added


def extract_page(html: str, url: str) -> Dict[str, str]:
    """
    Extract a grounding record from a web page.

    Title comes from og:title falling back to <title>; date from the
    article:published_time meta tag when present; content from
    <article>, substack's div.markup, <main>, or the whole body.
    """
    soup = BeautifulSoup(html, "html.parser")

    og_title = soup.find("meta", property="og:title")
    title = (og_title or {}).get("content") or (
        soup.title.get_text(strip=True) if soup.title else ""
    )

    published = soup.find("meta", property="article:published_time")
    date = iso_date((published or {}).get("content"))

    body = (
        soup.find("article")
        or soup.find("div", {"class": "markup"})
        or soup.find("main")
        or soup.body
        or soup
    )
    return {
        "title": title or url,
        "link": url,
        "date": date,
        "content": body.get_text("\n", strip=True),
    }


def fetch_url(name: str, url: str) -> int:
    """Fetch one web page and append it to the corpus. Returns docs added."""
    record = extract_page(_http_get(url), url)
    return append_corpus_records(name, [record])


def _element_html(el: Optional[ET.Element]) -> str:
    """The inner markup of a feed element (Atom xhtml content has children)."""
    if el is None:
        return ""
    parts = [el.text or ""]
    parts.extend(ET.tostring(child, encoding="unicode") for child in el)
    return "".join(parts)


def _entry_link(item: ET.Element) -> str:
    """The entry URL: RSS <link> text, or an Atom <link href> (rel=alternate first)."""
    links = item.findall("{*}link")
    for link in links:
        if (link.text or "").strip():
            return link.text.strip()
    for link in links:
        if link.get("rel", "alternate") == "alternate" and link.get("href"):
            return link.get("href", "")
    return links[0].get("href", "") if links else ""


def parse_feed(xml_text: str) -> List[Dict[str, str]]:
    """
    Parse an RSS/Atom/RDF feed into entries.

    Returns:
        List[Dict[str, str]]: One {"title", "link", "date", "html"} per
        entry; "html" is the entry's fullest available content markup.
    """
    root = ET.fromstring(xml_text)
    items = root.findall(".//{*}item") or root.findall(".//{*}entry")

    entries = []
    for item in items:
        date = ""
        for tag in ("pubDate", "published", "updated", "date"):
            date = iso_date(item.findtext(f"{{*}}{tag}"))
            if date:
                break
        html = ""
        for tag in ("encoded", "content", "summary", "description"):
            html = _element_html(item.find(f"{{*}}{tag}"))
            if html.strip():
                break
        entries.append(
            {
                "title": (item.findtext("{*}title") or "").strip(),
                "link": _entry_link(item),
                "date": date,
                "html": html,
            }
        )
    return entries


def resolve_feed(url: str) -> List[Dict[str, str]]:
    """
    Fetch a feed URL and parse its entries. Given a regular page instead
    of a feed, follow its rel=alternate feed link (so a blog homepage
    works as well as its /feed).
    """
    body = _http_get(url)
    try:
        return parse_feed(body)
    except ET.ParseError:
        pass
    soup = BeautifulSoup(body, "html.parser")
    alternate = soup.find(
        "link", {"type": ["application/rss+xml", "application/atom+xml"]}
    )
    if alternate and alternate.get("href"):
        feed_url = requests.compat.urljoin(url, alternate["href"])
        hx.say(f"following feed link: {feed_url}")
        return parse_feed(_http_get(feed_url))
    raise ValueError(f"{url} is neither a feed nor a page that links to one")


def fetch_feed(name: str, url: str, fetch_pages: bool = True) -> int:
    """
    Import an RSS/Atom feed into the corpus.

    Entries whose feed content is a teaser (shorter than
    FULL_PAGE_THRESHOLD chars) have their linked page fetched instead
    when fetch_pages is set.

    Returns:
        int: Number of documents added (already-imported links skipped).
    """
    entries = resolve_feed(url)
    records: List[Dict[str, Any]] = []
    for entry in entries:
        text = html_to_text(entry["html"])
        if fetch_pages and entry["link"] and len(text) < FULL_PAGE_THRESHOLD:
            try:
                page = extract_page(_http_get(entry["link"]), entry["link"])
                text = page["content"]
                time.sleep(1)
            except requests.RequestException as e:
                hx.warn(f"could not fetch {entry['link']}: {e}")
        if not text.strip():
            continue
        records.append(
            {
                "title": entry["title"] or entry["link"] or "untitled",
                "link": entry["link"],
                "date": entry["date"],
                "content": text,
            }
        )
    return append_corpus_records(name, records)


def import_pdf(name: str, path: str) -> int:
    """
    Extract a PDF's text and append it to the corpus as one document.

    Raises:
        RuntimeError: If pypdf is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(PYPDF_INSTALL_HINT)

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError(f"{path}: no extractable text (a scanned PDF?)")

    title = (reader.metadata.title if reader.metadata else "") or os.path.splitext(
        os.path.basename(path)
    )[0]
    # The file mtime says when the PDF was downloaded, not written, so
    # the date stays unknown (always retrievable) rather than wrong.
    return append_corpus_records(
        name,
        [{"title": title, "link": path, "date": "", "content": text}],
    )
