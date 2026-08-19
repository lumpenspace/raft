import os
import json
import requests
from bs4 import BeautifulSoup
from time import sleep
from random import randrange
from typing import Dict, Any, Generator, List


def fetch_json(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch JSON data from the given URL."""
    endpoint = f"{url}/api/v1/archive"
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


def fetch_html(url: str) -> str:
    """Fetch HTML content from the given URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def fetch_and_parse(url: str) -> Generator[Dict[str, Any], None, None]:
    """Fetch and parse blog posts from the given URL."""
    limit = 12
    offset = 0
    while True:
        params = {"limit": limit, "offset": offset}
        entries = fetch_json(url, params=params)
        if not entries:
            break
        for item in entries:
            link = item["canonical_url"]
            title = item["title"]
            date = item["post_date"]
            html = fetch_html(link)
            soup = BeautifulSoup(html, "html.parser")
            content = soup.find("div", {"class": "markup"})
            if content:
                yield {
                    "title": title,
                    "link": link,
                    "date": date,
                    "content": content.text,
                }
            timeout = randrange(2, 20)
            sleep(timeout)
        offset += limit


def main(blog: str) -> None:
    """Main function to fetch and save blog posts."""
    url = f"https://{blog}.substack.com"
    filename = f'data/{url.replace("https://", "").replace(".", "-")}.jsonl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for post in fetch_and_parse(url):
            f.write(json.dumps(post) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python substack_embeddings.py <blog_name>")
        sys.exit(1)
    main(sys.argv[1])
