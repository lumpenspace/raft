"""
Tweet mode: build a raft dataset from tweets via ariadne
(https://github.com/lumpenspace/ariadne).

Uses ariadne's Python API directly, so the tweet documents come back as
data rather than through a file on disk: thread texts become grounding
documents (data/{name}.jsonl, ready for `raft chunk`) and reply branches
become q/a transcripts (ready for `raft ft:gen`).
"""

import json
import os
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

from .convo_structurer import messages_to_exchanges, write_transcript
from . import hx
from .interactive import ask, ask_path, bail, choose, confirm

ARIADNE_INSTALL_HINT = (
    "ariadne is not installed. Install it with:\n"
    "  pip install git+https://github.com/lumpenspace/ariadne"
)

# Tweet conversations are short; batch them so generate_finetune doesn't
# crawl through hundreds of one-exchange transcript files.
EXCHANGES_PER_TRANSCRIPT = 40


def load_ariadne():
    """Import ariadne, or exit with an install hint."""
    try:
        import ariadne
    except ImportError:
        bail(ARIADNE_INSTALL_HINT)
    if not hasattr(ariadne, "build"):
        bail(
            "the installed ariadne is too old for raft 2.0 (no Python API).\n"
            "Upgrade with: pip install -U git+https://github.com/lumpenspace/ariadne"
        )
    return ariadne


def ask_build_options() -> Dict[str, Any]:
    """Ask where the tweets come from, and return ariadne build options."""
    source = choose(
        "Where should the tweets come from?",
        [
            "an X/Twitter archive export (zip or folder)",
            "a CSV/JSON/JSONL tweet dump",
            "a public user, fetched live (cheap sources first)",
        ],
        default=0,
    )

    options: Dict[str, Any] = {"allow_empty": True}
    if source == 0:
        options["archive"] = ask_path("Archive path (zip or folder)")
    elif source == 1:
        options["tweets_file"] = ask_path("Tweet dump path (csv/json/jsonl)")
    else:
        options["target_user"] = ask("Public handle to collect").strip("@")

    if "target_user" not in options:
        handle = ask("Handle whose tweets to select (empty = all loaded)", "")
        if handle:
            options["for_user"] = handle.strip("@")
        else:
            options["all_loaded"] = True

    since = ask("Only tweets on or after [YYYY-MM-DD] (empty = all)", "")
    if since:
        options["since"] = since

    if confirm("Only use replies as conversation starts?", default=False):
        options["replies_only"] = True

    return options


def iso_date(value: Any) -> str:
    """
    Normalise a tweet timestamp to YYYY-MM-DD.

    Archive exports carry Twitter's legacy format ("Mon Jan 01 12:00:00
    +0000 2024") while the API returns ISO 8601, so slicing the string is
    not enough.
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


def import_documents(name: str, documents: List[Dict[str, Any]], target: str) -> None:
    """
    Import ariadne raft documents as grounding corpus + transcripts.

    Args:
        name (str): Dataset name (data/{name}*).
        documents (List[Dict[str, Any]]): raft.documents.v1 objects.
        target (str): Handle/name of the person being emulated.
    """
    os.makedirs("data", exist_ok=True)
    corpus_path = f"data/{name}.jsonl"
    # Ariadne reports usernames without the leading @; match on the bare handle.
    handle = target.lstrip("@")
    seen_ids = set()
    n_docs = 0

    all_exchanges: List[List[str]] = []
    first_date = ""

    with open(corpus_path, "a") as corpus:
        for doc in documents:
            meta = doc.get("metadata", {})
            doc_id = doc.get("id") or str(meta.get("tweet_ids"))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            text = (doc.get("text") or "").strip()
            date = iso_date(meta.get("target_created_at"))
            if text:
                corpus.write(
                    json.dumps(
                        {
                            "title": f"tweet thread {meta.get('target_id', doc_id)}",
                            "link": meta.get("target_url") or "",
                            "date": date or "unknown",
                            "content": text,
                        }
                    )
                    + "\n"
                )
                n_docs += 1

            exchanges, _ = messages_to_exchanges(
                normalize_messages(doc.get("messages") or [], handle), handle
            )
            if exchanges and not first_date:
                first_date = date
            all_exchanges.extend(exchanges)

    n_transcripts = 0
    for start in range(0, len(all_exchanges), EXCHANGES_PER_TRANSCRIPT):
        batch = all_exchanges[start : start + EXCHANGES_PER_TRANSCRIPT]
        write_transcript(
            name,
            {"q": "Twitter interlocutors", "a": target},
            first_date or "unknown",
            "https://x.com",
            batch,
        )
        n_transcripts += 1

    hx.ok(
        f"imported {n_docs} grounding documents into {corpus_path} and "
        f"{len(all_exchanges)} exchanges into {n_transcripts} transcript file(s)"
    )
    if n_docs and not all_exchanges:
        hx.warn(
            f"no question/answer pairs were found. Those threads are\n"
            f"  probably {handle or target} talking to themselves, which makes good\n"
            f"  grounding material but no interview data. Try --replies-only\n"
            f"  sources, or add conversation examples with `raft interactive`."
        )


def normalize_messages(
    messages: List[Dict[str, Any]], target: str = ""
) -> List[Dict[str, Any]]:
    """
    Map ariadne's raft message shape onto the role/content/name shape the
    conversation structurer expects.

    Ariadne puts the body under `text` and assigns roles by position: the
    thread *root* is "assistant" and every later reply is "participant".
    That position says nothing about who we are emulating, so when the
    target appears in the thread their messages become the answers and
    everyone else's become questions. Only when the target is absent (or
    unnamed) do we fall back to ariadne's positional roles.
    """
    handle = target.lstrip("@").lower()

    def author_of(message: Dict[str, Any]) -> str:
        return (message.get("username") or message.get("author") or "").lstrip("@")

    target_present = handle and any(
        author_of(message).lower() == handle for message in messages
    )

    normalized: List[Dict[str, Any]] = []
    for message in messages:
        author = author_of(message)
        if target_present:
            role = "assistant" if author.lower() == handle else "user"
        else:
            role = "assistant" if message.get("role") == "assistant" else "user"
        normalized.append(
            {
                "role": role,
                "name": author,
                "content": message.get("content") or message.get("text") or "",
            }
        )
    return normalized


def run_tweet_mode(name: str = "", target: str = "", standalone: bool = True) -> None:
    """
    Run the interactive tweet-mode flow end to end.

    Args:
        name (str, optional): Dataset name; asked interactively if empty.
        target (str, optional): Person to emulate; asked if empty.
        standalone (bool): Print next steps and offer to chunk/embed.
            False when called as one step of `raft interactive`.
    """
    if standalone:
        hx.banner("tweets -> persona dataset, via ariadne")
    ariadne = load_ariadne()

    options = ask_build_options()
    if not target:
        target = ask(
            "Target to emulate (handle or name)",
            options.get("target_user") or options.get("for_user") or None,
        )
    if not name:
        name = ask("Dataset name", target.lstrip("@").lower())

    hx.step("reconstructing tweet threads with ariadne")
    result = ariadne.build(**options)

    for warning in result.warnings[:10]:
        hx.say(f"note: {warning}")
    if not result.conversations:
        bail("ariadne reconstructed no conversations from that source")

    documents = result.raft_documents()
    hx.ok(f"reconstructed {len(documents)} thread(s)")
    result.save_cache()

    import_documents(name, documents, target)

    if not standalone:
        return

    hx.step("next steps")
    hx.say(f"  raft chunk {name}   # chunk the grounding corpus")
    hx.say(f"  raft embed {name}   # embed + store in chromadb")
    hx.say(f"  raft ft:gen {name}  # generate the finetune dataset")
    hx.say(f"  raft ft:run {name}  # run the finetune")
    if confirm("Run chunk + embed now?", default=False):
        from . import embeddings_helpers, files_helper

        files_helper.chunker(name)
        embeddings_helpers.store_grounding_embeddings(name)
