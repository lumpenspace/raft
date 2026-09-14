"""
Tweet mode: build a raft dataset from tweets via ariadne
(https://github.com/lumpenspace/ariadne).

Uses ariadne's Python API directly, so the tweet documents come back as
data rather than through a file on disk: thread texts become grounding
documents (data/{name}.jsonl, ready for `raft chunk`) and reply branches
become q/a transcripts (ready for `raft ft:gen`).
"""

from typing import Any, Dict, List, Optional

from .convo_structurer import messages_to_exchanges, write_transcript
from . import hx
from .interactive import ask, ask_path, bail, choose, confirm
from .project import DatasetLike, DatasetPaths, dataset_paths
from .sources import in_window, iso_date  # noqa: F401  (iso_date re-exported; used below)
from . import state

ARIADNE_INSTALL_HINT = (
    "ariadne is not installed. Install it with:\n"
    "  pip install ariadne-x"
)


def load_ariadne():
    """Import ariadne, or exit with an install hint."""
    try:
        import ariadne
    except ImportError:
        bail(ARIADNE_INSTALL_HINT)
    if not hasattr(ariadne, "build"):
        bail(
            "the installed ariadne is too old for raft 2.0 (no Python API).\n"
            "Upgrade with: pip install -U ariadne-x"
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

    if confirm(
        "Also use the Community Archive? (completes cross-account threads, no key)",
        default=True,
    ):
        options["community_archive"] = True

    key = ask("twitterapi.io API key (paid; empty to skip)", "")
    if key:
        options["twitterapi_key"] = key

    return options


def select_documents(
    documents: List[Dict[str, Any]], since: str = "", until: str = "", limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """The shared fetch filters on ariadne documents: the window, then the newest `limit`."""
    dated = lambda doc: iso_date((doc.get("metadata") or {}).get("target_created_at"))  # noqa: E731
    kept = [doc for doc in documents if in_window(dated(doc), since, until)]
    if len(kept) < len(documents):
        hx.say(f"  {len(documents) - len(kept)} thread(s) outside {since or '...'}..{until or '...'} left out")
    if limit:
        kept = sorted(kept, key=dated, reverse=True)[:limit]
    return kept


def import_documents(
    dataset: DatasetLike,
    documents: List[Dict[str, Any]],
    target: str,
    role: str = "auto",
    since: str = "",
    until: str = "",
    limit: Optional[int] = None,
) -> None:
    """
    Import ariadne raft documents as grounding corpus + transcripts.

    Args:
        name (str): Dataset name (data/{name}*).
        documents (List[Dict[str, Any]]): raft.documents.v1 objects.
        target (str): Handle/name of the person being emulated.
        since, until, limit: the shared fetch window and cap (threads,
            newest first).
    """
    from .sources import append_corpus_records

    if role not in ("auto", "conversation", "corpus"):
        raise ValueError(f"unknown source role: {role}")
    documents = select_documents(documents, since, until, limit)
    paths = dataset_paths(dataset)
    # Ariadne reports usernames without the leading @; match on the bare handle.
    handle = target.lstrip("@")
    seen_ids = set()
    records: List[Dict[str, Any]] = []

    # One transcript per thread, each with its own date: what the persona
    # may recall while replying is everything dated before it.
    threads: List[Dict[str, Any]] = []

    for doc in documents:
        meta = doc.get("metadata", {})
        doc_id = doc.get("id") or str(meta.get("tweet_ids"))
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        text = (doc.get("text") or "").strip()
        date = iso_date(meta.get("target_created_at"))
        url = meta.get("target_url") or ""
        exchanges, questioner = messages_to_exchanges(
            normalize_messages(doc.get("messages") or [], handle), handle
        )
        destination = document_dataset_role(doc, exchanges) if role == "auto" else role

        if destination == "corpus" and text:
            records.append(
                {
                    "title": f"tweet thread {meta.get('target_id', doc_id)}",
                    "link": url,
                    "date": date or "unknown",
                    "content": text,
                }
            )

        elif destination == "conversation":
            if not exchanges:
                hx.warn(
                    f"conversation row {doc_id} had no usable target exchange; skipped"
                )
                continue
            network = "Bluesky" if "bsky.app" in url else "X"
            threads.append(
                {
                    "date": date or "unknown",
                    "url": url or ("https://bsky.app" if network == "Bluesky" else "https://x.com"),
                    "questioner": questioner or f"{network} interlocutors",
                    "context": f"a reply thread on {network}",
                    "exchanges": exchanges,
                }
            )

    n_docs = append_corpus_records(paths, records)

    threads.sort(key=lambda t: t["date"] if t["date"] != "unknown" else "")
    n_exchanges = 0
    for thread in threads:
        write_transcript(
            paths,
            {"q": thread["questioner"], "a": target},
            thread["date"],
            thread["url"],
            thread["exchanges"],
            context=thread["context"],
        )
        n_exchanges += len(thread["exchanges"])

    hx.ok(
        f"imported {n_docs} grounding documents into {paths.corpus_path} and "
        f"{n_exchanges} exchanges into {len(threads)} conversation(s)"
    )
    if n_docs and not threads:
        hx.warn(
            f"no question/answer pairs were found. Those threads are\n"
            f"  probably {handle or target} talking to themselves, which makes good\n"
            f"  grounding material but no interview data. Try --replies-only\n"
            f"  sources, or add conversation examples with `raft interactive`."
        )


def document_dataset_role(
    document: Dict[str, Any], exchanges: List[List[str]]
) -> str:
    """Choose exactly one Raft destination for an Ariadne document.

    New Ariadne rows explicitly carry ``metadata.dataset_role``. Older
    ``raft.documents.v1`` rows do not, so exchanges remain the compatibility
    signal: rows with a usable exchange are conversations; everything else is
    grounding corpus.
    """
    metadata = document.get("metadata") or {}
    explicit = metadata.get("dataset_role") or document.get("dataset_role")
    if explicit in ("conversation", "corpus"):
        return str(explicit)
    return "conversation" if exchanges else "corpus"


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


def build_options_for(user: str = "", archive: str = "", since: str = "") -> Dict[str, Any]:
    """
    The non-interactive X options behind `raft fetch tweets --user` /
    `--archive`: a public handle fetched live, or an archive export (its
    owner's tweets, or --user's). The Community Archive is always on; a
    twitterapi.io key comes from ariadne's own TWITTERAPI_IO_KEY.
    """
    options: Dict[str, Any] = {"allow_empty": True, "community_archive": True}
    if archive:
        options["archive"] = archive
        if user:
            options["for_user"] = user.lstrip("@")
        else:
            options["all_loaded"] = True
    else:
        options["target_user"] = user.lstrip("@")
    if since:
        options["since"] = since
    return options


def _gather_x(
    ariadne, dataset: DatasetLike, default_handle: str, role: str = "auto",
    options: Dict[str, Any] | None = None, until: str = "", limit: Optional[int] = None,
) -> int:
    """Run the X/Twitter branch and import its documents. Returns doc count."""
    options = options or ask_build_options()
    paths = dataset_paths(dataset)
    if paths.project:
        options["cache"] = str(paths.ariadne_cache_path)
    hx.step("reconstructing X threads with ariadne")
    result = ariadne.build(**options)
    for warning in result.warnings[:10]:
        hx.say(f"note: {warning}")
    documents = result.raft_documents()
    if not documents:
        hx.warn("no X conversations were reconstructed")
        return 0
    hx.ok(f"reconstructed {len(documents)} X thread(s)")
    if paths.project:
        result.save_cache(paths.ariadne_cache_path)
    else:
        result.save_cache()
    # ariadne >= 0.6 records what it could not fetch in its cache.
    unresolved = getattr(result, "unresolved_ids", lambda: [])()
    if unresolved:
        hx.say(
            f"{len(unresolved)} referenced tweet(s) could not be fetched; "
            "recover them later with `ariadne cache retry`, then re-run tweet mode"
        )
    handle = options.get("target_user") or options.get("for_user") or default_handle
    import_documents(paths, documents, handle, role=role, since=options.get("since", ""), until=until, limit=limit)
    return len(documents)


def _gather_bluesky(
    ariadne, dataset: DatasetLike, default_handle: str, role: str = "auto",
    handle: str = "", since: str = "", until: str = "", limit: Optional[int] = None,
) -> int:
    """Run the Bluesky branch and import its documents. Returns doc count."""
    if not hasattr(ariadne, "build_bluesky"):
        hx.warn(
            "the installed ariadne is too old for Bluesky -- upgrade with:\n"
            "  pip install -U ariadne-x"
        )
        return 0
    if not handle:
        handle = ask(
            "Bluesky handle (e.g. alice.bsky.social)",
            default_handle.lstrip("@") if "." in default_handle else None,
        )
        since = ask("Only posts on or after [YYYY-MM-DD] (empty = all)", "")
    hx.step("reconstructing Bluesky threads")
    result = ariadne.build_bluesky(handle, since=since or None, allow_empty=True)
    for warning in result.warnings[:10]:
        hx.say(f"note: {warning}")
    documents = result.raft_documents()
    if not documents:
        hx.warn("no Bluesky conversations were reconstructed")
        return 0
    hx.ok(f"reconstructed {len(documents)} Bluesky thread(s)")
    import_documents(dataset, documents, handle.lstrip("@"), role=role, since=since, until=until, limit=limit)
    return len(documents)


NETWORKS = ("x", "bluesky", "both")


def guess_network(user: str, archive: str) -> str:
    """A dotted handle (alice.bsky.social) is Bluesky; anything else, or an archive, is X."""
    if archive:
        return "x"
    return "bluesky" if "." in user.lstrip("@") else "x"


def run_tweet_mode(
    dataset: DatasetLike | str = "",
    target: str = "",
    standalone: bool = True,
    role: str = "auto",
    user: str = "",
    archive: str = "",
    network: str = "",
    since: str = "",
    until: str = "",
    limit: Optional[int] = None,
) -> None:
    """
    Run the tweet-mode flow end to end.

    Interactive by default. With `user` (a public X handle, or a dotted
    Bluesky handle) or `archive` (an X export) it asks nothing: that is
    `raft fetch tweets --user <handle>`. `network` (x, bluesky, both) is
    guessed from the handle when not given; `since`, `until` and `limit`
    are the shared fetch window and cap.

    Args:
        name (str, optional): Dataset name; asked interactively if empty.
        target (str, optional): Person to emulate; asked if empty.
        standalone (bool): Print next steps and offer to chunk/embed.
            False when called as one step of `raft interactive`.
    """
    if standalone:
        hx.banner("posts -> persona dataset, via ariadne")
    ariadne = load_ariadne()
    scripted = bool(user or archive)
    if network and network not in NETWORKS:
        raise ValueError(f"unknown network {network!r}: x, bluesky or both")
    if scripted and not network:
        network = guess_network(user, archive)
    if network == "both" and (archive or "." not in user):
        raise ValueError("--network both needs a dotted Bluesky handle in --user and no --archive")

    if not network:
        pick = choose(
            "Which network(s) should the dataset draw from?",
            ["X / Twitter", "Bluesky", "both -- merge into one dataset"],
            default=0,
        )
        network = NETWORKS[pick]
    want_x = network in ("x", "both")
    want_bsky = network in ("bluesky", "both")

    paths: DatasetPaths | None = dataset_paths(dataset) if dataset else None
    if paths and paths.project and not target:
        target = state.load_meta(paths).get("target", "")
    if not target and scripted:
        target = user.lstrip("@") or archive
    if not target:
        target = ask("Target to emulate (handle or name)")
    if paths is None:
        paths = dataset_paths(ask("Dataset name", target.lstrip("@").lower()))
    if paths.project and state.load_meta(paths).get("target") != target:
        state.update_meta(paths, target=target)

    total = 0
    if want_x:
        options = build_options_for(user, archive, since) if scripted else None
        total += _gather_x(ariadne, paths, target, role=role, options=options, until=until, limit=limit)
    if want_bsky:
        total += _gather_bluesky(
            ariadne, paths, target, role=role,
            handle=user.lstrip("@") if scripted else "", since=since, until=until, limit=limit,
        )

    if total == 0:
        bail("no conversations were reconstructed from the chosen source(s)")

    if not standalone:
        return

    hx.step("next steps")
    has_corpus = bool(state.dataset_status(paths)["corpus_docs"])
    if has_corpus:
        hx.say(f"  {paths.command('chunk')}   # chunk the grounding corpus")
        hx.say(f"  {paths.command('embed')}   # embed + store in chromadb")
    else:
        hx.say("Conversations only: no chunking or embedding needed.")
    hx.say(f"  {paths.command('ft:gen')}  # generate the finetune dataset")
    hx.say(f"  {paths.command('ft:run')}  # run the finetune")
    if has_corpus and confirm("Run chunk + embed now?", default=False):
        from . import embeddings_helpers, files_helper

        files_helper.chunker(paths)
        embeddings_helpers.store_grounding_embeddings(paths)
