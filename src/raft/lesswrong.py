"""
LessWrong -- and any other ForumMagnum forum, such as the EA Forum -- as
a source, through the public GraphQL API. (The Alignment Forum shares
LessWrong's database, so LessWrong covers it.)

Posts by the target become grounding documents. Comments become
conversation examples: for each comment by the target, whatever it
replies to is the questioner's side -- the post (title, author and its
opening) for a top-level comment, the chain of parent comments
otherwise -- and the target's comment is the answer. Where the target
and an interlocutor go back and forth in one branch, the branch becomes
a multi-turn exchange, and each comment by the target is an answer
exactly once. Top-level comments on the target's own posts (quick
takes, replies to nobody) are the target talking to themselves and go
to grounding instead.

A prolific commenter's history is also their past writing: with
`older_comments_as_grounding`, every comment not used as a conversation
answer becomes a dated grounding document (with a line of what it
replied to), so later conversations can recall it.

The API caps offset pagination at 2000 rows, so listings page by date
(`before`) instead, and parent comments are fetched in aliased batches.
"""

import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests

from . import hx
from .convo_structurer import messages_to_exchanges, write_transcript
from .interactive import ask, choose, confirm
from .project import DatasetLike, dataset_paths
from .sources import USER_AGENT, append_corpus_records, iso_date

FORUMS = [
    ("LessWrong (also covers the Alignment Forum)", "https://www.lesswrong.com"),
    ("EA Forum", "https://forum.effectivealtruism.org"),
]

PAGE_SIZE = 500
BATCH_SIZE = 40
REQUEST_DELAY = 0.5

# A top-level comment answers the post: title, author and this much of
# the body stand in for the questioner.
POST_EXCERPT_CHARS = 1500
# Interlocutors can be long-winded; the persona's own replies are never cut.
QUESTION_CHARS = 4000
# A comment kept as grounding opens with this much of what it replied to.
REPLY_CONTEXT_CHARS = 300

USER_FIELDS = "_id username displayName slug postCount commentCount"
POST_FIELDS = (
    "_id title pageUrl postedAt baseScore draft shortform isEvent "
    "userId user { username displayName } contents { markdown }"
)
COMMENT_FIELDS = (
    "_id postId parentCommentId postedAt baseScore deleted pageUrl "
    "userId user { username displayName } "
    "post { _id title pageUrl userId user { username displayName } } "
    "contents { markdown }"
)


def graphql(base_url: str, query: str) -> Dict[str, Any]:
    """
    Run one query and return its `data`. Raises on a request failure or
    a query that produced no data at all; per-alias errors (a missing
    document) leave that alias null and are the caller's to handle.
    """
    response = requests.post(
        f"{base_url}/graphql",
        json={"query": query},
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    errors = payload.get("errors") or []
    if errors and not any(value for value in data.values()):
        raise ValueError("; ".join(str(e.get("message", "?")) for e in errors[:3]))
    return data


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def resolve_user(base_url: str, handle: str) -> Dict[str, Any]:
    """The user record for a username / profile slug."""
    cleaned = handle.strip().lstrip("@")
    candidates = []
    for slug in (cleaned.lower(), cleaned.lower().replace("_", "-"), cleaned.lower().replace(" ", "-")):
        if slug not in candidates:
            candidates.append(slug)
    for slug in candidates:
        try:
            data = graphql(base_url, f"{{ user(input:{{selector:{{slug:{_quote(slug)}}}}}) {{ result {{ {USER_FIELDS} }} }} }}")
        except ValueError:
            continue
        user = (data.get("user") or {}).get("result")
        if user:
            return user
    raise ValueError(f"no user {handle!r} on {base_url}")


def iter_by_date(
    base_url: str, collection: str, view: str, user_id: str, fields: str
) -> Iterator[List[Dict[str, Any]]]:
    """
    A user's documents in a collection as pages, newest first, paging by
    `before` the last seen timestamp (offset paging stops at 2000).
    """
    seen = set()
    listed = 0
    before = ""
    while True:
        terms = f'view:"{view}", userId:{_quote(user_id)}, limit:{PAGE_SIZE}'
        if before:
            terms += f", before:{_quote(before)}"
        data = graphql(base_url, f"{{ {collection}(input:{{terms:{{{terms}}}}}) {{ results {{ {fields} }} }} }}")
        page = (data.get(collection) or {}).get("results") or []
        fresh = [item for item in page if item["_id"] not in seen]
        if not fresh:
            return
        seen.update(item["_id"] for item in fresh)
        listed += len(fresh)
        hx.say(f"  {listed} {collection} listed")
        yield fresh
        if len(page) < PAGE_SIZE:
            return
        before = page[-1]["postedAt"]
        time.sleep(REQUEST_DELAY)


def list_by_date(
    base_url: str,
    collection: str,
    view: str,
    user_id: str,
    fields: str,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Every document of a user in a collection (or the newest max_items)."""
    items: List[Dict[str, Any]] = []
    for page in iter_by_date(base_url, collection, view, user_id, fields):
        items.extend(page)
        if max_items and len(items) >= max_items:
            return items[:max_items]
    return items


def fetch_documents(
    base_url: str, kind: str, ids: Iterable[str], fields: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Comments or posts by id, in aliased batches. Missing (deleted) ids
    map to None.
    """
    wanted = sorted(set(ids))
    found: Dict[str, Optional[Dict[str, Any]]] = {}
    for start in range(0, len(wanted), BATCH_SIZE):
        batch = wanted[start : start + BATCH_SIZE]
        query = " ".join(
            f"d{i}: {kind}(input:{{selector:{{_id:{_quote(doc_id)}}}}}) {{ result {{ {fields} }} }}"
            for i, doc_id in enumerate(batch)
        )
        data = graphql(base_url, f"{{ {query} }}")
        for i, doc_id in enumerate(batch):
            found[doc_id] = (data.get(f"d{i}") or {}).get("result")
        if start + BATCH_SIZE < len(wanted):
            time.sleep(REQUEST_DELAY)
    return found


def text_of(document: Optional[Dict[str, Any]]) -> str:
    return (((document or {}).get("contents") or {}).get("markdown") or "").strip()


def author_of(document: Optional[Dict[str, Any]]) -> str:
    user = (document or {}).get("user") or {}
    return user.get("displayName") or user.get("username") or "someone"


def usable_post(post: Dict[str, Any]) -> bool:
    """A real post with a body: not a draft, event, or the shortform container."""
    return not (post.get("draft") or post.get("shortform") or post.get("isEvent")) and bool(text_of(post))


def post_record(post: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": post.get("title") or post["_id"],
        "link": post.get("pageUrl") or "",
        "date": iso_date(post.get("postedAt")),
        "content": text_of(post),
    }


def post_stub(post: Dict[str, Any]) -> str:
    """What a top-level comment replies to: the post's title, author and opening."""
    body = text_of(post)
    if len(body) > POST_EXCERPT_CHARS:
        body = body[:POST_EXCERPT_CHARS].rstrip() + " [...]"
    return f'"{post.get("title") or "untitled"}" by {author_of(post)}:\n\n{body}'.strip()


def resolve_ancestors(
    base_url: str, comments: List[Dict[str, Any]], known: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Every comment above the given comments in their threads, by id.
    `known` (comments and ancestors already fetched) is extended in place
    so successive pages only fetch what is new.
    """
    known = known if known is not None else {}
    known.update({c["_id"]: c for c in comments})
    missing = {c["parentCommentId"] for c in comments if c.get("parentCommentId")} - set(known)
    ancestors: Dict[str, Dict[str, Any]] = {}
    while missing:
        fetched = fetch_documents(base_url, "comment", missing, COMMENT_FIELDS)
        missing = set()
        for doc_id, doc in fetched.items():
            doc = doc or {"_id": doc_id, "deleted": True}
            ancestors[doc_id] = known[doc_id] = doc
            parent = doc.get("parentCommentId")
            if parent and parent not in known:
                missing.add(parent)
        hx.say(f"  {len(ancestors)} parent comment(s) resolved")
    return ancestors


def build_conversations(
    comments: List[Dict[str, Any]],
    target_id: str,
    ancestors: Dict[str, Dict[str, Any]],
    posts: Dict[str, Optional[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Turn the target's comments into exchange groups (one per thread
    branch, multi-turn where the branch alternates) and quick takes
    (top-level comments on the target's own posts).

    Newest comments are processed first, so a branch is emitted from its
    leaf, and every target comment is an answer exactly once.
    """
    by_id: Dict[str, Dict[str, Any]] = {c["_id"]: c for c in comments}
    by_id.update(ancestors)

    def chain(comment: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = [comment]
        visited = {comment["_id"]}
        while True:
            parent = by_id.get(path[-1].get("parentCommentId") or "")
            if parent is None or parent["_id"] in visited:
                break
            visited.add(parent["_id"])
            path.append(parent)
        return list(reversed(path))

    emitted = set()
    groups: List[Dict[str, Any]] = []
    quick_takes: List[Dict[str, Any]] = []
    for comment in sorted(comments, key=lambda c: c.get("postedAt") or "", reverse=True):
        if comment["_id"] in emitted or comment.get("deleted") or not text_of(comment):
            continue
        post = comment.get("post") or {}
        date = iso_date(comment.get("postedAt"))
        url = post.get("pageUrl") or comment.get("pageUrl") or ""
        if not comment.get("parentCommentId"):
            emitted.add(comment["_id"])
            if not post:
                continue  # a comment on a tag or wiki page: no thread to speak of
            if post.get("userId") == target_id:
                quick_takes.append(comment)
                continue
            full = posts.get(comment.get("postId") or "") or post
            groups.append({
                "date": date, "url": url, "title": post.get("title") or "", "questioner": author_of(full),
                "exchanges": [[post_stub(full), text_of(comment)]], "comment_ids": [comment["_id"]],
            })
            continue

        path = chain(comment)
        cut = 0
        for i, node in enumerate(path[:-1]):
            if node["_id"] in emitted:
                cut = i + 1
        path = path[cut:]
        messages = []
        for node in path:
            role = "assistant" if node.get("userId") == target_id else "user"
            content = text_of(node)
            if role == "user" and len(content) > QUESTION_CHARS:
                content = content[:QUESTION_CHARS].rstrip() + " [...]"
            messages.append({"role": role, "name": author_of(node), "content": content})
        for message in messages:
            if message["role"] == "user" and message["content"]:
                message["content"] = f'Re: "{post.get("title") or "untitled"}"\n\n{message["content"]}'
                break
        exchanges, questioner = messages_to_exchanges(messages, "")
        emitted.update(node["_id"] for node in path if node.get("userId") == target_id)
        if exchanges:
            # The branch is dated by the target's first reply in it: nothing
            # the persona recalls may postdate any answer it is trained on.
            dates = [iso_date(n.get("postedAt")) for n in path if n.get("userId") == target_id]
            groups.append({
                "date": min(d for d in dates if d) if any(dates) else date, "url": url,
                "title": post.get("title") or "", "questioner": questioner, "exchanges": exchanges,
                "comment_ids": [n["_id"] for n in path if n.get("userId") == target_id],
            })
    return groups, quick_takes


def _unique_titles(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Titles must be unique: the embedding store keys chunks by title and part."""
    used: Dict[str, int] = {}
    for record in records:
        used[record["title"]] = used.get(record["title"], 0) + 1
        if used[record["title"]] > 1:
            record["title"] = f"{record['title']} #{used[record['title']]}"
    return records


def quick_take_records(quick_takes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Grounding documents from quick takes, titles kept unique per day."""
    records = []
    for comment in sorted(quick_takes, key=lambda c: c.get("postedAt") or ""):
        date = iso_date(comment.get("postedAt")) or "undated"
        title = f"{(comment.get('post') or {}).get('title') or 'quick take'}, {date}"
        records.append({"title": title, "link": comment.get("pageUrl") or "", "date": iso_date(comment.get("postedAt")), "content": text_of(comment)})
    return _unique_titles(records)


def comment_records(
    comments: List[Dict[str, Any]], parents: Dict[str, Optional[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """
    Grounding documents from comments: each opens with a line of what it
    replied to (the parent comment, or the post's title), then the comment.
    """
    records = []
    for comment in sorted(comments, key=lambda c: c.get("postedAt") or ""):
        post = comment.get("post") or {}
        date = iso_date(comment.get("postedAt")) or "undated"
        parent = parents.get(comment.get("parentCommentId") or "")
        replied = text_of(parent) if parent else ""
        if replied:
            lead = " ".join(replied.split())
            if len(lead) > REPLY_CONTEXT_CHARS:
                lead = lead[:REPLY_CONTEXT_CHARS].rstrip() + " [...]"
            opening = f"Replying to {author_of(parent)} ({lead})"
        elif post:
            opening = f'Commenting on "{post.get("title") or "a post"}" by {author_of(post)}'
        else:
            opening = "Commenting on a wiki page"
        records.append({
            "title": f'comment on "{post["title"]}", {date}' if post.get("title") else f"wiki comment, {date}",
            "link": comment.get("pageUrl") or "",
            "date": iso_date(comment.get("postedAt")),
            "content": f"{opening}:\n\n{text_of(comment)}",
        })
    return _unique_titles(records)


def write_transcripts(
    dataset: DatasetLike, groups: List[Dict[str, Any]], forum_name: str, target_name: str
) -> int:
    """
    Write every thread branch as its own transcript, oldest first, dated
    and framed ("a LessWrong comment thread under the post ...").
    """
    article = "an" if forum_name[:1].lower() in "aeiou" else "a"
    for group in sorted(groups, key=lambda g: g["date"]):
        title = group.get("title") or "untitled"
        write_transcript(
            dataset,
            {"q": group.get("questioner") or f"{forum_name} commenters", "a": target_name},
            group["date"] or "unknown",
            group["url"],
            group["exchanges"],
            context=f'{article} {forum_name} comment thread under the post "{title}"',
        )
    return len(groups)


def import_lesswrong(
    dataset: DatasetLike,
    base_url: str,
    handle: str,
    role: str = "auto",
    max_conversations: Optional[int] = None,
    min_karma: Optional[int] = None,
    forum_name: str = "LessWrong",
    older_comments_as_grounding: bool = False,
) -> Dict[str, int]:
    """
    Import a user's forum activity.

    Args:
        role: "corpus" (posts and quick takes), "conversation" (comment
            threads), or "auto" (both).
        max_conversations: Stop after this many thread branches, newest
            first (comments are listed page by page until there are enough).
        min_karma: Skip comments scored below this.
        older_comments_as_grounding: Every comment not used as a
            conversation answer becomes a dated grounding document
            (needs a grounding role).

    Returns:
        {"documents", "exchanges", "transcripts"} counts.
    """
    if role not in ("auto", "conversation", "corpus"):
        raise ValueError(f"unknown source role: {role}")
    paths = dataset_paths(dataset)
    user = resolve_user(base_url, handle)
    target_name = user.get("displayName") or user.get("username") or handle
    hx.say(f"{target_name}: {user.get('postCount', '?')} post(s), {user.get('commentCount', '?')} comment(s)")
    summary = {"documents": 0, "exchanges": 0, "transcripts": 0}
    want_threads = role in ("auto", "conversation")

    if role in ("auto", "corpus"):
        hx.step("listing posts")
        posts = list_by_date(base_url, "posts", "userPosts", user["_id"], POST_FIELDS)
        summary["documents"] += append_corpus_records(paths, [post_record(p) for p in posts if usable_post(p)])

    grounding_comments = older_comments_as_grounding and role in ("auto", "corpus")
    hx.step("listing comments" + (" and the threads they reply to" if want_threads else ""))
    comments: List[Dict[str, Any]] = []
    known: Dict[str, Dict[str, Any]] = {}
    ancestors: Dict[str, Dict[str, Any]] = {}
    top_level_posts: Dict[str, Optional[Dict[str, Any]]] = {}
    groups: List[Dict[str, Any]] = []
    quick_takes: List[Dict[str, Any]] = []
    enough = False
    for page in iter_by_date(base_url, "comments", "allRecentComments", user["_id"], COMMENT_FIELDS):
        page = [c for c in page if not c.get("deleted") and text_of(c)]
        if min_karma is not None:
            page = [c for c in page if (c.get("baseScore") or 0) >= min_karma]
        comments.extend(page)
        if enough:
            continue  # only listing the rest for grounding
        if want_threads:
            ancestors.update(resolve_ancestors(base_url, page, known))
            post_ids = {
                c["postId"] for c in page
                if not c.get("parentCommentId") and c.get("postId") and (c.get("post") or {}).get("userId") != user["_id"]
            }
            top_level_posts.update(fetch_documents(base_url, "post", post_ids - set(top_level_posts), POST_FIELDS))
        groups, quick_takes = build_conversations(comments, user["_id"], ancestors, top_level_posts)
        if want_threads and max_conversations and len(groups) >= max_conversations:
            enough = True
            if not grounding_comments:
                break
    if max_conversations:
        groups = groups[:max_conversations]  # newest first
    if grounding_comments:
        used = {cid for g in groups for cid in g.get("comment_ids", [])} if want_threads else set()
        used.update(c["_id"] for c in quick_takes)
        older = [c for c in comments if c["_id"] not in used]
        hx.step(f"{len(older)} older comment(s) as grounding; fetching what they replied to")
        parents = dict(known)
        parents.update(fetch_documents(
            base_url, "comment",
            {c["parentCommentId"] for c in older if c.get("parentCommentId")} - set(parents), COMMENT_FIELDS,
        ))
        summary["documents"] += append_corpus_records(paths, comment_records(older, parents))

    if role in ("auto", "corpus") and quick_takes:
        summary["documents"] += append_corpus_records(paths, quick_take_records(quick_takes))
    if role in ("auto", "conversation") and groups:
        summary["transcripts"] = write_transcripts(paths, groups, forum_name, target_name)
        summary["exchanges"] = sum(len(g["exchanges"]) for g in groups)
    hx.ok(
        f"{summary['documents']} grounding document(s), {summary['exchanges']} exchange(s) "
        f"in {summary['transcripts']} conversation(s) from {forum_name}"
    )
    return summary


def run_lesswrong_source(dataset: DatasetLike, target: str, role: str = "auto") -> Dict[str, int]:
    """Ask which forum and user, then import."""
    forum = choose("Which forum?", [name for name, _ in FORUMS] + ["another ForumMagnum site"], default=0)
    if forum < len(FORUMS):
        forum_name, base_url = FORUMS[forum]
        forum_name = forum_name.split(" (")[0]
    else:
        base_url = ask("Forum base URL (e.g. https://www.lesswrong.com)").rstrip("/")
        forum_name = base_url.split("//")[-1]
    handle = ask("Username or profile slug", target.strip().lstrip("@").lower().replace(" ", "-"))
    max_conversations = min_karma = None
    if role != "corpus":
        raw = ask("Max conversations to import, newest first (empty = all)", "200")
        max_conversations = int(raw) if raw.strip().isdigit() else None
        raw = ask("Skip comments below this karma (empty = keep all)", "")
        min_karma = int(raw) if raw.strip().lstrip("-").isdigit() else None
    older = role != "conversation" and confirm(
        "Also use the comments beyond the conversations as grounding documents (their past writing)?", default=True
    )
    return import_lesswrong(
        dataset, base_url, handle, role=role, max_conversations=max_conversations, min_karma=min_karma,
        forum_name=forum_name, older_comments_as_grounding=older,
    )
