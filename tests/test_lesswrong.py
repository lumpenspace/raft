"""LessWrong source: listing, thread reconstruction, routing. Offline (GraphQL stubbed)."""
import json
import re
from unittest.mock import patch

import pytest

from raft import lesswrong, state
from raft.project import initialize_project

T, A, B = "user-t", "user-a", "user-b"


def user(uid, name):
    return {"_id": uid, "username": name.lower(), "displayName": name}


def post(pid, uid, title, body="", **extra):
    return {"_id": pid, "title": title, "pageUrl": f"https://lw/posts/{pid}", "postedAt": "2023-06-01T00:00:00.000Z",
            "baseScore": 10, "draft": None, "shortform": False, "isEvent": False, "userId": uid,
            "user": user(uid, uid[-1].upper()), "contents": {"markdown": body}, **extra}


POSTS = {
    "p1": post("p1", T, "Own post", "My own long post."),
    "p2": post("p2", T, "T's Shortform", "", shortform=True),
    "p3": post("p3", A, "A's post", "A" * 2000),
}


def comment(cid, uid, pid, parent, date, body, **extra):
    p = POSTS[pid] if pid else None
    return {"_id": cid, "postId": pid, "parentCommentId": parent, "postedAt": f"{date}T12:00:00.000Z", "baseScore": 5,
            "deleted": False, "pageUrl": f"https://lw/posts/{pid}?commentId={cid}", "userId": uid, "user": user(uid, uid[-1].upper()),
            "post": {"_id": p["_id"], "title": p["title"], "pageUrl": p["pageUrl"], "userId": p["userId"], "user": p["user"]} if p else None,
            "contents": {"markdown": body}, **extra}


TARGET_COMMENTS = [
    comment("c1", T, "p3", None, "2024-01-01", "Top-level take on A's post."),
    comment("c2", T, "p1", "a1", "2024-02-01", "Reply to A on my post."),
    comment("c3", T, "p1", "a2", "2024-03-01", "Second reply to A."),
    comment("c4", T, "p2", None, "2024-04-01", "A quick take."),
    comment("c5", T, "p3", None, "2024-05-01", "gone", deleted=True),
    comment("c6", T, None, None, "2024-06-01", "On a wiki page."),
]
OTHERS = {
    "a1": comment("a1", A, "p1", None, "2024-01-15", "A asks about the post."),
    "a2": comment("a2", A, "p1", "c2", "2024-02-15", "A follows up."),
}


def fake_graphql(base_url, query):
    if "user(input" in query:
        slug = re.search(r'slug:"([^"]+)"', query).group(1)
        if slug != "t":
            raise ValueError("app.missing_document")
        return {"user": {"result": {**user(T, "T"), "slug": "t", "postCount": 2, "commentCount": 6}}}
    if "posts(input" in query:
        uid = re.search(r'userId:"([^"]+)"', query).group(1)
        return {"posts": {"results": [] if "before:" in query else [p for p in POSTS.values() if p["userId"] == uid]}}
    if "comments(input" in query:
        return {"comments": {"results": [] if "before:" in query else TARGET_COMMENTS}}
    data = {}
    for alias, kind, doc_id in re.findall(r'(d\d+): (comment|post)\(input:\{selector:\{_id:"([^"]+)"', query):
        source = OTHERS if kind == "comment" else POSTS
        data[alias] = {"result": source.get(doc_id)}
    return data


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return initialize_project(tmp_path / "persona")[0]


@pytest.fixture(autouse=True)
def offline():
    with patch("raft.lesswrong.graphql", side_effect=fake_graphql), patch("raft.lesswrong.time.sleep"):
        yield


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_both_roles_split_posts_and_threads(project):
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="auto")
    assert summary == {"documents": 2, "exchanges": 3, "transcripts": 2}
    docs = read_jsonl(project.corpus_path)
    assert [d["title"] for d in docs] == ["Own post", "T's Shortform, 2024-04-01"]
    assert docs[0]["date"] == "2023-06-01" and docs[0]["link"] == "https://lw/posts/p1"
    assert docs[1]["content"] == "A quick take." and docs[1]["link"].endswith("commentId=c4")

    # oldest conversation first: the top-level comment answers the post's opening
    first = json.loads(project.transcript_path(1).read_text())
    assert first["participants"] == {"q": "A", "a": "T"}
    assert first["date"] == "2024-01-01" and first["url"] == "https://lw/posts/p3"
    assert first["context"] == 'a LessWrong comment thread under the post "A\'s post"'
    [(question, answer)] = first["exchanges"]
    assert question.startswith('"A\'s post" by A:\n\nAAAA') and question.endswith("[...]")
    assert len(question) < 1600 and answer == "Top-level take on A's post."
    # the back-and-forth on the own post is one multi-turn conversation, emitted
    # once and dated by the target's first reply in it
    second = json.loads(project.transcript_path(2).read_text())
    assert second["participants"] == {"q": "A", "a": "T"} and second["date"] == "2024-02-01"
    assert second["exchanges"] == [
        ['Re: "Own post"\n\nA asks about the post.', "Reply to A on my post."],
        ["A follows up.", "Second reply to A."],
    ]


def test_conversation_role_never_touches_grounding(project):
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="conversation")
    assert summary["transcripts"] == 2 and summary["documents"] == 0
    assert not project.corpus_path.exists()


def test_corpus_role_writes_posts_and_quick_takes_only(project):
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="corpus")
    assert summary == {"documents": 2, "exchanges": 0, "transcripts": 0}
    assert state.dataset_status(project)["transcripts"] == 0


def test_filters_by_karma_and_conversation_count(project):
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="conversation", max_conversations=1)
    assert summary["exchanges"] == 2  # the newest branch only: the two-turn thread on the own post
    transcript = json.loads(project.transcript_path(1).read_text())
    assert transcript["exchanges"][0][1] == "Reply to A on my post."
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="conversation", min_karma=6)
    assert summary["exchanges"] == 0


def test_conversation_cap_stops_paging(project):
    pages = []

    def graphql(base_url, query):
        if "user(input" in query:
            return {"user": {"result": {**user(T, "T"), "slug": "t"}}}
        if "comments(input" in query:
            pages.append(query)
            n = len(pages)
            return {"comments": {"results": [comment(f"x{n}", T, "p3", None, f"2024-0{n}-01", f"take {n}")]}}
        return {alias: {"result": POSTS.get(doc_id)} for alias, _, doc_id in re.findall(r'(d\d+): (comment|post)\(input:\{selector:\{_id:"([^"]+)"', query)}

    with patch("raft.lesswrong.graphql", side_effect=graphql), patch.object(lesswrong, "PAGE_SIZE", 1):
        groups_before = len(pages)
        summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="conversation", max_conversations=2)
    assert summary["exchanges"] == 2 and len(pages) - groups_before == 2


def test_resolve_user_tries_slug_variants():
    calls = []

    def graphql(base_url, query):
        calls.append(re.search(r'slug:"([^"]+)"', query).group(1))
        if calls[-1] == "eliezer-yudkowsky":
            return {"user": {"result": user("e", "Eliezer")}}
        raise ValueError("app.missing_document")

    with patch("raft.lesswrong.graphql", side_effect=graphql):
        assert lesswrong.resolve_user("https://lw", "@Eliezer_Yudkowsky")["displayName"] == "Eliezer"
        with pytest.raises(ValueError, match="no user"):
            lesswrong.resolve_user("https://lw", "nobody")
    assert calls == ["eliezer_yudkowsky", "eliezer-yudkowsky", "nobody"]


def test_list_by_date_pages_with_before_cursor():
    rows = [{"_id": f"c{i}", "postedAt": f"2024-01-{i:02d}T00:00:00.000Z"} for i in range(9, 0, -1)]
    queries = []

    def graphql(base_url, query):
        queries.append(query)
        before = re.search(r'before:"([^"]+)"', query)
        page = [r for r in rows if not before or r["postedAt"] < before.group(1)]
        return {"comments": {"results": page[:4]}}

    with patch("raft.lesswrong.graphql", side_effect=graphql), patch.object(lesswrong, "PAGE_SIZE", 4), \
         patch("raft.lesswrong.time.sleep"):
        got = lesswrong.list_by_date("https://lw", "comments", "allRecentComments", T, "_id postedAt")
        assert [r["_id"] for r in got] == [f"c{i}" for i in range(9, 0, -1)]
        assert len(queries) == 3 and 'before:"2024-01-06' in queries[1]
        assert len(lesswrong.list_by_date("https://lw", "comments", "allRecentComments", T, "_id postedAt", max_items=5)) == 5


def test_fetch_documents_batches_and_marks_missing():
    with patch.object(lesswrong, "BATCH_SIZE", 2):
        found = lesswrong.fetch_documents("https://lw", "comment", ["a1", "zz", "a2"], lesswrong.COMMENT_FIELDS)
    assert found["a1"]["_id"] == "a1" and found["a2"]["_id"] == "a2" and found["zz"] is None


def test_write_transcripts_one_dated_conversation_each(project):
    groups = [
        {"date": "2024-01-02", "url": "u2", "title": "Second", "questioner": "Bo", "exchanges": [["q3", "a3"]]},
        {"date": "2024-01-01", "url": "u1", "title": "First", "questioner": "", "exchanges": [["q1", "a1"], ["q2", "a2"]]},
    ]
    assert lesswrong.write_transcripts(project, groups, "EA Forum", "T") == 2
    first = json.loads(project.transcript_path(1).read_text())
    second = json.loads(project.transcript_path(2).read_text())
    assert first["date"] == "2024-01-01" and first["participants"]["q"] == "EA Forum commenters"
    assert first["context"] == 'an EA Forum comment thread under the post "First"'
    assert second["participants"] == {"q": "Bo", "a": "T"} and second["url"] == "u2"


def test_older_comments_become_grounding_with_their_reply_context(project):
    summary = lesswrong.import_lesswrong(project, "https://lw", "T", role="auto", max_conversations=1,
                                         older_comments_as_grounding=True)
    # the newest branch (c2, c3) is the one conversation; c1 is older and becomes grounding, c4 stays a quick take
    assert summary["transcripts"] == 1 and summary["exchanges"] == 2
    docs = read_jsonl(project.corpus_path)
    assert sorted(d["title"] for d in docs) == sorted([
        "Own post", "T's Shortform, 2024-04-01", 'comment on "A\'s post", 2024-01-01', "wiki comment, 2024-06-01"])
    older = next(d for d in docs if d["title"].startswith("comment on"))
    assert older["content"].startswith('Commenting on "A\'s post" by A:\n\nTop-level take')
    assert older["date"] == "2024-01-01" and older["link"].endswith("commentId=c1")
    # conversation answers never leak into grounding
    assert not any("Reply to A on my post" in d["content"] for d in docs)


def test_reply_context_comes_from_the_parent():
    parent = comment("a9", A, "p1", None, "2024-01-01", "A long parent " * 40)
    child = comment("c9", T, "p1", "a9", "2024-01-02", "My answer.")
    [record] = lesswrong.comment_records([child], {"a9": parent})
    assert record["content"].startswith("Replying to A (A long parent")
    assert "[...]" in record["content"] and record["content"].endswith("):\n\nMy answer.")
    assert len(record["content"]) < 400
