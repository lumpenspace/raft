"""raft comment: fetching a post, recall-prefilled prompts, generation backends, the page. Offline."""
from unittest.mock import patch

import pytest

from raft import cli, comment, state
from raft.project import initialize_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = initialize_project(tmp_path / "persona")[0]
    state.update_meta(project, target="Sam", thinking=True)
    return project


class Tok:
    """A chat template that ends its generation prompt with an open think block."""

    def __init__(self, think=True):
        self.think = think

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n" + ("<think>\n" if self.think else "")


def test_fetch_post_from_file_forum_and_page(tmp_path):
    f = tmp_path / "my-post.txt"
    f.write_text("Body of the post.")
    post = comment.fetch_post(str(f))
    assert post["title"] == "my post" and post["body"] == "Body of the post."

    def graphql(base, query):
        assert base == "https://www.lesswrong.com" and 'selector:{_id:"abc123"}' in query
        return {"post": {"result": {"title": "T", "pageUrl": "https://www.lesswrong.com/posts/abc123/t", "postedAt": "2026-09-10T00:00:00Z",
                                    "user": {"displayName": "lumpenspace"}, "contents": {"markdown": "Markdown body."}}}}

    with patch("raft.lesswrong.graphql", side_effect=graphql):
        post = comment.fetch_post("https://www.lesswrong.com/posts/abc123/some-slug?commentId=x")
    assert post == {"title": "T", "author": "lumpenspace", "date": "2026-09-10", "url": "https://www.lesswrong.com/posts/abc123/t", "body": "Markdown body."}
    with patch("raft.sources._http_get", return_value="<html><head><title>Page</title></head><body><article>Prose.</article></body></html>"):
        post = comment.fetch_post("https://example.com/essay")
    assert post["title"] == "Page" and post["body"] == "Prose." and post["author"] == "the author"
    with pytest.raises(ValueError):
        comment.fetch_post("not a url")


def test_question_and_stub_truncate():
    post = {"title": "T", "author": "A", "date": "", "url": "u", "body": "x" * 5000}
    assert comment.post_question(post).startswith('"T" by A:\n\nxxx') and comment.post_question(post).endswith("x")
    stub = comment.post_stub(post)
    assert stub.endswith("[...]") and len(stub) < 1600


def test_conversation_and_prefilled_prompt():
    post = {"title": "T", "author": "A", "date": "", "url": "u", "body": "Body."}
    messages = comment.conversation(post, "Sam", thinking=True, memories="from 2023: I said so.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert 'a comment thread under the post "T"' in messages[0]["content"] and "Before you reply, think" in messages[0]["content"]
    prompt = comment.render_prompt(Tok(), messages, "from 2023: I said so.", thinking=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\nRecalling what I have written before:\nfrom 2023: I said so.\n\n")
    prompt = comment.render_prompt(Tok(think=False), messages, "", thinking=True)
    assert prompt.endswith("assistant\n<think>\nNothing I have written before bears on this directly.\n\n")
    plain = comment.conversation(post, "Sam", thinking=False, memories="m")
    assert [m["role"] for m in plain] == ["system", "system", "user"]
    assert comment.render_prompt(Tok(think=False), plain, "m", thinking=False).endswith("assistant\n")


def test_split_output_flags_invented_recall():
    text = "<think>\nRecalling what I have written before:\nfrom 2023-06-06: \n I said X.\nfrom 2026-03-24: \n I never said this.\nSo I think Y.\n</think>\n\nThe reply."
    out = comment.split_output(text, prefilled=1)
    assert out["reply"] == "The reply." and out["thinking"].startswith("Recalling")
    assert out["invented"] == ["from 2026-03-24: I never said this."]
    assert comment.split_output("no think at all")["reply"] == "no think at all"


def test_comment_on_prefills_recall_and_reports(project, tmp_path, monkeypatch):
    model_dir = tmp_path / "mlx"
    model_dir.mkdir()
    monkeypatch.setenv("RAFT_MLX_MODEL", str(model_dir))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    post = {"title": "T", "author": "A", "date": "2026-09-10", "url": "u", "body": "Body."}
    prompts = []

    def generate(model, prompt, max_tokens, temperature):
        prompts.append(prompt)
        return "from 2024-01-01: \n made up.\nBecause of X.\n</think>\n\nMy comment."

    with patch("raft.comment.fetch_post", return_value=post), patch("raft.comment.recall_for", return_value="from 2023-06-06: \n I said X."), \
         patch("raft.comment.tokenizer_for", return_value=Tok()), patch("raft.comment.generate_mlx", side_effect=generate) as mlx:
        result = comment.comment_on(project, "https://x/posts/1")
    assert mlx.call_args.args[0] == str(model_dir)
    assert prompts[0].endswith("<think>\nRecalling what I have written before:\nfrom 2023-06-06: \n I said X.\n\n")
    assert result["reply"] == "My comment." and result["recall"].startswith("from 2023-06-06")
    assert result["invented"] == ["from 2024-01-01: made up."] and result["post"]["title"] == "T"
    # an endpoint and a non-directory model id mean the completions backend
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    with patch("raft.comment.fetch_post", return_value=post), patch("raft.comment.recall_for", return_value=""), \
         patch("raft.comment.tokenizer_for", return_value=Tok()), patch("raft.comment.generate_completions", return_value="ok</think>\n\nR") as rest:
        result = comment.comment_on(project, "u", model="served-model")
    assert rest.called and result["reply"] == "R" and result["thinking"].startswith("Nothing I have written")


def test_handle_comment_takes_url_or_text(project):
    with patch("raft.comment.comment_on", return_value={"reply": "r"}) as on:
        comment.handle_comment(project, {"url": "https://x/p"})
        assert on.call_args.args[1] == "https://x/p"
        comment.handle_comment(project, {"text": "A title\nThe body of the post."})
        path = on.call_args.args[1]
        assert path.endswith(".txt") and "A-title" in path
    with pytest.raises(ValueError):
        comment.handle_comment(project, {})


def test_cli_routes_comment(project, monkeypatch):
    monkeypatch.chdir(project.root)
    monkeypatch.setattr("sys.argv", ["raft", "comment", "--source", "https://x/p", "--model", "m"])
    with patch("raft.comment.comment_on", return_value={"recall": "", "thinking": "t", "invented": [], "reply": "the reply"}) as on:
        cli.main()
    assert on.call_args.args[1] == "https://x/p" and on.call_args.kwargs["model"] == "m"
    monkeypatch.setattr("sys.argv", ["raft", "comment", "--web", "8090"])
    with patch("raft.comment.serve_web") as web:
        cli.main()
    assert web.call_args.args[1] == 8090
    assert "Ask Sam" in comment.PAGE.format(target="Sam")
