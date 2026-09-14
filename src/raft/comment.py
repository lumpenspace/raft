"""
`raft comment`: hand the persona a post -- a URL, a LessWrong / EA Forum
post, or a text file -- and get its comment: recall from the store, the
think block opened with that recall exactly as in training, then the
persona's own reasoning and reply.

Generation runs one of two ways:

- mlx: the MLX-converted model (RAFT_MLX_MODEL, or --model) runs in this
  process through mlx-lm; the prompt is rendered with the model's chat
  template and the recall is prefilled into the think block.
- completions: an OpenAI-compatible /v1/completions endpoint
  (mlx_lm.server, vLLM, llama-server) at OPENAI_BASE_URL is handed the
  same prefilled raw prompt; the tokenizer for rendering comes from
  RAFT_MLX_MODEL / --model.

`raft comment --web 8090` serves a small page that does the same.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from . import hx, state
from .memories import MemoryManager, MetaDataKeyEnum
from .oai_finetune import recall_text
from .project import DatasetLike, dataset_paths
from .prompt_manager import PromptManager

# A post is handed over whole up to this many characters; recall is queried
# with its opening, the way training saw a post (see lesswrong.post_stub).
BODY_CHARS = 30000
STUB_CHARS = 1500
MAX_TOKENS = 1400

FORUM_HOSTS = ("lesswrong.com", "alignmentforum.org", "forum.effectivealtruism.org", "greaterwrong.com")
POST_FIELDS = "_id title pageUrl postedAt user { displayName username } contents { markdown }"

_MODELS: Dict[str, Any] = {}


# -- the post ------------------------------------------------------------------


def fetch_post(source: str) -> Dict[str, str]:
    """
    A post as {"title", "author", "date", "url", "body"} from a local file,
    a LessWrong-family URL (through the GraphQL API, so the body is
    markdown) or any other web page.
    """
    path = Path(source).expanduser()
    if path.is_file():
        return {"title": path.stem.replace("-", " ").replace("_", " "), "author": "the author", "date": "",
                "url": str(path), "body": path.read_text(encoding="utf-8", errors="ignore")}
    url = urlparse(source)
    if not url.scheme:
        raise ValueError(f"{source!r} is neither a file nor a URL")
    host = url.netloc.lower().removeprefix("www.")
    match = re.match(r"^/posts/([A-Za-z0-9]+)", url.path)
    if any(host.endswith(h) for h in FORUM_HOSTS) and match:
        from .lesswrong import graphql

        base = f"{url.scheme}://{url.netloc}"
        if host.endswith("greaterwrong.com"):
            base = "https://www.lesswrong.com"
        data = graphql(base, f'{{ post(input:{{selector:{{_id:"{match.group(1)}"}}}}) {{ result {{ {POST_FIELDS} }} }} }}')
        post = (data.get("post") or {}).get("result")
        if not post:
            raise ValueError(f"no post {match.group(1)} on {base}")
        user = post.get("user") or {}
        return {"title": post.get("title") or "", "author": user.get("displayName") or user.get("username") or "the author",
                "date": (post.get("postedAt") or "")[:10], "url": post.get("pageUrl") or source,
                "body": ((post.get("contents") or {}).get("markdown") or "").strip()}
    from . import sources

    page = sources.extract_page(sources._http_get(source), source)
    return {"title": page["title"], "author": "the author", "date": page["date"], "url": source, "body": page["content"]}


def post_question(post: Dict[str, str], chars: int = BODY_CHARS) -> str:
    body = post["body"].strip()
    if len(body) > chars:
        body = body[:chars].rstrip() + " [...]"
    return f'"{post["title"]}" by {post["author"]}:\n\n{body}'


def post_stub(post: Dict[str, str]) -> str:
    return post_question(post, STUB_CHARS)


# -- recall and the prompt ---------------------------------------------------------


def recall_for(dataset: DatasetLike, post: Dict[str, str], target: str) -> str:
    """What the persona recalls for this post: dated to today, never stored."""
    today = datetime.now().date().isoformat()
    manager = MemoryManager(dataset, {
        MetaDataKeyEnum.DATE: today,
        MetaDataKeyEnum.PARTICIPANTS: {"q": post["author"], "a": target},
        MetaDataKeyEnum.URL: post["url"],
    })
    return manager.get_similar_and_summarize([post_stub(post), ""], "", store=False)


def conversation(post: Dict[str, str], target: str, thinking: bool, memories: str = "") -> List[Dict[str, str]]:
    """The messages, framed like a training example: one system message, the post."""
    today = datetime.now().date().isoformat()
    system = dict(PromptManager().get_interview_system_message(
        post["author"], target, today, context=f'a comment thread under the post "{post["title"]}"', thinking=thinking,
    ))
    messages = [system]
    if memories and not thinking:
        messages.append({"role": "system", "content": f"Earlier writing of yours that may bear on this:\n{memories}"})
    messages.append({"role": "user", "content": post_question(post)})
    return messages


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]], memories: str, thinking: bool) -> str:
    """
    The raw prompt: the chat template's generation prompt, then -- for a
    thinking persona -- the think block opened with the recall, exactly the
    shape it was trained on, so the model continues with its reasoning.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not thinking:
        return prompt
    if not prompt.rstrip().endswith("<think>"):
        prompt = prompt + "<think>\n"
    return prompt + recall_text(memories) + "\n\n"


RECALL_LINE = re.compile(r"^\s*from \d{4}-\d{2}-\d{2}:", re.MULTILINE)


def split_output(text: str, prefilled: int = 0) -> Dict[str, Any]:
    """
    {"thinking", "reply", "invented"}: the reasoning the model wrote, its
    reply, and any recall lines beyond the `prefilled` ones -- those the
    model made up, so they are flagged and never presented as memories.
    """
    thinking, sep, reply = text.partition("</think>")
    if not sep:
        thinking, reply = "", text
    thinking = thinking.replace("<think>", "", 1)
    invented = []
    for match in list(RECALL_LINE.finditer(thinking))[prefilled:]:
        end = thinking.find("\n", match.end())
        line = thinking[match.start(): end if end != -1 else None].strip()
        rest = thinking[end + 1: thinking.find("\n", end + 1) if end != -1 else 0].strip() if end != -1 else ""
        invented.append((line + " " + rest).strip()[:160])
    return {"thinking": thinking.strip(), "reply": reply.strip(), "invented": invented}


# -- generation ------------------------------------------------------------------


def load_mlx(model_dir: str):
    """(model, tokenizer) through mlx-lm, cached per directory."""
    if model_dir not in _MODELS:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError("mlx-lm is not installed: pip install mlx-lm (Apple silicon only)") from e
        hx.step(f"loading {model_dir}")
        _MODELS[model_dir] = load(model_dir)
    return _MODELS[model_dir]


def generate_mlx(model_dir: str, prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = 0.7) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load_mlx(model_dir)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temperature, top_p=0.9))


def generate_completions(model: str, prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = 0.7) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "x")
    response = client.completions.create(model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
    return response.choices[0].text or ""


def tokenizer_for(model_dir: str):
    """A tokenizer that renders chat templates, from the model directory."""
    try:
        return load_mlx(model_dir)[1]
    except RuntimeError:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir)


def comment_on(dataset: DatasetLike, source: str, model: str = "", backend: str = "auto",
               max_tokens: int = MAX_TOKENS, temperature: float = 0.7) -> Dict[str, Any]:
    """
    The persona's comment on a post. Returns the post, what it recalled,
    its thinking (with any invented recall flagged), the reply and timings.
    """
    paths = dataset_paths(dataset)
    meta = state.load_meta(paths)
    target = meta.get("target") or paths.name
    thinking = bool(meta.get("thinking"))
    model = model or os.environ.get("RAFT_MLX_MODEL", "")
    if not model:
        raise ValueError("no model: pass --model or set RAFT_MLX_MODEL to the MLX-converted persona")
    if backend == "auto":
        backend = "completions" if os.environ.get("OPENAI_BASE_URL") and not Path(model).expanduser().is_dir() else "mlx"

    started = time.time()
    post = fetch_post(source)
    hx.step(f'"{post["title"]}" by {post["author"]} ({len(post["body"])} chars)')
    memories = recall_for(paths, post, target)
    recalled = time.time()
    messages = conversation(post, target, thinking, memories)
    prompt = render_prompt(tokenizer_for(model), messages, memories, thinking)
    if backend == "mlx":
        text = generate_mlx(model, prompt, max_tokens, temperature)
    else:
        text = generate_completions(model, prompt, max_tokens, temperature)
    opening = recall_text(memories)
    output = split_output(
        (f"<think>\n{opening}\n\n{text}") if thinking else text, prefilled=len(RECALL_LINE.findall(opening))
    )
    return {
        "post": {k: v for k, v in post.items() if k != "body"} | {"chars": len(post["body"])},
        "recall": memories.strip(),
        "thinking": output["thinking"],
        "invented": output["invented"],
        "reply": output["reply"],
        "seconds": {"recall": round(recalled - started, 1), "generate": round(time.time() - recalled, 1)},
    }


# -- the web page --------------------------------------------------------------------

PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{target} · raft</title>
<style>
  :root {{ --bg:#0f1117; --fg:#e6e6e6; --dim:#8a8f98; --mint:#67dfc2; --chrome:#4d7cff; --card:#161a23; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-weight: 600; font-size: 20px; margin: 0 0 4px; }} h1 span {{ color: var(--mint); }} h1 small {{ color: var(--chrome); font-weight: 400; }}
  p.lead {{ color: var(--dim); margin: 0 0 20px; }}
  form {{ display: grid; gap: 10px; }}
  input, textarea {{ width: 100%; box-sizing: border-box; background: var(--card); color: var(--fg); border: 1px solid #262b36; border-radius: 8px; padding: 10px 12px; font: inherit; }}
  textarea {{ min-height: 110px; resize: vertical; }}
  button {{ justify-self: start; background: var(--mint); color: #0b1a16; border: 0; border-radius: 8px; padding: 10px 18px; font: inherit; font-weight: 600; cursor: pointer; }}
  button[disabled] {{ opacity: .5; cursor: wait; }}
  section {{ margin-top: 28px; }} section h2 {{ font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); margin: 0 0 8px; }}
  .card {{ background: var(--card); border: 1px solid #262b36; border-radius: 10px; padding: 14px 16px; white-space: pre-wrap; }}
  .think {{ color: var(--dim); font-size: 14px; }} .invented {{ color: #f2b36b; }} .meta {{ color: var(--dim); font-size: 13px; margin-top: 8px; }}
  #status {{ color: var(--mint); min-height: 1.5em; }}
</style>
<main>
  <h1><span>≋</span> {target} <small>· ⟡ hyperplex</small></h1>
  <p class="lead">Give {target} a post — a URL (LessWrong, EA Forum, any page) or pasted text — and get the comment: what came to mind from earlier writing, the thinking, the reply.</p>
  <form id="f">
    <input id="url" placeholder="https://www.lesswrong.com/posts/…" autocomplete="off">
    <textarea id="text" placeholder="…or paste the post here (optional title on the first line)"></textarea>
    <button id="go">Ask {target}</button>
    <div id="status"></div>
  </form>
  <section id="out" hidden>
    <h2>Recall</h2><div class="card think" id="recall"></div>
    <section><h2>Thinking</h2><div class="card think" id="thinking"></div><div class="meta" id="invented"></div></section>
    <section><h2>Reply</h2><div class="card" id="reply"></div><div class="meta" id="meta"></div></section>
  </section>
</main>
<script>
const f = document.getElementById('f'), go = document.getElementById('go'), st = document.getElementById('status');
f.onsubmit = async (e) => {{
  e.preventDefault(); go.disabled = true; st.textContent = 'recalling, then thinking — a minute or two…';
  document.getElementById('out').hidden = true;
  try {{
    const r = await fetch('/comment', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ url: document.getElementById('url').value.trim(), text: document.getElementById('text').value }}) }});
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    document.getElementById('recall').textContent = d.recall || '(nothing came to mind)';
    document.getElementById('thinking').textContent = d.thinking;
    document.getElementById('invented').innerHTML = d.invented.length ? '<span class="invented">Invented while thinking (not real memories): ' + d.invented.map(s => s.replace(/</g, '&lt;')).join(' · ') + '</span>' : '';
    document.getElementById('reply').textContent = d.reply;
    document.getElementById('meta').textContent = `“${{d.post.title}}” by ${{d.post.author}} · recall ${{d.seconds.recall}}s · generation ${{d.seconds.generate}}s`;
    document.getElementById('out').hidden = false; st.textContent = '';
  }} catch (err) {{ st.textContent = 'error: ' + err.message; }}
  go.disabled = false;
}};
</script>
"""


def handle_comment(dataset: DatasetLike, payload: Dict[str, Any], model: str = "") -> Dict[str, Any]:
    """One request of the page: a URL, or pasted text written to a temp file."""
    url = (payload.get("url") or "").strip()
    text = (payload.get("text") or "").strip()
    if url:
        return comment_on(dataset, url, model=model)
    if not text:
        raise ValueError("give a URL or some text")
    import tempfile

    title, _, body = text.partition("\n")
    if len(title) > 120 or not body.strip():
        title, body = "a post", text
    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix=re.sub(r"[^A-Za-z0-9]+", "-", title)[:60] + "-", delete=False) as f:
        f.write(body.strip())
    try:
        return comment_on(dataset, f.name, model=model)
    finally:
        os.unlink(f.name)


def serve_web(dataset: DatasetLike, port: int, model: str = "", host: str = "0.0.0.0") -> None:
    """Serve the page; one comment at a time (the model is one process)."""
    paths = dataset_paths(dataset)
    target = state.load_meta(paths).get("target") or paths.name
    page = PAGE.format(target=target).encode("utf-8")
    import threading

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: N802
            hx.say(fmt % args)

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/comment":
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                with lock:
                    result = handle_comment(paths, payload, model=model)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
            except Exception as e:  # noqa: BLE001 -- the page shows the error
                hx.warn(f"comment failed: {e}")
                self._send(200, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

    hx.banner(f"{target} comments on posts")
    hx.say(f"serving on http://{host}:{port}/ -- empty line does nothing here; Ctrl-C stops")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def run_comment(dataset: DatasetLike, source: str = "", model: str = "", web: int = 0) -> None:
    """The `raft comment` action: one comment on stdout, or the web page."""
    if web:
        serve_web(dataset, web, model=model)
        return
    if not source:
        hx.fail("give a post with --source <url or file>, or --web <port>")
        sys.exit(2)
    result = comment_on(dataset, source, model=model)
    hx.say("recall:\n" + (result["recall"] or "(nothing came to mind)"))
    hx.say("thinking:\n" + result["thinking"])
    for line in result["invented"]:
        hx.warn(f"invented while thinking: {line}")
    print(result["reply"])
