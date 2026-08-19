"""
Turn conversation examples and text sources -- structured or not -- into
the datasets raft consumes:

- transcripts: data/{name}_transcript_{i}.json
  {"participants": {"q": ..., "a": ...}, "date": ..., "url": ...,
   "exchanges": [[question, answer], ...]}
- grounding corpus: data/{name}.jsonl, one {"title", "link", "date",
  "content"} object per line (the input of `raft chunk`).

Unstructured files are converted with an LLM; structured ones (raft
transcripts, chat-message arrays, grounding jsonl) are recognised and
passed through.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from . import hx

STRUCTURER_MODEL = os.environ.get("RAFT_LLM_MODEL", "gpt-4o")

# Keep single LLM calls comfortably inside the context window.
MAX_STRUCTURE_CHARS = 24000

STRUCTURER_SYSTEM_PROMPT = """\
You extract question/answer exchanges from a raw conversation record
(interview, chat log, podcast transcript, email thread...).

The target is the person being emulated: their utterances are answers,
everyone else's are questions. Merge consecutive utterances by the same
side. Drop utterances that are not part of a question/answer flow
(stage directions, ads, timestamps).

Reply with JSON only, in this exact shape:
{
  "participants": {"q": "<questioner name>", "a": "<target name>"},
  "date": "<ISO date if inferable, else null>",
  "url": "<source url if present, else null>",
  "exchanges": [["<question>", "<answer>"], ...]
}"""


def transcript_path(name: str, index: int) -> str:
    """Return the path of transcript #index for a dataset name."""
    return f"data/{name}_transcript_{index}.json"


def next_transcript_index(name: str) -> int:
    """Return the first unused transcript index for a dataset name."""
    i = 1
    while os.path.exists(transcript_path(name, i)):
        i += 1
    return i


def write_transcript(
    name: str,
    participants: Dict[str, str],
    date: str,
    url: str,
    exchanges: List[List[str]],
    index: Optional[int] = None,
) -> str:
    """
    Write a transcript file in the format generate_finetune expects.

    Returns:
        str: The path written.
    """
    if index is None:
        index = next_transcript_index(name)
    path = transcript_path(name, index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "participants": participants,
                "date": date,
                "url": url,
                "exchanges": exchanges,
            },
            f,
            indent=2,
        )
    return path


def is_transcript(data: Any) -> bool:
    """Check whether parsed JSON already matches the transcript schema."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("exchanges"), list)
        and isinstance(data.get("participants"), dict)
        and {"q", "a"} <= set(data["participants"])
    )


def parse_json_or_jsonl(raw: str) -> Any:
    """
    Parse a string as JSON, falling back to JSONL (returning the list of
    parsed lines). Returns None if it is neither.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return None
    return parsed or None


def is_message_array(value: Any) -> bool:
    """Check whether a value looks like a chat-message array."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(m, dict) for m in value)
        and any("role" in m for m in value)
    )


def find_message_arrays(data: Any) -> List[List[Dict[str, Any]]]:
    """
    Collect every chat-message array in a parsed conversation file.

    Handles a bare message array, {"messages": [...]}, ariadne's
    {"conversations": [{"messages": [...]}, ...]}, and JSONL files whose
    lines are any of the above.

    Returns:
        List[List[Dict[str, Any]]]: One entry per conversation found.
    """
    if is_message_array(data):
        return [data]
    if isinstance(data, dict):
        if is_message_array(data.get("messages")):
            return [data["messages"]]
        if isinstance(data.get("conversations"), list):
            found = []
            for convo in data["conversations"]:
                found.extend(find_message_arrays(convo))
            return found
        return []
    if isinstance(data, list):
        found = []
        for item in data:
            found.extend(find_message_arrays(item))
        return found
    return []


def messages_to_exchanges(
    messages: List[Dict[str, Any]], target: str
) -> Tuple[List[List[str]], str]:
    """
    Convert a chat-message array (role/content, optionally name) into
    [question, answer] exchanges, treating the target's (or assistant's)
    turns as answers.

    Returns:
        Tuple[List[List[str]], str]: The exchanges and the most frequent
        questioner name found (empty string if none).
    """

    def is_answer(msg: Dict[str, Any]) -> bool:
        if target and msg.get("name", "").lower() == target.lower():
            return True
        return msg.get("role") == "assistant"

    exchanges: List[List[str]] = []
    q_names: Dict[str, int] = {}
    pending_q: List[str] = []
    for msg in messages:
        content = str(msg.get("content", "")).strip()
        if not content or msg.get("role") == "system":
            continue
        if is_answer(msg):
            if pending_q:
                exchanges.append(["\n".join(pending_q), content])
                pending_q = []
            elif exchanges:
                exchanges[-1][1] += f"\n{content}"
        else:
            pending_q.append(content)
            q_name = msg.get("name", "")
            if q_name:
                q_names[q_name] = q_names.get(q_name, 0) + 1

    questioner = max(q_names, key=lambda k: q_names[k]) if q_names else ""
    return exchanges, questioner


def structure_raw_conversation(raw_text: str, target: str) -> Dict[str, Any]:
    """
    Use an LLM to turn an unstructured conversation into transcript JSON.

    Args:
        raw_text (str): The raw conversation text.
        target (str): Name of the person being emulated.

    Returns:
        Dict[str, Any]: participants/date/url/exchanges dict.
    """
    client = OpenAI()
    if len(raw_text) > MAX_STRUCTURE_CHARS:
        hx.warn(
            f"source is {len(raw_text)} chars; structuring the first "
            f"{MAX_STRUCTURE_CHARS} -- split long files for full coverage"
        )
        raw_text = raw_text[:MAX_STRUCTURE_CHARS]
    response = client.chat.completions.create(
        model=STRUCTURER_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": STRUCTURER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"The target is: {target}\n\n---\n\n{raw_text}",
            },
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    if not isinstance(data.get("exchanges"), list) or not data["exchanges"]:
        raise ValueError("the LLM could not extract any exchanges")
    data.setdefault("participants", {})
    data["participants"].setdefault("q", "Interviewer")
    data["participants"].setdefault("a", target)
    return data


def import_conversation_file(name: str, path: str, target: str) -> str:
    """
    Import one conversation file (structured or not) as a transcript.

    Recognises: raft transcript JSON, chat-message arrays (bare list or
    {"messages": [...]} / {"conversations": [...]}), anything else is
    treated as raw text and structured with an LLM.

    Returns:
        str: The transcript path written.
    """
    with open(path) as f:
        raw = f.read()

    data = parse_json_or_jsonl(raw) if path.endswith((".json", ".jsonl")) else None
    today = datetime.now(timezone.utc).date().isoformat()

    if is_transcript(data):
        return write_transcript(
            name,
            data["participants"],
            data.get("date") or today,
            data.get("url") or path,
            data["exchanges"],
        )

    conversations = find_message_arrays(data)
    if conversations:
        exchanges: List[List[str]] = []
        questioners: Dict[str, int] = {}
        for messages in conversations:
            got, questioner = messages_to_exchanges(messages, target)
            exchanges.extend(got)
            if questioner:
                questioners[questioner] = questioners.get(questioner, 0) + 1
        if not exchanges:
            raise ValueError(f"{path}: no question/answer exchanges found")
        top = max(questioners, key=lambda k: questioners[k]) if questioners else ""
        meta = data if isinstance(data, dict) else {}
        return write_transcript(
            name,
            {"q": top or "Interviewer", "a": target},
            meta.get("date") or today,
            meta.get("url") or path,
            exchanges,
        )

    hx.step(f"{path}: unstructured, extracting exchanges with {STRUCTURER_MODEL}")
    structured = structure_raw_conversation(raw, target)
    return write_transcript(
        name,
        structured["participants"],
        structured.get("date") or today,
        structured.get("url") or path,
        structured["exchanges"],
    )


def import_text_source_file(name: str, path: str) -> int:
    """
    Append a text source (structured grounding jsonl or raw text) to the
    grounding corpus data/{name}.jsonl.

    Returns:
        int: Number of documents appended.
    """
    corpus_path = f"data/{name}.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(path) as f:
        raw = f.read()

    records: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        try:
            parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if all(isinstance(p, dict) and "content" in p for p in parsed):
                records = parsed
        except json.JSONDecodeError:
            records = []

    if not records:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        title = re.sub(r"[_-]+", " ", os.path.splitext(os.path.basename(path))[0])
        records = [
            {
                "title": title,
                "link": path,
                "date": mtime.date().isoformat(),
                "content": raw,
            }
        ]

    with open(corpus_path, "a") as f:
        for record in records:
            record.setdefault("title", "untitled")
            record.setdefault("link", path)
            record.setdefault("date", datetime.now(timezone.utc).date().isoformat())
            f.write(json.dumps(record) + "\n")
    return len(records)
