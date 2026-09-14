"""Capture real wizard ANSI output in a PTY using an isolated demo dataset.

No source fetching, API requests, or training jobs are performed.
Run with the repository's Python environment. Replay the output in a terminal
emulator for screenshots (the capture itself is not a fabricated UI).
"""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "playwright" / "wizard"
OUTPUT.mkdir(parents=True, exist_ok=True)

SETUP = '''
from raft import hx, flows, state, serve
from raft.project import initialize_project
from raft.convo_structurer import write_transcript
from raft.sources import append_corpus_records
p, _ = initialize_project("demo-persona")
state.update_meta(p, target="Alex Rivers")
hx.banner("build a persona dataset and finetune it")
'''
DATA = '''
write_transcript(p, {"q":"Interviewer","a":"Alex Rivers"}, "2024-01-01", "demo", [["What matters most?","Curiosity, and the time to follow it."]])
append_corpus_records(p, [{"title":"A note on curiosity","link":"demo:essay","date":"2023-01-01","content":"Start with the question that will not leave you alone."}])
'''
SCENES = {
    "sources": (SETUP + 'flows.plan_sources()', []),
    "roles": (SETUP + 'flows.plan_sources()', [b'1\r']),
    "phases": (SETUP[:SETUP.index("hx.banner")] + DATA + 'flows.run_interactive(p)', []),
    "prep": (SETUP + DATA + 'flows.phase_prep(p)', []),
    "eval": (SETUP + DATA + 'state.record_finetuned_model(p,"ft:demo-persona","openai")\nflows.phase_eval(p)', []),
    "serve": (SETUP + 'state.record_finetuned_model(p,"ft:demo-persona","openai")\nserve.run_serve(p, standalone=False)', []),
    "train": (SETUP + DATA + 'p.finetune_openai_path.write_text("demo fixture")\nflows.phase_train(p)', []),
    "conversations-only": (SETUP + 'write_transcript(p, {"q":"Interviewer","a":"Alex Rivers"}, "2024-01-01", "demo", [["Why?","Because."]])\nflows.phase_prep(p)', []),
}


def read_available(master, duration=0.7):
    data = b""
    until = time.monotonic() + duration
    while time.monotonic() < until:
        if select.select([master], [], [], 0.1)[0]:
            try:
                data += os.read(master, 65536)
            except OSError:
                break
    return data


for name, (code, keys) in SCENES.items():
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 106, 0, 0))
    env = {**os.environ, "TERM":"xterm-256color", "COLORTERM":"truecolor", "PYTHONPATH": str(ROOT / "src"), "OPENAI_API_KEY":"screenshot-fixture-not-a-real-key"}
    env.pop("NO_COLOR", None)
    env.pop("HYPERPLEX_PLAIN", None)
    with tempfile.TemporaryDirectory(prefix="raft-capture-") as cwd:
        process = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=cwd, env=env,
                                   stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        data = read_available(master, 2)
        for key in keys:
            os.write(master, key)
            data += read_available(master)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        os.close(master)
    (OUTPUT / f"{name}.json").write_text(json.dumps({"ansi":data.decode(),"cols":106,"rows":28}))
    print(name)
