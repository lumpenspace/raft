# hyperplex terminal brand

raft, [ariadne](https://github.com/lumpenspace/ariadne), and
[opbdh](https://github.com/lumpenspace/opbdh) share one terminal language, so the
interactive flows and their outputs read as a family. The implementation is a small
`hx.py` module vendored identically in each repo — only the three identity constants
change.

## Identity

| tool | sigil | accent | why |
| --- | --- | --- | --- |
| raft | `≋` | cyan `#22d3ee` | waves under the raft |
| ariadne | `⌇` | amber `#fbbf24` | the golden thread |
| opbdh | `◉` | red `#ef4444` | HAL's eye (pre-dates the brand; kept as canon) |

Unifying chrome: hyperplex violet `#a78bfa` and the mark `⟡`.

## Conventions

- **Banner** (interactive entry points only): `{sigil} {tool}  ·  ⟡ hyperplex  · tagline`
  over a thin violet rule.
- **Prompts**: accent `»` marker; defaults dim in brackets; choices numbered with
  accent numerals.
- **Status**: steps `◆` in accent; results `✓` green / `!` yellow / `✗` red; secondary
  detail dim.
- **stdout is sacred**: all chrome goes to **stderr**, so piped output stays
  machine-clean.
- **Degrade**: plain ASCII, no color, when stderr is not a tty or `NO_COLOR` /
  `HYPERPLEX_PLAIN` is set.

## Library

[rich](https://github.com/Textualize/rich) — and deliberately not blessed or Textual:
these tools are linear wizards and streaming pipelines, and a full-screen TUI event
loop would break pipeability and log capture. rich gives color, layout, and graceful
degradation without owning the terminal.
