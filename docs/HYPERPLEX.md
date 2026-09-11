# hyperplex terminal brand

raft, [ariadne](https://github.com/lumpenspace/ariadne), and
[opbdh](https://github.com/lumpenspace/opbdh) share one terminal language, so the
interactive flows and their outputs read as a family. The implementation is a small
`hx.py` module vendored identically in each repo — only the three identity constants
change.

## Identity

| tool | sigil | accent | why |
| --- | --- | --- | --- |
| raft | `≋` | mint `#67dfc2` | waves under the raft |
| ariadne | `⌇` | amber `#fbbf24` | the golden thread |
| opbdh | `◉` | red `#ef4444` | HAL's eye (pre-dates the brand; kept as canon) |

Unifying chrome: hyperplex blue `#4d7cff` — the accent of
[hyperplex.org](https://hyperplex.org), so terminal and web share one palette —
and the mark `⟡`. raft's mint is the site's linked-node color.

## Conventions

- **Banner** (interactive entry points only): `{sigil} {tool}  ·  ⟡ hyperplex  · tagline`
  over a thin chrome-blue rule.
- **Prompts**: accent `»` marker; defaults dim in brackets; choices numbered with
  accent numerals.
- **Choosing**: on a tty, `choose` menus navigate with ↑/↓ or j/k (wrapping),
  a digit jumps to that option, Enter confirms — and the menu collapses to a
  one-line record of the choice so transcripts stay readable. Off-tty (pipes,
  scripts) it falls back to the numbered prompt, so scripted input keeps
  working. The key handling is hand-rolled termios + ANSI repaint — no extra
  dependency, no screen takeover. opbdh's questionary prompts already behave
  this way natively; `hx.questionary_style()` skins them to the palette.
- **Status**: steps `◆` in accent; results `✓` green / `!` yellow / `✗` red; secondary
  detail dim.
- **stdout is sacred**: all chrome goes to **stderr**, so piped output stays
  machine-clean.
- **Text arguments are data**: every helper escapes its message/prompt/option
  strings before interpolating them into rich markup (`hx.esc`) — file paths,
  model ids, and scraped document text must never reach the markup parser
  (a stray `[/]` would raise, `[int]` would vanish).
- **Degrade**: plain ASCII, no color, when stderr is not a tty or `NO_COLOR` /
  `HYPERPLEX_PLAIN` is set.

## Library

[rich](https://github.com/Textualize/rich) — and deliberately not blessed or Textual:
these tools are linear wizards and streaming pipelines, and a full-screen TUI event
loop would break pipeability and log capture. rich gives color, layout, and graceful
degradation without owning the terminal.
