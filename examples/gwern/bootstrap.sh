#!/usr/bin/env bash
# gwern, from LessWrong alone: the 200 newest comment threads as dated
# conversations; posts, quick takes and every older comment as grounding.
#
#   examples/gwern/bootstrap.sh              # creates ./gwern
#   examples/gwern/bootstrap.sh ~/personas/gwern
#   CONVERSATIONS=0 examples/gwern/bootstrap.sh   # every thread
#
# The import needs no API key (LessWrong's GraphQL API is public). What
# comes after -- chunk, embed, ft:gen, ft:run -- needs an OpenAI key or the
# RAFT_*_BASE_URL endpoints; see README.md next to this script.
set -euo pipefail

dir="${1:-gwern}"
raft init "$dir"
cd "$dir"
if [ -n "$(ls -A conversations 2>/dev/null)" ]; then
  echo "$dir already has conversations; the importer appends, so start from an empty project." >&2
  exit 1
fi

raft lesswrong --user gwern --conversations "${CONVERSATIONS:-200}"

cat <<EOF

$dir is ready. From inside it:

  raft chunk && raft embed      # the corpus into the memory store
  raft ft:gen --thinking        # recall + reasoning in <think>, one example per exchange
  raft ft:run --model Qwen/Qwen3.8-27B --provider runpod --method qlora \\
    --gpu-count 2 --vram-gb 94 --gradient-accumulation 4 --epochs 2 --max-length 8192
  raft comment --source <url>   # or: raft comment --web 8090, raft serve

or simply: raft interactive
EOF
