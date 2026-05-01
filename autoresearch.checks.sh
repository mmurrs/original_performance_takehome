#!/bin/bash
set -e
cd /Users/matt/perf_takehome

# Ensure tests/ is unchanged vs origin/main (anti-cheat)
DIFF=$(git diff origin/main -- tests/)
if [ -n "$DIFF" ]; then
  echo "FAIL: tests/ modified vs origin/main"
  echo "$DIFF" | head -40
  exit 1
fi
echo "OK: tests/ unchanged"
