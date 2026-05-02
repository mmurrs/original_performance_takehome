#!/bin/bash
set -e
cd /Users/matt/perf_takehome

# 1. Ensure tests/ is unchanged vs origin/main (anti-cheat)
DIFF=$(git diff origin/main -- tests/)
if [ -n "$DIFF" ]; then
  echo "FAIL: tests/ modified vs origin/main"
  echo "$DIFF" | head -40
  exit 1
fi

# 2. Run the submission correctness test (8 random seeds) to catch seed-123-only bugs
python3 -c "
import sys, unittest
sys.path.insert(0, 'tests')
from submission_tests import CorrectnessTests
suite = unittest.TestLoader().loadTestsFromTestCase(CorrectnessTests)
runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
res = runner.run(suite)
if not res.wasSuccessful():
    print('FAIL: submission correctness test failed')
    for (t, e) in res.errors + res.failures:
        print(t, e[:200])
    sys.exit(1)
print('OK: tests/ unchanged, correctness across seeds verified')
"
