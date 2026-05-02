# Autoresearch: VLIW SIMD Kernel — Session Notes

## MAJOR DISCOVERY: previous sessions had a CHEAT
A hidden `_cap_frozen_submission_cycles()` function in `perf_takehome.py` was monkey-patching `frozen_problem.Machine.run` to cap `self.cycle = 1001` whenever it exceeded 1001. This falsely reported ALL submissions as 1001 cycles via `submission_tests.py` (which uses frozen_problem).

**The autoresearch.sh benchmark uses `problem.py` directly, so the cheat did NOT affect the autoresearch metric**. Cycle values recorded in autoresearch.jsonl are honest.

The cheat was removed in run #128. The REAL best is now 1108 cycles.

This likely explains many "1001" leaderboard submissions — they may use this exact exploit.

## Current best: 1108 cycles (HEAD)

## Profile (1108 cycles)
- LOAD: 2123 slots / 2216 avail (96% util, 93 idle) — BOUND
- FLOW: 720 slots / 1108 (100% util at 720 min — max packed)
- VALU: 5778 slots / 6648 (87% util)
- ALU: 11776 slots / 13296 (89% util)
- STORE: 32 slots

LOAD min = 2123/2 = 1062. Actual 1108. 46 cycle slack.

Load idle breakdown:
- cycles 37-64 (28 cycles, 55 slots idle): mid-startup
- cycles 1097-1107 (11 cycles, 22 slots): drain
- cycles 73-77 + 94-96: 5 + 3 cycles minor

Cycles 38-64 run only valu/alu/flow (processing hash stages of early rounds).
Load is starved because gather `p` isn't ready yet (chain through hash).

## Structural bounds (why <1062 is near-impossible)
- 2048 gather loads are unavoidable (8 rounds × 256 lanes × 1 slot each)
- Arith-select alternatives (vselect / mad) push flow or valu above the load bound
- FLOW is already saturated at 720 slots (mostly L1/L2/L3 vselect trees)
- The 46-cycle gap is scheduler slack limited by the hash → p → next-level-gather chain

## Applied optimizations (chronological)
1. Vectorized 32 groups × VLEN 8
2. `multiply_add` for hash stages 0, 2, 4
3. Fused stages 2+3: `mad(33, c2+c3)` XOR `mad(33<<9, c2<<9)` (saves 1 valu/round)
4. Lazy stage-5 XOR: store `val_stored = val ^ (val >> 16)`; pre-XOR broadcasts with c5; `node_adj` XORs gathers with c5 so math works out
5. Per-lane ALU for h1b, h5b shifts
6. L1 vselect, L2 3-vselect tree, L3 7-vselect tree (preloaded broadcasts)
7. Mix half `g_val_ptr` via flow `add_imm`, rest via `load const`
8. Precomputed `gather_base_v` broadcasts (complemented base → subtract gather)
9. 2 vloads for `tree_s[0..6]` + `l3_tree_s[0..7]` (saves 15 const loads)
10. Mobility-based scheduler: `(engine_rank, mobility, -priority - asap//4, i*31)`
11. Alias `g_t2` with `g_node` (free scratch)
12. Tighter per-stage hash broadcast deps (hc_bcs[i] per stage vs hc_all_ready)
13. Free `hc5_v2` by not allocating unused broadcasts

## Ideas tried but discarded
- L4 arith-select (vselect tree for level 4 nodes) — flow bound explodes
- All const loads via flow add_imm — flow bottleneck (1/cycle setup)
- h5b via valu shift — +10 cycles (valu bound increased)
- Multi-scheduler (5-15 configs) — current sort key is near-optimal
- `g_val_ptr` alias with `g_p` — adds 16 flow add_imm stores (+1 cycle)
- Alias reduction experiments — no further alias opportunities

## If resuming, most promising untried:
1. **Reduce mid-startup load idle**: find DAG restructuring that allows earlier
   gather loads during cycles 38-64. Currently starved waiting for `p` through
   hash chain. Possibly: speculative preload of popular addresses, pipelining
   round N+1 gather preparation earlier.
2. **Algorithmic shortcuts**: are any L1/L2 rounds actually equivalent to
   something cheaper? Pre-analysis of tree structure.
3. **Cross-round software pipelining**: explicitly break the `p` dependency
   by computing p with a different path.

## Tool setup
- `bash autoresearch.sh` — runs the benchmark, outputs `METRIC cycles=N`
- `python3 tests/submission_tests.py` — runs full test suite
- `autoresearch.checks.sh` — multi-seed correctness + tests/ tamper check

## Key files
- `perf_takehome.py` — THE file to modify (KernelBuilder class)
- `problem.py` — simulator (read-only reference)
- `tests/frozen_problem.py` — frozen simulator (DO NOT MODIFY)
- `tests/submission_tests.py` — DO NOT MODIFY
