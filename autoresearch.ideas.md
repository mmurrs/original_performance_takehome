# Autoresearch: VLIW SIMD Kernel — Session Notes (pre-compaction)

## Current best: 1110 cycles (133.09× speedup from 147734 baseline)
- HEAD commit: `d322ee3` ("i*31 tiebreak")
- Directory: `/Users/matt/perf_takehome` (autoresearch workingDir)
- 117 experiments logged
- All 9 submission_tests pass (multi-seed correctness verified)

## Target: 1001 cycles (per kerneloptimization.fun leaderboard)
- Many leaderboard leaders tied at 1001 (Paul1365972, iheartcomputer, Kevin, mghandi, etc.)
- Gap: 109 cycles
- None publish code

## Profile of 1110
- LOAD: 2123 slots → 1062 min (packed at 2/slot, 49 idle cycles)
  - 2048 gather loads (8 L4+ rounds × 256 lanes)
  - 41 const + 34 vload = 75 setup
  - 49 idle: 22 startup + 13 mid + 14 drain
- VALU: 5779 slots → 964 min
- ALU: 11776 slots → 982 min (8192 shifts h1b/h5b + 3584 bit extracts)
- FLOW: 720 slots → 720 min
- Scratch: 1448/1536

## Applied optimizations (key commits)
1. Vectorized 32 groups × VLEN 8
2. multiply_add for hash stages 0, 2, 4
3. **Fused stages 2+3** via precomputed constants: `mad(33, c2+c3)` XOR `mad(33<<9, c2<<9)` (saves 1 valu per round)
4. **Lazy stage-5 XOR**: store `val_stored = val_pre5 ^ (val_pre5 >> 16)`, pre-XOR broadcasts with c5, use `-` with complemented gather_base, swap vselect arg orders (saves h5a per round; node_adj for gathers)
5. Per-lane ALU for h1b, h5b shifts and bit extracts
6. L1 vselect (flow), L2 3-vselect tree, L3 7-vselect tree
7. g_val_ptr: 50/50 mix of flow add_imm / load-const
8. Precomputed gather_base_v broadcasts
9. vload tree values (2 vloads for tree_s[0..6] + l3_tree_s[0..7])
10. Mobility-based scheduler: `(engine_rank, mobility, -priority - asap//4, i*31)` where mobility = max_depth - asap - priority
11. g_t2 aliased to g_node
12. l2_b1_tmps=1, l3_tmp0/1/2=2 shared with serialization chains
13. Strengthened autoresearch.checks.sh with multi-seed correctness (caught bogus kernels passing seed=123 only)

## Theoretical analysis (why 1001 is hard)
- LOAD min = 2123/2 = 1062. 1001 requires ≤2002 load slots, i.e., eliminate ≥121 slots.
- Eliminating an L4 gather round saves 256 load slots but adds:
  - Via 15-vselect tree: +480 flow slots per round → flow min exceeds 1001. Bad.
  - Via multilinear mad: +960 valu per round → valu bound >> 1001. Bad.
  - Via hybrid (N=9 groups vselect, rest gather): theoretical max min ≈990-1030. Would need 128+ scratch for 16 L4 broadcast vectors; currently only 88 free.
- Early LSB (compute p-bit from val_post_h4): same critical path, costs extra valu/alu.
- All my analyses suggest 1001 requires some mix of L4 arith + tight scratch management I haven't figured out.

## AgentCash / Exa research (cost ~$0.07 USDC on Base)
- Public forks all above 1200: clocksmith 1288, JoeHowarth 1296, SumitKumar 1313, obviyus 1524, scottmaran ~similar
- MarioC-hub uses "Staggered stream starts" — offset group starts to spread load traffic
- obviyus uses "Global list scheduler + BFS levels + priority queues per engine"
- SumitKumar uses "Linear interpolation nodes 0-14 + batch-outer processing"
- My 1110 beats all public forks

## Key blockers to 1001
1. **Scratch budget**: only 88 free. L4 broadcast vectors need 128. Without freeing scratch, can't fit L4 arith-select.
2. **Engine balance**: currently near-optimal for 8 gather rounds. Any reduction of gather increases another engine past 1001.
3. **Unknown trick**: 1001 solutions are private; may use approach I haven't found.

## If resuming, most promising untried avenues
1. **Reduce scratch** (free 40+ words somehow):
   - Alias more tmps
   - Remove unused broadcasts (hash_vecs[1][1], hash_vecs[3][1] are allocated but unused)
   - Eliminate g_val_ptr by using add_imm inline on stores (but flow bound might tighten)
2. **Then implement L4 hybrid**: N groups use 15-vselect tree, rest gather. Target N≈9 for flow/load balance.
3. **Stream staggering**: delay group i's first ops by i × STEP for better load pipelining during startup.
4. **Try software pipelining via explicit rotating registers**: replicate val state across 2-3 rounds to enable deeper overlap.

## Tool setup
- agentcash MCP server accessible via: `python3 /Users/matt/mcp_client.py call <tool> '<json_args>'`
- Has Exa web search at $0.01/call via x402 (balance ~$4.57 USDC remaining)
- Leaderboard: https://www.kerneloptimization.fun/api/leaderboard

## Key files
- `perf_takehome.py`: THE kernel (only file to edit)
- `tests/` DO NOT MODIFY (autoresearch.checks.sh verifies)
- `autoresearch.jsonl`: experiment log at /Users/matt/autoresearch.jsonl
- `autoresearch.checks.sh`: runs multi-seed correctness + tests/ diff check

## Final session notes (post research)
- Leaderboard verified: 1001 cycles achieved by many (top tier). Not public code.
- My 1110 beats all public forks (clocksmith 1288, JoeHowarth 1296, SumitKumar 1313, etc).
- Tried: stream stagger, multi-config scheduler, const→flow ops, h5b valu, multiple scheduler heuristics — none break through.
- Hard blockers for 1001:
  1. LOAD min = 1062 based on 2123 slots. To reach 1001 min = 2002 slots, must cut 121+ slots.
  2. 2048 gather loads are unavoidable via load_offset (1 per lane).
  3. Arith-select alternatives (vselect, mad, multilinear) all push other engines (flow/valu) above 1001.
  4. Setup const loads (41) can't all move to flow (1/cycle bottleneck).
  
Creative paths I couldn't realize:
- Restructuring memory layout for vload-friendly gather
- Fusing more hash stages (only h2+h3 fusable via algebra)
- Speculative/parallel execution (no ISA hardware support)
- Software pipelining (my list scheduler already does this implicitly)
