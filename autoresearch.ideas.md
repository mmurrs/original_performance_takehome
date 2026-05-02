# Autoresearch: VLIW SIMD Kernel Optimization

## Current best: 1154 cycles (from 147734 baseline)

## Profile (at 1154)
- ALU: 12801 slots, 1067 min (fully packed)
- VALU: 6502 slots, 1084 min (bottleneck: 94% utilization)
- LOAD: 2133 slots, 1067 min
- FLOW: 736 slots, 736 min
- Scratch: 1465/1536 (71 free)

## Key optimizations in place
1. Vectorized 32 groups × VLEN=8
2. multiply_add for hash stages 0, 2, 4 (constants: 4097, 33, 9)
3. Stage 1, 3 shifts on per-lane ALU (freed valu slots)
4. Level-0 uses root_v broadcast (no gather/select)
5. Level-1 uses vselect (flow engine) with 2 precomputed broadcast vectors
6. Level-2 uses 3-vselect tree (flow)
7. Level-3 uses 7-vselect binary tree (flow) with shared l3_tmp0/1/2 slots
8. Level-4+ uses gather with precomputed base vectors
9. g_val_ptr via flow add_imm (frees 32 load slots)
10. x0-based node_war tracking (decoupled cross-round node dep)
11. Split val/p dep chains (level-0 skip p)
12. Bit extracts on per-lane ALU (p_update, level-3 b0/b1)
13. g_t2 aliased to g_node (shared scratch)

## Remaining ideas
- **Early LSB via algebra**: LSB(val_post5) = LSB(val_post4) ^ bit16(val_post4) ^ 1.
  Analysis: same critical depth as current, so no speedup.
- **Level-4 arith-select**: 15 vselects/group would push flow to 1440 min — worse than current.
- **Combined hash stage fusion**: XOR doesn't distribute over +/*, so no algebraic simplification.
- **Scheduler improvements**: tried ASAP-first, priority-first — no improvement. Greedy is near-optimal here.

## Hard bottleneck
- 10 gather rounds at 128 load cycles each = 1280 min load → reduced to 1067 via L3 arith-select
- Hash critical path ~10-11 valu stages per round × pipelining → 1084 min valu
- Gap to 1001 requires either:
  - Reducing hash stage count (no known way with this ISA)
  - Eliminating more gather rounds (L4+ too expensive)
