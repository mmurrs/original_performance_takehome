# Autoresearch: VLIW SIMD Kernel — Final State

## Best: 1148 cycles (128.7x speedup from 147734 baseline)
- All 9 submission_tests pass
- Below Opus 4.5 improved harness (1363)
- Below Opus 4.5 11hr harness (1487)
- Gap to "stretch" target (1001): 147 cycles

## Profile at 1148
- ALU: 12801 slots → 1067 min
- VALU: 6502 slots → 1084 min ← bottleneck
- LOAD: 2133 slots → 1067 min (fully packed)
- FLOW: 736 slots → 736 min
- 70% of bundles are 3+ engines saturated; near-optimal packing.

## Why we stop at ~1148
- Hash has 11 serial VALU stages per round × 16 rounds × 32 groups / 6 valu/cycle ≈ 940 cycles.
- Plus per-round overhead (bit extract, p update, gather addressing) = ~140 cycles valu extra.
- Total VALU min ~1084. Scheduler achieves ~1148 (98.5% efficiency).

## Key optimizations
1. Vectorized 32 groups × VLEN=8
2. multiply_add for hash stages 0, 2, 4
3. Stage 1, 3 shifts on per-lane ALU (free valu)
4. Level 0: broadcast root_v (no gather)
5. Level 1: vselect (flow engine)
6. Level 2: 3-vselect tree (flow)
7. Level 3: 7-vselect tree + l3_tmp2 to skip b1_hi recompute
8. Level 4–10: scatter gather via load_offset + precomputed base vectors
9. g_val_ptr via flow add_imm (saves 32 load slots)
10. x0-tracked node_war fence for cross-round decoupling
11. Split val/p/node dep chains
12. Per-lane ALU bit extracts where beneficial
13. g_t2 aliased to g_node (saves 256 scratch words)
14. Shared tmps l2=1, l3=2 (sweet spot; more slots = worse scheduling, fewer = too tight)
15. Scheduler: engine_rank, -priority, -i tiebreak (prefer later-inserted ops)

## Ideas tried and rejected
- In-place hash shifts (val >>=k): required strict dep order → added critical path cycle
- Move more shifts/XORs to ALU: ALU oversaturates
- Level-3 multilinear arith-select (VALU): requires too many scratch vectors
- Level-4 arith-select: 15 vselects/group × 2 rounds = 960 flow slots → flow becomes bottleneck
- Early LSB via bit-16 XOR trick: no critical path benefit
- Fuse hash stages 2+3: same slot count, scheduler didn't benefit
- Various scheduler orderings: all within ±5 cycles of baseline

## Remaining but hard
- ISA extension (fused XOR-ADD, vector gather from scratch) not available
- Cross-round pipelining already tight (p_update → gather chain is the fundamental dep)
