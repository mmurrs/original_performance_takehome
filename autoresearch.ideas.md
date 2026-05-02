# Autoresearch: VLIW SIMD Kernel — Final State (Run 69)

## Best: 1146 cycles (128.9× speedup from 147734 baseline)

## Summary
- All 9 submission_tests pass
- Well below Opus 4.5 improved harness (1363)
- Well below Opus 4.5 11hr harness (1487)
- Still 145 cycles from stretch target (1001)

## Profile at 1146
- ALU: 12801 slots → 1067 min
- VALU: 6502 slots → 1084 min ← bottleneck
- LOAD: 2133 slots → 1067 min (fully packed)
- FLOW: 736 slots → 736 min
- Scratch: 1457/1536 used
- Body load-idle: only 70 cycles (6%)
- Prolog/epilog: ~12 cycles total

## Key optimizations
1. Vectorized 32 groups × VLEN=8
2. multiply_add for hash stages 0, 2, 4
3. Stage 1, 3 shifts on per-lane ALU
4. Level 0: root_v broadcast (no gather)
5. Level 1: vselect (flow)
6. Level 2: 3-vselect tree (flow) + shared l2_b1_tmps=1
7. Level 3: 7-vselect tree + l3_tmp2 to skip b1_hi recompute
8. Level 4–10: load_offset gather + precomputed gather_base_v
9. g_val_ptr via flow add_imm (frees 32 load slots)
10. Split val/p dep chains (level-0 skip pdep)
11. Per-lane ALU bit extracts where beneficial
12. g_t2 aliased to g_node (saves 256 scratch)
13. Shared tmps: l2=1, l3=2 (sweet spot; tighter = scheduling thrash, looser = worse packing)
14. Scheduler: (engine_rank, -priority, -i) — prefer later-inserted ops as tiebreak
15. Reverse load_offset lane order (-2 cycles)
16. Strengthened correctness check (tests across 8+ random seeds)

## Ideas tried and rejected
- In-place hash shifts: required strict dep order, adds critical path cycle
- Moving more ops to ALU: ALU oversaturates → worse
- Level-3 multilinear arith: too many scratch vectors needed
- Level-4 arith-select: 15 vselects/group → flow becomes bottleneck
- Early LSB via bit-16 XOR: no critical path benefit
- Fuse hash stage 2+3: same slot count, worse scheduling
- Priority/ASAP/engine scheduler variations: within ±5 cycles

## Hard bottleneck
VALU at 1084 min: hash has 11 serial valu stages per round × pipelining.
ISA has no way to reduce hash op count further.
