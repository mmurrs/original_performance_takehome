# VLIW SIMD Kernel — Optimization Progress

## Final: 1148 cycles (128.7× speedup from 147734 baseline)

### Submission tests passing: 9/9
- Below Claude Opus 4.5 improved harness (1363)
- Below Claude Opus 4.5 11hr harness (1487)
- Above stretch target (1001)

## Architecture summary
VLIW ISA: 12 ALU + 6 VALU + 2 LOAD + 2 STORE + 1 FLOW slots/cycle.
VLEN=8 vector lanes. 32 groups × 8 lanes = 256 element batch.
16 rounds of tree traversal + hashing.

## Key optimizations (ordered by impact)
1. **Vectorize** (147734 → ~5000) — VLEN=8 SIMD with vload/vstore, multiply_add for hash stages 0/2/4
2. **List scheduler + gather** (→ ~2000) — Critical-path priority scheduler, load_offset gather
3. **Per-lane ALU bit extracts** (→ 1759) — Stage 1, 3 shifts and p_update bit on ALU frees VALU
4. **L3 arith-select via vselect** (→ 1289) — Binary tree of flow vselects replaces 2 gather rounds
5. **L3 scratch sharing + b1_hi elim** (→ 1189) — Shared l3_tmp0/1/2 with serialization chain
6. **Precompute gather base vectors** (→ 1201) — Saves 1 VALU/gather
7. **L1 via vselect** (→ 1154) — Single flow op instead of multiply_add
8. **l2_b1_tmps = 1** (→ 1148) — Tighter serialization improves packing
9. **Scheduler tiebreak -i** (→ 1148) — Prefer later-inserted ops

## Final profile
- VALU: 6502 slots → 1084 min (bottleneck; 94% utilization)
- ALU: 12801 slots → 1067 min
- LOAD: 2133 slots → 1067 min (fully packed)
- FLOW: 736 slots → 736 min
- Only ~14 under-filled bundles in 1148 total (99% saturated packing)

## Why 1148 and not lower
- Hash has 11 serial VALU stages per round × 16 rounds = 176 stages → ~940 VALU cycles min
- Per-round overhead (bit extract, p update, level select) ~140 more VALU cycles
- Load min is 1067; we're only 80 cycles over → scheduler near-optimal
- No ISA feature to reduce hash op count (XOR doesn't fuse with *, +)

## Notes on attempted optimizations (see autoresearch.ideas.md)
- In-place hash shifts: adds strict dep, longer critical path
- L4 arith-select: too many flow ops (1440 min)
- Early LSB via XOR: no critical path benefit when analyzed carefully
- Fuse hash stages 2+3: same slot count, no benefit
