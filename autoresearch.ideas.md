# Autoresearch ideas / backlog

## Stuck region: 1717 cycles
Load engine at 1330 slots (99.9% packed). VALU theoretical 1193, ALU 1003.
Best seen (lost session): 1448.

## High-value ideas not yet explored
- **Merge g_node into g_t1** (alias them) to free 256 scratch words → enables level-3 arith-select.
- **Eliminate g_t2** via in-place shift: `h1b: val >> c -> val; h1c: t1 ^ val -> val`. Would save 256 scratch words.
- **Level-3 arith-select** via multilinear expansion: `result = A + B*b0 + C*b1 + D*b2 + E*b0b1 + F*b0b2 + G*b1b2 + H*b0b1b2`. 8 const broadcasts + 3 bit extracts + 4 product mads + 7 accumulator mads. Saves 256 load cycles over 2 rounds. Needs scratch savings first.
- **Cross-round dep relaxation**: track fine-grained fences (node-WAR via x0, t1/t2-WAR via final_val, p-WAR via prev p_up). Previous attempt failed correctness — need careful bookkeeping.
- **Better scheduler priority**: try (-priority, engine_rank, round_index) to prefer earlier rounds' ops. Or weight by resource pressure.
- **Tree prefetch**: preload entire levels 3..6 into scratch (127 values) during setup → then everything below level 7 can be arith-selected. But load_offset still hits memory, so unclear benefit.
- **LSB of final_val without full stage 5**: LSB(final) = LSB(val_post4) ^ bit16(val_post4) ^ 1. Could start next-round gather earlier by computing p bit pre-stage-5.

## Dead ends
- Algebraic refactor of level-2 (3-mad chain) — longer critical path than tree form.
- Moving stage-1,5 shifts both to ALU — over-saturates ALU.
- Moving level-2 bit extracts to ALU — ALU is 12-wide but during level-0/1/2 rounds other ALU work already saturates.

## Notes
- Bottleneck is inherently gather-load throughput at 1280 min (10 rounds × 256 / 2).
- To beat 1280, MUST eliminate gather rounds via arith-select.
- L3 arith-select is feasible (est ~1300-1400 cycles post).
- L4+ arith-select too expensive (VALU-bound >> load savings).
