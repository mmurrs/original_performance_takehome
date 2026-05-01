# Autoresearch: VLIW SIMD Kernel Optimization

## Objective
Minimize cycle count of `KernelBuilder.build_kernel` in `perf_takehome.py`.
Target: < 1001 cycles. Current: 1448 cycles.

## Benchmark
```bash
bash autoresearch.sh
# Outputs: METRIC cycles=N (or INCORRECT on failure)
```

## Key Constraint
**DO NOT modify anything in tests/ directory.**
Only modify `perf_takehome.py`.

## Architecture Summary
- VLIW processor: alu(12), valu(6), load(2), store(2), flow(1) slots per cycle
- VLEN=8 (vector length), batch_size=256 (32 groups), 16 rounds
- Writes take effect at END of cycle (reads before writes)
- `multiply_add(dest, a, b, c)` = a*b+c in 1 VALU slot
- `load_offset(dest, addr, k)` = scatter load, 1 LOAD slot
- `vload/vstore` = contiguous 8-element vector load/store

## Algorithm
Per round, per group of 8 items:
1. Compute gather addresses: addr[i] = forest_ptr + idx[i]
2. Gather 8 tree node values from memory (8 load_offset ops = 4 cycles at 2/cycle)
3. XOR val with node: val ^= node (1 VALU)
4. Hash val through 6 stages (9 VALU cycles with multiply_add optimization)
5. Branch: idx = 2*idx + 1 + (val&1), wrap if >= n_nodes (5 VALU cycles)

## Current Implementation
- 32 groups processed simultaneously via list scheduler
- idx/val kept in scratch across all 16 rounds (only vload at start, vstore at end)
- Broadcast optimization for level-0 rounds (rounds 0, 11)
- Level-1 arithmetic selection for rounds 1, 12
- Cross-round pipelining: consecutive normal rounds scheduled together for overlap

## THE BOTTLENECK
**Load throughput**: 12 normal rounds × 256 scatter-gathers / 2 per cycle = 1536 load-cycles.
This ALONE exceeds the 1099 target.
VALU throughput: 16 × 96 = 1536 cycles (but overlaps with loads).

## Ideas to Explore
1. **Extend arithmetic selection to levels 2-3**: Use multiply_add binary selection tree
   to replace gathers for rounds at tree levels 2-4. The challenge: selection adds VALU ops
   that may exceed the load savings. Key: minimize per-group selection VALU ops.

2. **Cross-round pipelining ALL rounds**: Currently only pairs normal rounds.
   Could pair broadcast/level-1 rounds with adjacent normal rounds if aliasing is fixed.

3. **Preload tree levels into scratch**: For levels with few nodes (≤16),
   vload contiguous ranges into scratch vectors. Use arithmetic to select.

4. **Reduce hash VALU ops**: Find algebraic shortcuts or ISA tricks.
   E.g., can any 2-cycle hash stages be done in 1 cycle?

5. **Process-then-store batching**: Instead of per-round bodies, process multiple
   rounds per group in a tighter loop.

6. **Shared gather results**: At early tree levels, many items share the same
   node. Could detect duplicates or preload all unique nodes.

## Dead Ends
- ALU hash offload: scalar ALU is 4x less efficient per element than VALU
- Level 2+ arithmetic selection with current VALU budget: total VALU > load savings
  (selection adds too many ops to already-tight VALU schedule)

## Files
- `perf_takehome.py`: THE file to modify (KernelBuilder class)
- `problem.py`: Simulator and ISA (read-only reference)
- `tests/submission_tests.py`: Validation (DO NOT MODIFY)
- `tests/frozen_problem.py`: Frozen simulator for testing (DO NOT MODIFY)
