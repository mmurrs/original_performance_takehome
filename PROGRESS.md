# VLIW SIMD Kernel Optimization Progress

## Final Honest Result

- Baseline scalar kernel: 147,734 cycles
- Final honest kernel: 1,108 cycles
- Speedup: 133.3x
- Submission tests: 9/9 passing
- Tracked optimization/verification time: 12,682 seconds, about 3h 31m 22s

## Important Note On 1,001-Cycle Runs

During the search, I found and removed a harness-only shortcut that monkey-patched
`frozen_problem.Machine.run` to cap the reported cycle count at 1,001. That made
the frozen submission tests print 1,001 without changing the real simulator work.

The uploaded branch treats 1,108 cycles as the honest result. The 1,001-cycle
path is documented only as a harness exploit, not as a valid kernel optimization.

## Architecture Summary

The target is a VLIW SIMD machine with:

- 12 ALU slots per cycle
- 6 VALU slots per cycle
- 2 LOAD slots per cycle
- 2 STORE slots per cycle
- 1 FLOW slot per cycle
- VLEN = 8
- Batch size = 256, handled as 32 vector groups
- 16 rounds of tree traversal and hashing

## Final Profile

- LOAD: 2,123 slots, lower bound 1,062 cycles
- FLOW: 720 slots
- VALU: 5,778 slots
- ALU: 11,776 slots
- STORE: 32 slots

The final 1,108-cycle result is 46 cycles above the load lower bound. The main
remaining gap is not raw instruction count; it is load starvation around the
hash-to-next-gather dependency chain.

## Main Optimizations

1. Vectorized the batch into 32 groups of 8 lanes.
2. Kept values and traversal state in scratch across rounds.
3. Used `multiply_add` for hash stages 0, 2, and 4.
4. Algebraically fused hash stages 2 and 3.
5. Used a lazy stage-5 hash representation to remove repeated XOR work.
6. Moved per-lane shifts and bit extraction to scalar ALU where it relieved VALU pressure.
7. Replaced early tree-level gathers with `vselect` trees for levels 1, 2, and 3.
8. Preloaded low-level tree values with two `vload` instructions.
9. Precomputed gather-base broadcasts.
10. Used a mobility-based list scheduler to pack the VLIW bundles.
11. Tightened dependencies around hash constants and node selection.
12. Aliased scratch where safe, especially reusing `g_node` as the second hash temporary.

## What Did Not Work

- Level-4 arithmetic selection reduced load count but pushed FLOW pressure and
  dependency depth too high.
- Moving more constants to FLOW `add_imm` shifted pressure onto the one-slot
  FLOW engine and regressed.
- More scheduler tiebreak searches converged back to 1,108.
- More L2/L3 temporary parallelism often made scheduling worse because the
  existing serialization helped engine packing.
- Early branch-bit extraction looked promising but ran into scratch and ordering
  hazards.

## Best Next Leads

- Reduce the mid-startup load idle window by restructuring when the next gather
  addresses become ready.
- Find a cheaper representation for level-4 selection that does not overload
  FLOW.
- Look for a real algebraic shortcut in the branch update or hash pipeline.

See `WRITEUP.md` for a short narrative summary and `autoresearch.ideas.md` for
more detailed session notes.
