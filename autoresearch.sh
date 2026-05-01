#!/bin/bash
set -e
cd /Users/matt/perf_takehome

# Run the benchmark and extract cycle count
OUTPUT=$(python3 -c "
import random, sys
sys.path.insert(0, '.')
from problem import Tree, Input, build_mem_image, reference_kernel2, Machine, N_CORES
from perf_takehome import KernelBuilder

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

kb = KernelBuilder()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
machine.enable_pause = False
machine.enable_debug = False
machine.run()

for ref_mem in reference_kernel2(mem): pass
inp_values_p = ref_mem[6]
correct = machine.mem[inp_values_p:inp_values_p+256] == ref_mem[inp_values_p:inp_values_p+256]
if not correct:
    print('INCORRECT')
    sys.exit(1)
print(f'METRIC cycles={machine.cycle}')
" 2>&1)

echo "$OUTPUT"
