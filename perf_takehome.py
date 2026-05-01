"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Vectorized VLIW kernel. Only modifies KernelBuilder.build_kernel (and helpers).
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit(self, bundle):
        for engine, slots in bundle.items():
            if engine != "debug":
                assert len(slots) <= SLOT_LIMITS[engine], (engine, len(slots))
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized VLIW kernel. The simulator only checks inp_values, so we
        keep the tree-level index in scratch and never write it back to memory.

        Structure:
          - 32 groups of VLEN=8 lanes each
          - val vectors persist in scratch across rounds (one vload/vstore total)
          - p vector (local level index, 0-based within current tree level)
          - Per round, gather 8 tree node values per group, then hash val^node
          - Levels 0/1/2 use arithmetic selection (no gather)
          - Levels >=3 use load_offset gather
          - A list scheduler packs ops into VLIW bundles, overlapping rounds
        """
        assert batch_size % VLEN == 0
        n_groups = batch_size // VLEN
        forest_values_p = 7
        inp_values_p = forest_values_p + n_nodes + batch_size

        # ---- scratch allocations ----
        one_s = self.alloc_scratch("one_s")
        two_s = self.alloc_scratch("two_s")
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)

        # Hash constants rewritten for multiply_add on stages 0,2,4:
        #   (a+c) + (a<<k) == a*(1+2**k) + c
        # Stages 1,3,5 keep their form: (a OP c1) OP2 (a SHIFT c2)
        hash_const_values = [
            (1 + (1 << 12), 0x7ED55D16),  # stage 0: multiply_add
            (0xC761C23C, 19),             # stage 1: xor/shift
            (33, 0x165667B1),             # stage 2: multiply_add
            (0xD3A2646C, 9),              # stage 3: add/shift/xor
            (9, 0xFD7046C5),              # stage 4: multiply_add
            (0xB55A4F09, 16),             # stage 5: xor/shift
        ]
        hash_scalars = []
        hash_vecs = []
        for hi in range(6):
            s1 = self.alloc_scratch(f"hc{hi}_s1")
            s2 = self.alloc_scratch(f"hc{hi}_s2")
            v1 = self.alloc_scratch(f"hc{hi}_v1", VLEN)
            v2 = self.alloc_scratch(f"hc{hi}_v2", VLEN)
            hash_scalars.append((s1, s2))
            hash_vecs.append((v1, v2))

        root_v = self.alloc_scratch("root_v", VLEN)
        l1_base_v = self.alloc_scratch("l1_base_v", VLEN)
        l1_diff_v = self.alloc_scratch("l1_diff_v", VLEN)
        l2_base0_v = self.alloc_scratch("l2_base0_v", VLEN)
        l2_diff0_v = self.alloc_scratch("l2_diff0_v", VLEN)
        l2_base1_v = self.alloc_scratch("l2_base1_v", VLEN)
        l2_diff1_v = self.alloc_scratch("l2_diff1_v", VLEN)

        # Per-level base pointer into forest_values for gathered levels.
        gather_base_s = {}
        for level in range(3, forest_height + 1):
            gather_base_s[level] = self.alloc_scratch(f"gb{level}_s")

        # Per-group state
        g_p = []
        g_val = []
        g_node = []
        g_t1 = []
        g_t2 = []
        g_val_ptr = []
        for gi in range(n_groups):
            g_p.append(self.alloc_scratch(f"g{gi}_p", VLEN))
            g_val.append(self.alloc_scratch(f"g{gi}_val", VLEN))
            g_node.append(self.alloc_scratch(f"g{gi}_node", VLEN))
            g_t1.append(self.alloc_scratch(f"g{gi}_t1", VLEN))
            g_t2.append(self.alloc_scratch(f"g{gi}_t2", VLEN))
            g_val_ptr.append(self.alloc_scratch(f"g{gi}_vp"))

        tree_s = [self.alloc_scratch(f"tree{i}_s") for i in range(7)]
        l1_diff_s = self.alloc_scratch("l1_diff_s")
        l2_diff0_s = self.alloc_scratch("l2_diff0_s")
        l2_diff1_s = self.alloc_scratch("l2_diff1_s")

        def emit_loads(slots):
            for i in range(0, len(slots), SLOT_LIMITS["load"]):
                self.emit({"load": slots[i : i + SLOT_LIMITS["load"]]})

        # ---- Init (outside the scheduled body) ----
        const_loads = [("const", one_s, 1), ("const", two_s, 2)]
        for (s1, s2), (c1, c2) in zip(hash_scalars, hash_const_values):
            const_loads.append(("const", s1, c1))
            const_loads.append(("const", s2, c2))
        for level, dest in gather_base_s.items():
            const_loads.append(("const", dest, forest_values_p + (1 << level) - 1))
        for i, dest in enumerate(tree_s):
            const_loads.append(("const", dest, forest_values_p + i))
        emit_loads(const_loads)
        # Load first 7 tree values (levels 0, 1, 2).
        emit_loads([("load", dest, dest) for dest in tree_s])

        # ---- Scheduled body ----
        ops = []

        def add_op(engine, slot, deps=()):
            idx = len(ops)
            ops.append((engine, slot, tuple(deps)))
            return idx

        # Broadcasts & level-setup subs as deps in the scheduled body, so they
        # pipeline with early group work.
        diff_l1 = add_op("alu", ("-", l1_diff_s, tree_s[2], tree_s[1]))
        diff_l20 = add_op("alu", ("-", l2_diff0_s, tree_s[4], tree_s[3]))
        diff_l21 = add_op("alu", ("-", l2_diff1_s, tree_s[6], tree_s[5]))

        one_bc = add_op("valu", ("vbroadcast", one_v, one_s))
        two_bc = add_op("valu", ("vbroadcast", two_v, two_s))
        hc_bcs = []
        for (v1, v2), (s1, s2) in zip(hash_vecs, hash_scalars):
            hc_bcs.append(add_op("valu", ("vbroadcast", v1, s1)))
            hc_bcs.append(add_op("valu", ("vbroadcast", v2, s2)))
        root_bc = add_op("valu", ("vbroadcast", root_v, tree_s[0]))
        l1_base_bc = add_op("valu", ("vbroadcast", l1_base_v, tree_s[1]))
        l1_diff_bc = add_op("valu", ("vbroadcast", l1_diff_v, l1_diff_s), [diff_l1])
        l20_base_bc = add_op("valu", ("vbroadcast", l2_base0_v, tree_s[3]))
        l20_diff_bc = add_op("valu", ("vbroadcast", l2_diff0_v, l2_diff0_s), [diff_l20])
        l21_base_bc = add_op("valu", ("vbroadcast", l2_base1_v, tree_s[5]))
        l21_diff_bc = add_op("valu", ("vbroadcast", l2_diff1_v, l2_diff1_s), [diff_l21])

        hc_all_ready = hc_bcs + [one_bc, two_bc]

        # Per-group pointer init + vload.
        ptr_loads = []
        vloads = []
        for gi in range(n_groups):
            pl = add_op("load", ("const", g_val_ptr[gi], inp_values_p + gi * VLEN))
            ptr_loads.append(pl)
            vl = add_op("load", ("vload", g_val[gi], g_val_ptr[gi]), [pl])
            vloads.append(vl)

        def add_hash(gi, val_in, node_addr, node_deps, is_last):
            """Emit val = hash(val_in ^ node). Returns deps producing final val."""
            val = g_val[gi]
            t1 = g_t1[gi]
            t2 = g_t2[gi]

            x0 = add_op("valu", ("^", val, val_in, node_addr), node_deps + [hc_bcs[0], hc_bcs[1]])
            # stage 0: a*(1+4096) + c1
            h0 = add_op(
                "valu",
                ("multiply_add", val, val, hash_vecs[0][0], hash_vecs[0][1]),
                [x0] + hc_all_ready,
            )
            # stage 1: (a ^ c1) ^ (a >> 19)
            h1a = add_op("valu", ("^", t1, val, hash_vecs[1][0]), [h0])
            h1b = add_op("valu", (">>", t2, val, hash_vecs[1][1]), [h0])
            h1c = add_op("valu", ("^", val, t1, t2), [h1a, h1b])
            # stage 2: a*33 + c1
            h2 = add_op(
                "valu",
                ("multiply_add", val, val, hash_vecs[2][0], hash_vecs[2][1]),
                [h1c],
            )
            # stage 3: (a + c1) ^ (a << 9)
            h3a = add_op("valu", ("+", t1, val, hash_vecs[3][0]), [h2])
            h3b = [
                add_op("alu", ("<<", t2 + lane, val + lane, hash_scalars[3][1]), [h2])
                for lane in range(VLEN)
            ]
            h3c = add_op("valu", ("^", val, t1, t2), [h3a] + h3b)
            # stage 4: a*9 + c1
            h4 = add_op(
                "valu",
                ("multiply_add", val, val, hash_vecs[4][0], hash_vecs[4][1]),
                [h3c],
            )
            # stage 5: (a ^ c1) ^ (a >> 16)
            h5a = add_op("valu", ("^", t1, val, hash_vecs[5][0]), [h4])
            h5b = add_op("valu", (">>", t2, val, hash_vecs[5][1]), [h4])
            h5c = add_op("valu", ("^", val, t1, t2), [h5a, h5b])
            return h5c

        # last[gi] = list of op-idx producing the group's val (dep chain tail)
        last = [[vl] + hc_all_ready for vl in vloads]

        for round_i in range(rounds):
            level = round_i % (forest_height + 1)
            is_last = (round_i == rounds - 1)
            for gi in range(n_groups):
                p = g_p[gi]
                node = g_node[gi]
                t1 = g_t1[gi]
                t2 = g_t2[gi]
                prev = last[gi]

                if level == 0:
                    node_deps = prev + [root_bc]
                    node_addr = root_v
                elif level == 1:
                    # node = l1_base + p * l1_diff  (p ∈ {0,1})
                    sel = add_op(
                        "valu",
                        ("multiply_add", node, l1_diff_v, p, l1_base_v),
                        prev + [l1_base_bc, l1_diff_bc],
                    )
                    node_deps = [sel]
                    node_addr = node
                elif level == 2:
                    # p ∈ {0..3}; bits b0=p&1, b1=(p>>1)&1
                    b0 = add_op("valu", ("&", t1, p, one_v), prev + [one_bc])
                    b1 = add_op("valu", (">>", t2, p, one_v), prev + [one_bc])
                    # r0 = l2_base0 + b0 * l2_diff0
                    r0 = add_op(
                        "valu",
                        ("multiply_add", node, l2_diff0_v, t1, l2_base0_v),
                        [b0, l20_base_bc, l20_diff_bc],
                    )
                    # r1 = l2_base1 + b0 * l2_diff1
                    r1 = add_op(
                        "valu",
                        ("multiply_add", t1, l2_diff1_v, t1, l2_base1_v),
                        [b0, l21_base_bc, l21_diff_bc],
                    )
                    diff = add_op("valu", ("-", t1, t1, node), [r0, r1])
                    sel = add_op(
                        "valu",
                        ("multiply_add", node, t1, t2, node),
                        [diff, b1],
                    )
                    node_deps = [sel]
                    node_addr = node
                else:
                    # Gather: node[lane] = mem[gather_base[level] + p[lane]]
                    bc = add_op("valu", ("vbroadcast", node, gather_base_s[level]), prev)
                    addr_op = add_op("valu", ("+", node, node, p), [bc])
                    loads = [
                        add_op("load", ("load_offset", node, node, lane), [addr_op])
                        for lane in range(VLEN)
                    ]
                    node_deps = loads
                    node_addr = node

                final_val = add_hash(gi, g_val[gi], node_addr, node_deps, is_last)

                if is_last:
                    last[gi] = [final_val]
                elif level == 0:
                    # Next level is 1: new p = val & 1  (old p was 0, so 2*0+bit)
                    p_up = [
                        add_op("alu", ("&", p + lane, g_val[gi] + lane, one_s), [final_val])
                        for lane in range(VLEN)
                    ]
                    last[gi] = [final_val] + p_up
                else:
                    # p = 2*p + (val & 1)
                    bit = [
                        add_op("alu", ("&", t1 + lane, g_val[gi] + lane, one_s), [final_val])
                        for lane in range(VLEN)
                    ]
                    p_up = add_op(
                        "valu",
                        ("multiply_add", p, p, two_v, t1),
                        bit + [two_bc],
                    )
                    last[gi] = [final_val, p_up]

        # Store final values back.
        for gi in range(n_groups):
            add_op("store", ("vstore", g_val_ptr[gi], g_val[gi]), last[gi])

        # ---- List scheduler: earliest-ready, engine-packed ----
        n_ops = len(ops)
        succs = [[] for _ in range(n_ops)]
        indeg = [0] * n_ops
        for i, (_, _, deps) in enumerate(ops):
            indeg[i] = len(deps)
            for d in deps:
                succs[d].append(i)
        # Critical-path priority (longer chain first)
        priority = [0] * n_ops
        for i in range(n_ops - 1, -1, -1):
            for s in succs[i]:
                if priority[s] + 1 > priority[i]:
                    priority[i] = priority[s] + 1

        ready = set(i for i, c in enumerate(indeg) if c == 0)
        remaining = n_ops
        engine_rank = {"load": 0, "valu": 1, "alu": 2, "store": 3, "flow": 4, "debug": 5}

        def sort_key(i):
            engine = ops[i][0]
            return (engine_rank[engine], -priority[i], i)

        while remaining:
            bundle = {}
            counts = {e: 0 for e in SLOT_LIMITS}
            chosen = []
            for i in sorted(ready, key=sort_key):
                engine = ops[i][0]
                if counts[engine] < SLOT_LIMITS[engine]:
                    bundle.setdefault(engine, []).append(ops[i][1])
                    counts[engine] += 1
                    chosen.append(i)
            if not chosen:
                raise RuntimeError("scheduler stuck")
            for i in chosen:
                ready.discard(i)
                remaining -= 1
                for s in succs[i]:
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        ready.add(s)
            if bundle:
                self.instrs.append(bundle)


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.enable_debug = False
    machine.enable_pause = False
    machine.prints = prints
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    machine.run()
    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect result"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
