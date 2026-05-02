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
        h3a_alu_rounds = {15}

        # ---- scratch allocations ----
        one_s = self.alloc_scratch("one_s")
        two_s = self.alloc_scratch("two_s")
        four_s = self.alloc_scratch("four_s")
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        four_v = self.alloc_scratch("four_v", VLEN)

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
        l1_val1_v = self.alloc_scratch("l1_val1_v", VLEN)
        l2_base0_v = self.alloc_scratch("l2_base0_v", VLEN)
        l2_diff0_v = self.alloc_scratch("l2_diff0_v", VLEN)
        l2_base1_v = self.alloc_scratch("l2_base1_v", VLEN)
        l2_diff1_v = self.alloc_scratch("l2_diff1_v", VLEN)

        # Per-level base pointer into forest_values for gathered levels.
        gather_base_s = {}
        for level in range(4, forest_height + 1):
            gather_base_s[level] = self.alloc_scratch(f"gb{level}_s")
        # Pre-broadcast gather bases once at setup (saves 1 valu/group/round).
        gather_base_v = {}
        for level in gather_base_s:
            gather_base_v[level] = self.alloc_scratch(f"gb{level}_v", VLEN)

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
            g_t2.append(g_node[-1])
            g_val_ptr.append(self.alloc_scratch(f"g{gi}_vp"))

        tree_s = [self.alloc_scratch(f"tree{i}_s") for i in range(7)]
        l3_tree_s = [self.alloc_scratch(f"l3_tree{i}_s") for i in range(8)]
        l3_tree_v = [self.alloc_scratch(f"l3_tree{i}_v", VLEN) for i in range(8)]
        l1_diff_s = self.alloc_scratch("l1_diff_s")
        l2_diff0_s = self.alloc_scratch("l2_diff0_s")
        l2_diff1_s = self.alloc_scratch("l2_diff1_s")
        l2_b1_tmps = [self.alloc_scratch(f"l2_b1_tmp{i}", VLEN) for i in range(1)]
        l3_tmp0 = [self.alloc_scratch(f"l3_tmp0_{i}", VLEN) for i in range(2)]
        l3_tmp1 = [self.alloc_scratch(f"l3_tmp1_{i}", VLEN) for i in range(2)]
        l3_tmp2 = [self.alloc_scratch(f"l3_tmp2_{i}", VLEN) for i in range(2)]

        def emit_loads(slots):
            for i in range(0, len(slots), SLOT_LIMITS["load"]):
                self.emit({"load": slots[i : i + SLOT_LIMITS["load"]]})

        # ---- Init (outside the scheduled body) ----
        # Emit const loads via scheduler so they can overlap with early valu
        # (broadcasts) and vloads.
        const_load_ops = []  # (dest, val) tuples
        const_load_ops.append((one_s, 1))
        const_load_ops.append((two_s, 2))
        const_load_ops.append((four_s, 4))
        for (s1, s2), (c1, c2) in zip(hash_scalars, hash_const_values):
            const_load_ops.append((s1, c1))
            const_load_ops.append((s2, c2))
        for level, dest in gather_base_s.items():
            const_load_ops.append((dest, forest_values_p + (1 << level) - 1))
        for i, dest in enumerate(tree_s):
            const_load_ops.append((dest, forest_values_p + i))
        for i, dest in enumerate(l3_tree_s):
            const_load_ops.append((dest, forest_values_p + 7 + i))

        # ---- Scheduled body ----
        ops = []

        def add_op(engine, slot, deps=()):
            idx = len(ops)
            ops.append((engine, slot, tuple(deps)))
            return idx

        # Emit const loads as scheduler ops (no deps).
        const_ops = {}  # dest -> op_idx
        for dest, val in const_load_ops:
            const_ops[dest] = add_op("load", ("const", dest, val))
        # Load the first 7 tree values (levels 0, 1, 2).
        tree_load_ops = []
        for dest in tree_s:
            tree_load_ops.append(add_op("load", ("load", dest, dest), [const_ops[dest]]))
        l3_tree_load_ops = []
        for dest in l3_tree_s:
            l3_tree_load_ops.append(add_op("load", ("load", dest, dest), [const_ops[dest]]))

        # Broadcasts & level-setup subs as deps in the scheduled body, so they
        # pipeline with early group work.
        diff_l1 = add_op("alu", ("-", l1_diff_s, tree_s[2], tree_s[1]), [tree_load_ops[1], tree_load_ops[2]])

        one_bc = add_op("valu", ("vbroadcast", one_v, one_s), [const_ops[one_s]])
        two_bc = add_op("valu", ("vbroadcast", two_v, two_s), [const_ops[two_s]])
        four_bc = add_op("valu", ("vbroadcast", four_v, four_s), [const_ops[four_s]])
        hc_bcs = []
        for hi, ((v1, v2), (s1, s2)) in enumerate(zip(hash_vecs, hash_scalars)):
            hc_bcs.append(add_op("valu", ("vbroadcast", v1, s1), [const_ops[s1]]))
            if hi in (1, 3):
                hc_bcs.append(const_ops[s2])
            else:
                hc_bcs.append(add_op("valu", ("vbroadcast", v2, s2), [const_ops[s2]]))
        root_bc = add_op("valu", ("vbroadcast", root_v, tree_s[0]), [tree_load_ops[0]])
        l1_base_bc = add_op("valu", ("vbroadcast", l1_base_v, tree_s[1]), [tree_load_ops[1]])
        l1_diff_bc = add_op("valu", ("vbroadcast", l1_diff_v, l1_diff_s), [diff_l1])
        l1_val1_bc = add_op("valu", ("vbroadcast", l1_val1_v, tree_s[2]), [tree_load_ops[2]])
        l20_base_bc = add_op("valu", ("vbroadcast", l2_base0_v, tree_s[3]), [tree_load_ops[3]])
        l20_diff_bc = add_op("valu", ("vbroadcast", l2_diff0_v, tree_s[4]), [tree_load_ops[4]])
        l21_base_bc = add_op("valu", ("vbroadcast", l2_base1_v, tree_s[5]), [tree_load_ops[5]])
        l21_diff_bc = add_op("valu", ("vbroadcast", l2_diff1_v, tree_s[6]), [tree_load_ops[6]])

        # Broadcast gather-base scalars to vectors once.
        gb_bcs = {}
        for level, s_addr in gather_base_s.items():
            gb_bcs[level] = add_op("valu", ("vbroadcast", gather_base_v[level], s_addr),
                                    [const_ops[s_addr]])
        l3_bcs = [
            add_op("valu", ("vbroadcast", vec, scalar), [load_op])
            for vec, scalar, load_op in zip(l3_tree_v, l3_tree_s, l3_tree_load_ops)
        ]

        hc_all_ready = hc_bcs + [one_bc, two_bc]

        # g_val_ptr: base + gi*VLEN. Compute via flow add_imm (free engine)
        # to avoid using 32 load slots for consts.
        val_base_s = self.alloc_scratch("val_base_s")
        const_ops[val_base_s] = add_op("load", ("const", val_base_s, inp_values_p))
        ptr_loads = []
        vloads = []
        for gi in range(n_groups):
            # Mix: every other group via flow, rest via load to balance engines
            if gi % 2 == 0:
                pl = add_op("flow", ("add_imm", g_val_ptr[gi], val_base_s, gi * VLEN), [const_ops[val_base_s]])
            else:
                pl = add_op("load", ("const", g_val_ptr[gi], inp_values_p + gi * VLEN))
            ptr_loads.append(pl)
            vl = add_op("load", ("vload", g_val[gi], g_val_ptr[gi]), [pl])
            vloads.append(vl)

        def add_hash(gi, val_in, node_addr, node_deps, is_last, round_i):
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
            h1b = [
                add_op("alu", (">>", t2 + lane, val + lane, hash_scalars[1][1]), [h0])
                for lane in range(VLEN)
            ]
            h1c = add_op("valu", ("^", val, t1, t2), [h1a] + h1b)
            # stage 2: a*33 + c1
            h2 = add_op(
                "valu",
                ("multiply_add", val, val, hash_vecs[2][0], hash_vecs[2][1]),
                [h1c],
            )
            # stage 3: (a + c1) ^ (a << 9)
            if round_i in h3a_alu_rounds:
                h3a = [
                    add_op("alu", ("+", t1 + lane, val + lane, hash_scalars[3][0]), [h2])
                    for lane in range(VLEN)
                ]
            else:
                h3a = add_op("valu", ("+", t1, val, hash_vecs[3][0]), [h2])
            h3b = [
                add_op("alu", ("<<", t2 + lane, val + lane, hash_scalars[3][1]), [h2])
                for lane in range(VLEN)
            ]
            h3a_deps = h3a if isinstance(h3a, list) else [h3a]
            h3c = add_op("valu", ("^", val, t1, t2), h3a_deps + h3b)
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

        # last_val[gi] = deps producing final_val (for val chain to next x0)
        # last_p[gi]   = deps producing latest p_up (for p-reading in next round)
        last_val = [[vl] + hc_all_ready for vl in vloads]
        last_p = [[] for _ in range(n_groups)]
        l2_tmp_last = [None] * len(l2_b1_tmps)
        l3_tmp_last = [None] * len(l3_tmp0)

        for round_i in range(rounds):
            level = round_i % (forest_height + 1)
            is_last = (round_i == rounds - 1)
            for gi in range(n_groups):
                p = g_p[gi]
                node = g_node[gi]
                t1 = g_t1[gi]
                t2 = g_t2[gi]
                vdeps = last_val[gi]
                pdeps = last_p[gi]

                if level == 0:
                    # Level-0: node_addr = root_v, no p needed.
                    node_deps = vdeps + [root_bc]
                    node_addr = root_v
                elif level == 1:
                    # p ∈ {0,1}; vselect picks between the two tree values
                    # (precomputed as broadcasts l1_base_v and l1_base_v+diff).
                    sel = add_op(
                        "flow",
                        ("vselect", node, p, l1_val1_v, l1_base_v),
                        vdeps + pdeps + [l1_base_bc, l1_val1_bc],
                    )
                    node_deps = [sel]
                    node_addr = node
                elif level == 2:
                    b1_tmp_i = gi % len(l2_b1_tmps)
                    b1_deps = vdeps + pdeps + [two_bc]
                    if l2_tmp_last[b1_tmp_i] is not None:
                        b1_deps.append(l2_tmp_last[b1_tmp_i])
                    b1_addr = l2_b1_tmps[b1_tmp_i]
                    b1 = add_op("valu", ("&", b1_addr, p, two_v), b1_deps)
                    r0 = add_op("flow", ("vselect", node, t1, l2_diff0_v, l2_base0_v),
                                vdeps + pdeps + [l20_base_bc, l20_diff_bc])
                    r1 = add_op("flow", ("vselect", t1, t1, l2_diff1_v, l2_base1_v),
                                [r0, l21_base_bc, l21_diff_bc])
                    sel = add_op("flow", ("vselect", node, b1_addr, t1, node), [r1, b1])
                    l2_tmp_last[b1_tmp_i] = sel
                    node_deps = [sel]
                    node_addr = node
                elif level == 3:
                    # Eight possible nodes exist at level 3, so select from
                    # broadcasts instead of repeating scatter loads per item.
                    tmp_i = gi % len(l3_tmp0)
                    tmp0 = l3_tmp0[tmp_i]
                    tmp1 = l3_tmp1[tmp_i]
                    tmp2 = l3_tmp2[tmp_i]
                    start_deps = vdeps + pdeps
                    if l3_tmp_last[tmp_i] is not None:
                        start_deps.append(l3_tmp_last[tmp_i])

                    r0 = add_op("flow", ("vselect", node, t1, l3_tree_v[1], l3_tree_v[0]), start_deps + l3_bcs[:2])
                    r1 = add_op("flow", ("vselect", tmp0, t1, l3_tree_v[3], l3_tree_v[2]), [r0] + l3_bcs[2:4])
                    b1_alu = [
                        add_op("alu", ("&", tmp1 + lane, p + lane, two_s), [r1])
                        for lane in range(VLEN)
                    ]
                    n0 = add_op("flow", ("vselect", node, tmp1, tmp0, node), b1_alu + [r1])

                    # t1 still holds b0 from b0_alu (unchanged by b1_alu/n0);
                    # no need to recompute.
                    r2 = add_op("flow", ("vselect", tmp0, t1, l3_tree_v[5], l3_tree_v[4]), [n0] + l3_bcs[4:6])
                    r3 = add_op("flow", ("vselect", tmp2, t1, l3_tree_v[7], l3_tree_v[6]), [r2] + l3_bcs[6:8])
                    # tmp1 still holds b1; use it directly as cond
                    n1 = add_op("flow", ("vselect", tmp0, tmp1, tmp2, tmp0), [r3])
                    b2 = add_op("valu", ("&", tmp1, p, four_v), [n1, four_bc])
                    sel = add_op("flow", ("vselect", node, tmp1, tmp0, node), [b2, n1, n0])
                    l3_tmp_last[tmp_i] = sel
                    node_deps = [sel]
                    node_addr = node
                else:
                    # Gather: addr = base_v + p (one fewer valu op by using
                    # pre-broadcasted base vector)
                    addr_op = add_op("valu", ("+", node, gather_base_v[level], p),
                                      vdeps + pdeps + [gb_bcs[level]])
                    loads = [
                        add_op("load", ("load_offset", node, node, lane), [addr_op])
                        for lane in range(VLEN - 1, -1, -1)
                    ]
                    node_deps = loads
                    node_addr = node

                final_val = add_hash(gi, g_val[gi], node_addr, node_deps, is_last, round_i)

                if is_last or level == forest_height:
                    last_val[gi] = [final_val]
                    last_p[gi] = []
                elif level == 0:
                    p_up = [
                        add_op("alu", ("&", p + lane, g_val[gi] + lane, one_s),
                               [final_val] + pdeps)
                        for lane in range(VLEN)
                    ]
                    last_val[gi] = [final_val]
                    last_p[gi] = p_up
                else:
                    bit = [
                        add_op("alu", ("&", t1 + lane, g_val[gi] + lane, one_s), [final_val])
                        for lane in range(VLEN)
                    ]
                    p_up = add_op(
                        "valu",
                        ("multiply_add", p, p, two_v, t1),
                        bit + [two_bc] + pdeps,
                    )
                    last_val[gi] = [final_val]
                    last_p[gi] = [p_up]

        # Store final values back.
        for gi in range(n_groups):
            add_op("store", ("vstore", g_val_ptr[gi], g_val[gi]), last_val[gi])

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
            return (engine_rank[engine], -priority[i], -i)

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
