"""Verifiable contract tests for Tensor.mha.

Multi-head scaled dot-product attention over flattened row-major float
lists: queries q*h*d (self), keys k*h*d, values k*h*d, optional q*k
bool mask shared by every head. Tensors are flattened by token, head and
feature dimension: index ((token*h + head)*d + r). Standard library only;
discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def qidx(i, head, r, h, d):
    """Flat query/output index for token i, head, feature r."""
    return (i * h + head) * d + r


def kidx(j, head, r, h, d):
    """Flat key/value index for token j, head, feature r."""
    return (j * h + head) * d + r


def allowed_positions(q, k, mask):
    """Allowed key indices per query row for the given mask or None."""
    rows = []
    for i in range(q):
        if mask is None:
            rows.append(list(range(k)))
        else:
            rows.append([j for j in range(k) if mask[i * k + j]])
    return rows


def ref_mha(qs, ks, vs, q, k, h, d, mask=None):
    """Independent hand computation in the prescribed operation order.

    Scores S = Q K^T / sqrt(d) per head, accumulated in (i, j, head, r)
    order; per-head per-row stable softmax over allowed positions gives P;
    Y = P V accumulated in (i, head, r, j) order. Masked positions stay
    0.0. Returns (y, probs_flat) with probs indexed ((head*q + i)*k + j).

    Every accumulation is an explicit ascending loop starting from 0.0;
    built-in sum() must not be used because it does not guarantee strict
    left-to-right float addition and can differ by one ULP.
    """
    scale = math.sqrt(float(d))
    scores = [0.0] * (h * q * k)
    for i in range(q):
        for j in range(k):
            for head in range(h):
                acc = 0.0
                for r in range(d):
                    acc += (
                        qs[qidx(i, head, r, h, d)]
                        * ks[kidx(j, head, r, h, d)]
                    )
                scores[(head * q + i) * k + j] = acc / scale
    allowed = allowed_positions(q, k, mask)
    probs = [0.0] * (h * q * k)
    for i in range(q):
        positions = allowed[i]
        for head in range(h):
            sbase = (head * q + i) * k
            m = max(scores[sbase + j] for j in positions)
            z = {j: math.exp(scores[sbase + j] - m) for j in positions}
            total = 0.0
            for j in positions:
                total += z[j]
            for j in positions:
                probs[sbase + j] = z[j] / total
    y = [0.0] * (q * h * d)
    for i in range(q):
        for head in range(h):
            for r in range(d):
                acc = 0.0
                for j in allowed[i]:
                    acc += (
                        probs[(head * q + i) * k + j]
                        * vs[kidx(j, head, r, h, d)]
                    )
                y[qidx(i, head, r, h, d)] = acc
    return y, probs


def ref_mha_grad(qs, ks, vs, q, k, h, d, probs, g, mask=None):
    """H = G V^T, D = P*(H - row_sum(P*H)), dQ = D K/s, dK = D^T Q/s,
    dV = P^T G per head, with s = sqrt(d); masked positions contribute 0.
    Returns (dq, dk, dv). All accumulations are explicit ascending loops
    from 0.0, matching the prescribed operation order exactly.
    """
    scale = math.sqrt(float(d))
    allowed = allowed_positions(q, k, mask)
    h_mat = [0.0] * (h * q * k)
    d_mat = [0.0] * (h * q * k)
    for i in range(q):
        for head in range(h):
            for j in allowed[i]:
                acc = 0.0
                for r in range(d):
                    acc += (
                        g[qidx(i, head, r, h, d)]
                        * vs[kidx(j, head, r, h, d)]
                    )
                h_mat[(head * q + i) * k + j] = acc
    for i in range(q):
        for head in range(h):
            sbase = (head * q + i) * k
            row_sum = 0.0
            for j in allowed[i]:
                row_sum += probs[sbase + j] * h_mat[sbase + j]
            for j in allowed[i]:
                d_mat[sbase + j] = (
                    probs[sbase + j] * (h_mat[sbase + j] - row_sum)
                )
    dq = [0.0] * (q * h * d)
    dk = [0.0] * (k * h * d)
    dv = [0.0] * (k * h * d)
    for i in range(q):
        for head in range(h):
            for r in range(d):
                acc = 0.0
                for j in allowed[i]:
                    acc += (
                        d_mat[(head * q + i) * k + j]
                        * ks[kidx(j, head, r, h, d)]
                    )
                dq[qidx(i, head, r, h, d)] = acc / scale
    for j in range(k):
        for head in range(h):
            for r in range(d):
                acc_k = 0.0
                acc_v = 0.0
                for i in range(q):
                    if j in allowed[i]:
                        acc_k += (
                            d_mat[(head * q + i) * k + j]
                            * qs[qidx(i, head, r, h, d)]
                        )
                        acc_v += (
                            probs[(head * q + i) * k + j]
                            * g[qidx(i, head, r, h, d)]
                        )
                dk[kidx(j, head, r, h, d)] = acc_k / scale
                dv[kidx(j, head, r, h, d)] = acc_v
    return dq, dk, dv


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


# q=2, k=3, h=2, d=2: Q length 8, K/V length 12.
Q = [0.3, -0.5, 1.2, 0.1, -0.6, 0.8, 0.4, -0.2]
K = [0.7, 0.2, -0.4, 0.9, 0.1, -0.8, 0.5, -0.3, 0.6, 0.2, -0.1, 0.7]
V = [1.0, -1.0, 0.5, 2.0, -1.5, 0.25, 0.75, -0.5, 1.25, 0.4, -0.7, 0.9]
MASK = [True, True, False, True, False, True]
W8 = [(-1.0) ** i * 0.375 for i in range(8)]

CASES = [
    (Q, K, V, 2, 3, 2, 2, MASK),
    (Q, K, V, 2, 3, 2, 2, None),
    # A single head reduces to ordinary attention.
    ([0.3, -0.5], [0.7, 0.2, -0.4, 0.9], [1.0, -1.0, 0.5, 2.0],
     1, 2, 1, 2, [True, False]),
    ([0.3, -0.5, 1.2, 0.1, -0.6, 0.8],
     [0.7, 0.2, -0.4, 0.9],
     [1.0, -1.0, 0.5, 2.0], 3, 2, 1, 2,
     [True, False, True, True, False, True]),
    # q=1, k=1, h=3, d=1: one token, three heads, scalar features.
    ([0.25, -0.4, 0.8], [0.5, 0.2, -0.6], [-0.75, 0.3, 1.1],
     1, 1, 3, 1, [True]),
    ([0.25, -0.4, 0.8], [0.5, 0.2, -0.6], [-0.75, 0.3, 1.1],
     1, 1, 3, 1, None),
]


class ForwardValueTest(unittest.TestCase):

    def test_hand_computed_cases(self):
        for qs, ks, vs, q, k, h, d, mask in CASES:
            with self.subTest(q=q, k=k, h=h, d=d, mask=mask):
                out = Tensor(qs).mha(
                    Tensor(ks), Tensor(vs), q, k, h, d, mask
                )
                expected, _ = ref_mha(qs, ks, vs, q, k, h, d, mask)
                self.assertEqual(out.data, expected)
                self.assertEqual(len(out.data), q * h * d)

    def test_heads_are_independent_slices_of_attention(self):
        # Each head of mha must equal a single-head attention call on the
        # per-head flattened slices, interleaved back together.
        q, k, h, d = 2, 3, 2, 2
        result = Tensor(Q).mha(Tensor(K), Tensor(V), q, k, h, d, MASK)
        expected = [0.0] * (q * h * d)
        for head in range(h):
            qs = [Q[qidx(i, head, r, h, d)] for i in range(q) for r in range(d)]
            ks = [K[kidx(j, head, r, h, d)] for j in range(k) for r in range(d)]
            vs = [V[kidx(j, head, r, h, d)] for j in range(k) for r in range(d)]
            part = Tensor(qs).attention(Tensor(ks), Tensor(vs), q, k, d, MASK)
            for i in range(q):
                for r in range(d):
                    expected[qidx(i, head, r, h, d)] = part.data[i * d + r]
        self.assertEqual(result.data, expected)

    def test_masked_probabilities_are_zero_in_every_head(self):
        result = Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, MASK)
        _, expected_p = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        self.assertEqual(
            result.data, ref_mha(Q, K, V, 2, 3, 2, 2, MASK)[0]
        )
        # Key 2 masked for query 0 and key 1 masked for query 1, both heads.
        for head in range(2):
            self.assertEqual(expected_p[(head * 2 + 0) * 3 + 2], 0.0)
            self.assertEqual(expected_p[(head * 2 + 1) * 3 + 1], 0.0)

    def test_single_allowed_key_copies_its_value(self):
        mask = [False, True, False, False, True, False]
        out = Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, mask)
        expected = [0.0] * 8
        for i in range(2):
            for head in range(2):
                for r in range(2):
                    expected[qidx(i, head, r, 2, 2)] = V[
                        kidx(1, head, r, 2, 2)
                    ]
        self.assertEqual(out.data, expected)

    def test_gradcheck_queries(self):
        for qs, ks, vs, q, k, h, d, mask in CASES:
            with self.subTest(q=q, k=k, h=h, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * h * d)]
                passed, err = gradcheck(
                    lambda t, ks=ks, vs=vs, q=q, k=k, h=h, d=d, mask=mask,
                    w=weights:
                        t.mha(Tensor(ks), Tensor(vs), q, k, h, d, mask)
                        .dot(Tensor(w)),
                    qs,
                )
                self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_keys(self):
        for qs, ks, vs, q, k, h, d, mask in CASES:
            with self.subTest(q=q, k=k, h=h, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * h * d)]
                passed, err = gradcheck(
                    lambda t, qs=qs, vs=vs, q=q, k=k, h=h, d=d, mask=mask,
                    w=weights:
                        Tensor(qs).mha(t, Tensor(vs), q, k, h, d, mask)
                        .dot(Tensor(w)),
                    ks,
                )
                self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_values(self):
        for qs, ks, vs, q, k, h, d, mask in CASES:
            with self.subTest(q=q, k=k, h=h, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * h * d)]
                passed, err = gradcheck(
                    lambda t, qs=qs, ks=ks, q=q, k=k, h=h, d=d, mask=mask,
                    w=weights:
                        Tensor(qs).mha(Tensor(ks), t, q, k, h, d, mask)
                        .dot(Tensor(w)),
                    vs,
                )
                self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_matches_reference(self):
        g = list(W8)
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2, MASK).backward(g)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, g, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_backward_no_mask_matches_reference(self):
        g = list(W8)
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2).backward(g)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, g)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_fully_masked_key_gets_zero_key_value_grad(self):
        # Key 2 is masked for both queries, so both heads' key/value rows
        # for token 2 get zero grad.
        mask = [True, True, False, False, True, False]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2, mask).backward(list(W8))
        for head in range(2):
            for r in range(2):
                self.assertEqual(key.grad[kidx(2, head, r, 2, 2)], 0.0)
                self.assertEqual(value.grad[kidx(2, head, r, 2, 2)], 0.0)

    def test_only_parents_requiring_grad_receive_grads(self):
        q = Tensor(Q, True)
        key = Tensor(K, False)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(list(W8))
        self.assertIsNotNone(q.grad)
        self.assertIsNone(key.grad)
        self.assertIsNotNone(value.grad)

    def test_no_grad_no_graph(self):
        out = Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, MASK)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward([1.0] * 8)

    def test_snapshot_survives_mutation_and_replacement(self):
        g = list(W8)
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        mask = list(MASK)
        out = q.mha(key, value, 2, 3, 2, 2, mask)
        q.data[0] = 999.0
        q.data = [9.0] * 8
        key.data = [8.0] * 12
        value.data = [7.0] * 12
        mask[0] = False
        mask.append(True)
        out.backward(g)
        expected_y, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        self.assertEqual(out.data, expected_y)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, g, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_shared_key_and_value_merges_roles(self):
        # The same tensor plays both roles: its grad must equal dK + dV.
        w = list(K)
        weights = list(W8)
        passed, err = gradcheck(
            lambda t, qs=Q, weights=weights:
                Tensor(qs).mha(t, t, 2, 3, 2, 2, MASK).dot(Tensor(weights)),
            w,
        )
        self.assertTrue(passed, f"max abs error {err}")

        q = Tensor(Q, True)
        shared = Tensor(w, True)
        g = list(W8)
        q.mha(shared, shared, 2, 3, 2, 2, MASK).backward(g)
        _, probs = ref_mha(Q, w, w, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, w, w, 2, 3, 2, 2, probs, g, MASK)
        self.assertEqual(shared.grad, [a + b for a, b in zip(dk, dv)])

    def test_all_three_parents_identical_merges_roles(self):
        # q=2, k=2, h=2, d=2: all three roles share one length-8 tensor.
        a = list(Q)
        weights = list(W8)
        passed, err = gradcheck(
            lambda t, weights=weights:
                t.mha(t, t, 2, 2, 2, 2).dot(Tensor(weights)),
            a,
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_repeated_backward_accumulates(self):
        g = list(W8)
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(g)
        out.backward(g)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, g, MASK)
        self.assertEqual(q.grad, [2.0 * x for x in dq])
        self.assertEqual(key.grad, [2.0 * x for x in dk])
        self.assertEqual(value.grad, [2.0 * x for x in dv])

    def test_failed_backward_leaves_existing_grads_unchanged(self):
        g = list(W8)
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(g)
        saved_q, saved_k, saved_v = list(q.grad), list(key.grad), list(value.grad)
        for bad in (None, 1.0, True, 1, [1.0], [1.0] * 9,
                    [1.0, float("inf")] + [1.0] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    out.backward(bad)
                self.assertEqual(q.grad, saved_q)
                self.assertEqual(key.grad, saved_k)
                self.assertEqual(value.grad, saved_v)
        # A subsequent valid backward still accumulates on the first one.
        out.backward(g)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, g, MASK)
        self.assertEqual(q.grad, [2.0 * x for x in dq])
        self.assertEqual(key.grad, [2.0 * x for x in dk])
        self.assertEqual(value.grad, [2.0 * x for x in dv])


class ErrorContractTest(unittest.TestCase):

    def test_key_value_not_tensor(self):
        for bad in (None, "x", [0.5], 1, (K,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(bad, Tensor(V), 2, 3, 2, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), bad, 2, 3, 2, 2)

    def test_dims_type(self):
        for bad in (None, 1.0, 0.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), bad, 3, 2, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, bad, 2, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, bad, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_dims_bool_is_type_error(self):
        with self.assertRaises(TypeError):
            Tensor(Q).mha(Tensor(K), Tensor(V), True, 3, 2, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).mha(Tensor(K), Tensor(V), 2, False, 2, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, True, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, False)

    def test_dims_non_positive(self):
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), bad, 3, 2, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, bad, 2, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, bad, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).mha(Tensor(K), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(make_bad(data=0.5), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor(K), make_bad(data=0.5), 2, 3, 2, 2)

    def test_data_empty_list(self):
        empty = make_bad(data=[])
        with self.assertRaises(ValueError):
            empty.mha(Tensor(K), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(empty, Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor(K), empty, 2, 3, 2, 2)

    def test_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.1] * 7).mha(Tensor(K), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor([0.1] * 11), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor(K), Tensor([0.1] * 13), 2, 3, 2, 2)

    def test_element_type(self):
        for bad in ([0.1, 1] + [0.3] * 10,
                    [0.1, True] + [0.3] * 10,
                    [0.1, "x"] + [0.3] * 10,
                    [0.1, None] + [0.3] * 10):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(make_bad(data=bad), Tensor(V), 2, 3, 2, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), make_bad(data=bad), 2, 3, 2, 2)

    def test_query_element_type(self):
        bad = [1] + [0.3] * 7
        with self.assertRaises(TypeError):
            make_bad(data=bad).mha(Tensor(K), Tensor(V), 2, 3, 2, 2)

    def test_requires_grad_type(self):
        for role in ("q", "k", "v"):
            with self.subTest(role=role):
                qt = make_bad(requires_grad=1) if role == "q" else Tensor(Q)
                kt = make_bad(requires_grad=1) if role == "k" else Tensor(K)
                vt = make_bad(requires_grad=1) if role == "v" else Tensor(V)
                with self.assertRaises(TypeError):
                    qt.mha(kt, vt, 2, 3, 2, 2)

    def test_non_finite_elements(self):
        for bad in ([0.1, float("inf")] + [0.3] * 10,
                    [float("nan"), 0.2] + [0.3] * 10,
                    [0.1, -float("inf")] + [0.3] * 10):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(make_bad(data=bad), Tensor(V), 2, 3, 2, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), make_bad(data=bad), 2, 3, 2, 2)

    def test_dims_checked_before_data(self):
        # A non-positive dimension is a ValueError even with malformed data.
        bad = make_bad(data=[0.1, 1])
        with self.assertRaises(ValueError):
            bad.mha(Tensor(K), Tensor(V), 0, 3, 2, 2)

    def test_mask_type(self):
        for bad in (True, False, 1, 1.0, "tt", (True,) * 6, {0: True}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_element_type(self):
        for bad in ([True, 1, True, True, True, True],
                    [True, True, 0.0, True, True, True],
                    [True, None, True, True, True, True],
                    [True, "t", True, True, True, True]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_length(self):
        for bad in ([], [True] * 5, [True] * 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_row_without_true(self):
        for bad in ([False, False, False, True, False, True],
                    [True, True, False, False, False, False],
                    [False] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_non_finite_forward_raises_value_error(self):
        big = [1.7e300] * 8
        with self.assertRaises(ValueError):
            big_t = Tensor(big)
            big_t.mha(Tensor([1.7e300] * 12), Tensor(V), 2, 3, 2, 2)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.q = Tensor(Q, True)
        self.out = self.q.mha(
            Tensor(K, True), Tensor(V, True), 2, 3, 2, 2, MASK
        )

    def test_upstream_omitted(self):
        with self.assertRaises(ValueError):
            self.out.backward()

    def test_upstream_scalar_for_vector_out(self):
        with self.assertRaises(ValueError):
            self.out.backward(1.0)

    def test_upstream_bool_int_are_value_errors(self):
        for bad in (True, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.out.backward(bad)

    def test_upstream_type_errors(self):
        for bad in (None, "x", (1.0,) * 8):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_length(self):
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * 7)
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * 9)

    def test_upstream_element_type(self):
        for bad in ([1.0, 1] + [1.0] * 6,
                    [1.0, True] + [1.0] * 6,
                    [1.0, "x"] + [1.0] * 6,
                    [1.0, None] + [1.0] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_non_finite(self):
        with self.assertRaises(ValueError):
            self.out.backward([float("inf")] + [1.0] * 7)
        with self.assertRaises(ValueError):
            self.out.backward([1.0, float("nan")] + [1.0] * 6)

    def test_failed_upstream_leaves_no_grad(self):
        for bad in (None, True, 1, 1.0, [1.0],
                    [1.0, float("inf")] + [1.0] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.out.backward(bad)
                self.assertIsNone(self.q.grad)


if __name__ == "__main__":
    unittest.main()
