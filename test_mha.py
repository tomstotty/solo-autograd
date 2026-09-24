"""Verifiable contract tests for Tensor.mha.

Multi-head scaled dot-product attention over flattened float lists:
queries q x h x d (self), keys k x h x d, values k x h x d, all
flattened token-major then head then dimension, plus an optional q x k
bool mask shared by every head. Standard library only; discovered by
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


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

    Per head: scores S = Q K^T / sqrt(d) accumulated in (i, j, r) order;
    per-row stable softmax over allowed positions gives P; Y = P V
    accumulated in (i, h, r, j) order. Masked positions stay 0.0.
    Returns (y, probs) with probs laid out head-major.

    Every accumulation is an explicit ascending loop starting from 0.0;
    built-in sum() must not be used because it does not guarantee strict
    left-to-right float addition and can differ by one ULP.
    """
    scale = math.sqrt(float(d))
    scores = [0.0] * (h * q * k)
    for hh in range(h):
        for i in range(q):
            for j in range(k):
                acc = 0.0
                for r in range(d):
                    acc += qs[(i * h + hh) * d + r] * ks[(j * h + hh) * d + r]
                scores[(hh * q + i) * k + j] = acc / scale
    allowed = allowed_positions(q, k, mask)
    probs = [0.0] * (h * q * k)
    for hh in range(h):
        for i in range(q):
            row = (hh * q + i) * k
            positions = allowed[i]
            m = max(scores[row + j] for j in positions)
            z = {j: math.exp(scores[row + j] - m) for j in positions}
            total = 0.0
            for j in positions:
                total += z[j]
            for j in positions:
                probs[row + j] = z[j] / total
    y = [0.0] * (q * h * d)
    for i in range(q):
        for hh in range(h):
            for r in range(d):
                acc = 0.0
                for j in allowed[i]:
                    acc += probs[(hh * q + i) * k + j] * vs[(j * h + hh) * d + r]
                y[(i * h + hh) * d + r] = acc
    return y, probs


def ref_mha_grad(qs, ks, vs, q, k, h, d, probs, g, mask=None):
    """Per head: H = G V^T, D = P*(H - row_sum(P*H)), dQ = D K/s,
    dK = D^T Q/s, dV = P^T G, with s = sqrt(d); masked positions
    contribute 0. Returns (dq, dk, dv). All accumulations are explicit
    ascending loops from 0.0, matching the prescribed operation order
    exactly.
    """
    scale = math.sqrt(float(d))
    allowed = allowed_positions(q, k, mask)
    h_mat = [0.0] * (h * q * k)
    d_mat = [0.0] * (h * q * k)
    for hh in range(h):
        for i in range(q):
            for j in range(k):
                acc = 0.0
                for r in range(d):
                    acc += g[(i * h + hh) * d + r] * vs[(j * h + hh) * d + r]
                h_mat[(hh * q + i) * k + j] = acc
    for hh in range(h):
        for i in range(q):
            row = (hh * q + i) * k
            row_sum = 0.0
            for j in allowed[i]:
                row_sum += probs[row + j] * h_mat[row + j]
            for j in allowed[i]:
                d_mat[row + j] = probs[row + j] * (h_mat[row + j] - row_sum)
    dq = [0.0] * (q * h * d)
    dk = [0.0] * (k * h * d)
    dv = [0.0] * (k * h * d)
    for hh in range(h):
        for i in range(q):
            for r in range(d):
                acc = 0.0
                for j in allowed[i]:
                    acc += d_mat[(hh * q + i) * k + j] * ks[(j * h + hh) * d + r]
                dq[(i * h + hh) * d + r] = acc / scale
    for hh in range(h):
        for j in range(k):
            for r in range(d):
                acc_k = 0.0
                acc_v = 0.0
                for i in range(q):
                    if j in allowed[i]:
                        acc_k += d_mat[(hh * q + i) * k + j] * qs[(i * h + hh) * d + r]
                        acc_v += probs[(hh * q + i) * k + j] * g[(i * h + hh) * d + r]
                dk[(j * h + hh) * d + r] = acc_k / scale
                dv[(j * h + hh) * d + r] = acc_v
    return dq, dk, dv


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


# q = 2 tokens, k = 3 keys, h = 2 heads, d = 2 dims.
Q = [0.3, -0.5, 1.2, 0.1, -0.6, 0.8, 0.4, -0.2]
K = [0.7, 0.2, -0.4, 0.9, 0.1, -0.8, 0.5, -0.3, 1.1, 0.6, -0.7, 0.4]
V = [1.0, -1.0, 0.5, 2.0, -1.5, 0.25, 0.75, -0.5, 1.5, -0.25, 0.3, 0.9]
MASK = [True, True, False, True, False, True]
G = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.3, -0.4]

CASES = [
    (Q, K, V, 2, 3, 2, 2, MASK),
    (Q, K, V, 2, 3, 2, 2, None),
    ([0.3, -0.5], [0.7, 0.2, -0.4, 0.9], [1.0, -1.0, 0.5, 2.0],
     1, 2, 2, 1, [True, False]),
    ([0.3, -0.5], [0.7, 0.2, -0.4, 0.9], [1.0, -1.0, 0.5, 2.0],
     1, 2, 1, 2, [True, False]),
    ([0.25], [0.5], [-0.75], 1, 1, 1, 1, [True]),
    ([0.25, -0.5], [0.5, 0.75], [-0.75, 1.25], 1, 1, 2, 1, None),
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

    def test_single_head_matches_attention(self):
        # With h == 1 the result must equal single-head attention.
        qs, ks, vs, q, k, d = Q[:4], K[:6], V[:6], 2, 3, 2
        out = Tensor(qs).mha(Tensor(ks), Tensor(vs), q, k, 1, d, MASK)
        expected, _ = ref_mha(qs, ks, vs, q, k, 1, d, MASK)
        self.assertEqual(out.data, expected)

    def test_single_allowed_key_copies_its_value(self):
        mask = [False, True, False, False, True, False]
        out = Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, mask)
        # Each query attends only to key 1, so each head row of every
        # query equals the matching head row of value token 1.
        expected = [-1.5, 0.25, 0.75, -0.5, -1.5, 0.25, 0.75, -0.5]
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
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2, MASK).backward(G)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, G, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_backward_no_mask_matches_reference(self):
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2).backward(G)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, G)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_fully_masked_key_gets_zero_key_value_grad(self):
        # Key 2 is masked for both queries, in every head.
        mask = [True, True, False, False, True, False]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.mha(key, value, 2, 3, 2, 2, mask).backward(G)
        # Token 2 occupies two head rows of length d in the flat layout.
        self.assertEqual(key.grad[8:12], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(value.grad[8:12], [0.0, 0.0, 0.0, 0.0])

    def test_only_parents_requiring_grad_receive_grads(self):
        q = Tensor(Q, True)
        key = Tensor(K, False)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(G)
        self.assertIsNotNone(q.grad)
        self.assertIsNone(key.grad)
        self.assertIsNotNone(value.grad)

    def test_no_grad_no_graph(self):
        out = Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, MASK)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward(G)

    def test_snapshot_survives_mutation_and_replacement(self):
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
        out.backward(G)
        expected_y, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        self.assertEqual(out.data, expected_y)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, G, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_shared_key_and_value_merges_roles(self):
        # The same tensor plays both roles: its grad must equal dK + dV.
        w = [0.7, 0.2, -0.4, 0.9, 0.1, -0.8, 0.5, -0.3, 1.1, 0.6, -0.7, 0.4]
        weights = [(-1.0) ** i * 0.375 for i in range(8)]
        passed, err = gradcheck(
            lambda t, qs=Q, weights=weights:
                Tensor(qs).mha(t, t, 2, 3, 2, 2, MASK).dot(Tensor(weights)),
            w,
        )
        self.assertTrue(passed, f"max abs error {err}")

        q = Tensor(Q, True)
        shared = Tensor(w, True)
        q.mha(shared, shared, 2, 3, 2, 2, MASK).backward(G)
        _, probs = ref_mha(Q, w, w, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, w, w, 2, 3, 2, 2, probs, G, MASK)
        self.assertEqual(shared.grad, [a + b for a, b in zip(dk, dv)])

    def test_all_three_parents_identical_merges_roles(self):
        a = [0.3, -0.5, 1.2, 0.1, -0.6, 0.8, 0.4, -0.2]
        weights = [(-1.0) ** i * 0.375 for i in range(8)]
        passed, err = gradcheck(
            lambda t, weights=weights:
                t.mha(t, t, 2, 2, 2, 2).dot(Tensor(weights)),
            a,
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_repeated_backward_accumulates(self):
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(G)
        out.backward(G)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, G, MASK)
        self.assertEqual(q.grad, [2.0 * x for x in dq])
        self.assertEqual(key.grad, [2.0 * x for x in dk])
        self.assertEqual(value.grad, [2.0 * x for x in dv])

    def test_failed_backward_leaves_existing_grads_unchanged(self):
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.mha(key, value, 2, 3, 2, 2, MASK)
        out.backward(G)
        saved_q = list(q.grad)
        saved_k = list(key.grad)
        saved_v = list(value.grad)
        bad_grads = (
            None,
            1.0,
            1,
            True,
            "x",
            [1.0],
            [1.0] * 9,
            [1.0, float("inf")] + [1.0] * 6,
            [1.0, 1] + [1.0] * 6,
        )
        for bad in bad_grads:
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    out.backward(bad)
                self.assertEqual(q.grad, saved_q)
                self.assertEqual(key.grad, saved_k)
                self.assertEqual(value.grad, saved_v)
        # A subsequent valid backward still accumulates on the first one.
        out.backward(G)
        _, probs = ref_mha(Q, K, V, 2, 3, 2, 2, MASK)
        dq, dk, dv = ref_mha_grad(Q, K, V, 2, 3, 2, 2, probs, G, MASK)
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

    def test_data_element_not_float(self):
        with self.assertRaises(TypeError):
            make_bad(data=[0.5] * 7 + [1]).mha(
                Tensor(K), Tensor(V), 2, 3, 2, 2
            )
        with self.assertRaises(TypeError):
            Tensor(Q).mha(make_bad(data=[True] * 12), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).mha(Tensor(K), make_bad(data=[0.5] * 11 + [2]), 2, 3, 2, 2)

    def test_data_non_finite(self):
        with self.assertRaises(ValueError):
            make_bad(data=[float("inf")] * 8).mha(
                Tensor(K), Tensor(V), 2, 3, 2, 2
            )
        with self.assertRaises(ValueError):
            Tensor(Q).mha(make_bad(data=[float("nan")] * 12), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(
                Tensor(K), make_bad(data=[float("-inf")] * 12), 2, 3, 2, 2
            )

    def test_requires_grad_not_bool(self):
        with self.assertRaises(TypeError):
            make_bad(data=list(Q), requires_grad=1).mha(
                Tensor(K), Tensor(V), 2, 3, 2, 2
            )

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor(Q[:-1]).mha(Tensor(K), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor(K[:-1]), Tensor(V), 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).mha(Tensor(K), Tensor(V[:-1]), 2, 3, 2, 2)

    def test_mask_type(self):
        for bad in (0, 1.0, "x", (True,) * 6):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_length(self):
        for bad in ([], [True], [True] * 5, [True] * 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_element_not_bool(self):
        for bad in ([1] * 6, [True, True, False, True, False, 1],
                    [None] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).mha(Tensor(K), Tensor(V), 2, 3, 2, 2, bad)

    def test_mask_row_all_false(self):
        with self.assertRaises(ValueError):
            Tensor(Q).mha(
                Tensor(K), Tensor(V), 2, 3, 2, 2,
                [False, False, False, True, False, True],
            )
        with self.assertRaises(ValueError):
            Tensor(Q).mha(
                Tensor(K), Tensor(V), 2, 3, 2, 2,
                [True, True, False, False, False, False],
            )

    def test_failed_forward_leaves_no_trace(self):
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        with self.assertRaises(ValueError):
            q.mha(key, value, 2, 3, 2, 2, [False] * 3 + [True] * 3)
        self.assertIsNone(q.grad)
        self.assertIsNone(key.grad)
        self.assertIsNone(value.grad)


if __name__ == "__main__":
    unittest.main()
