"""Verifiable contract tests for Tensor.attention.

Single-head scaled dot-product attention over flattened row-major float
lists: queries q x d (self), keys k x d, values k x d, optional q x k
bool mask. Standard library only; discovered by
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


def ref_attention(qs, ks, vs, q, k, d, mask=None):
    """Independent hand computation in the prescribed operation order.

    Scores S = Q K^T / sqrt(d) accumulated in (i, j, r) order; per-row
    stable softmax over allowed positions gives P; Y = P V accumulated
    in (i, r, j) order. Masked positions stay 0.0. Returns (y, probs).

    Every accumulation is an explicit ascending loop starting from 0.0;
    built-in sum() must not be used because it does not guarantee strict
    left-to-right float addition and can differ by one ULP.
    """
    scale = math.sqrt(float(d))
    scores = [0.0] * (q * k)
    for i in range(q):
        for j in range(k):
            acc = 0.0
            for r in range(d):
                acc += qs[i * d + r] * ks[j * d + r]
            scores[i * k + j] = acc / scale
    allowed = allowed_positions(q, k, mask)
    probs = [0.0] * (q * k)
    for i in range(q):
        positions = allowed[i]
        m = max(scores[i * k + j] for j in positions)
        z = {j: math.exp(scores[i * k + j] - m) for j in positions}
        total = 0.0
        for j in positions:
            total += z[j]
        for j in positions:
            probs[i * k + j] = z[j] / total
    y = [0.0] * (q * d)
    for i in range(q):
        for r in range(d):
            acc = 0.0
            for j in allowed[i]:
                acc += probs[i * k + j] * vs[j * d + r]
            y[i * d + r] = acc
    return y, probs


def ref_attention_grad(qs, ks, vs, q, k, d, probs, g, mask=None):
    """H = G V^T, D = P*(H - row_sum(P*H)), dQ = D K/s, dK = D^T Q/s,
    dV = P^T G, with s = sqrt(d); masked positions contribute 0.
    Returns (dq, dk, dv). All accumulations are explicit ascending loops
    from 0.0, matching the prescribed operation order exactly.
    """
    scale = math.sqrt(float(d))
    allowed = allowed_positions(q, k, mask)
    h_mat = [[0.0] * k for _ in range(q)]
    d_mat = [[0.0] * k for _ in range(q)]
    for i in range(q):
        for j in allowed[i]:
            acc = 0.0
            for r in range(d):
                acc += g[i * d + r] * vs[j * d + r]
            h_mat[i][j] = acc
    for i in range(q):
        row_sum = 0.0
        for j in allowed[i]:
            row_sum += probs[i * k + j] * h_mat[i][j]
        for j in allowed[i]:
            d_mat[i][j] = probs[i * k + j] * (h_mat[i][j] - row_sum)
    dq = [0.0] * (q * d)
    dk = [0.0] * (k * d)
    dv = [0.0] * (k * d)
    for i in range(q):
        for r in range(d):
            acc = 0.0
            for j in allowed[i]:
                acc += d_mat[i][j] * ks[j * d + r]
            dq[i * d + r] = acc / scale
    for j in range(k):
        for r in range(d):
            acc_k = 0.0
            acc_v = 0.0
            for i in range(q):
                if j in allowed[i]:
                    acc_k += d_mat[i][j] * qs[i * d + r]
                    acc_v += probs[i * k + j] * g[i * d + r]
            dk[j * d + r] = acc_k / scale
            dv[j * d + r] = acc_v
    return dq, dk, dv


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


Q = [0.3, -0.5, 1.2, 0.1]
K = [0.7, 0.2, -0.4, 0.9, 0.1, -0.8]
V = [1.0, -1.0, 0.5, 2.0, -1.5, 0.25]
MASK = [True, True, False, True, False, True]
W4 = [0.7, -1.1, 0.4, 0.2]

CASES = [
    (Q, K, V, 2, 3, 2, MASK),
    (Q, K, V, 2, 3, 2, None),
    ([0.3, -0.5], [0.7, 0.2, -0.4, 0.9], [1.0, -1.0, 0.5, 2.0],
     1, 2, 2, [True, False]),
    ([0.3, -0.5, 1.2, 0.1, -0.6, 0.8],
     [0.7, 0.2, -0.4, 0.9],
     [1.0, -1.0, 0.5, 2.0], 3, 2, 2,
     [True, False, True, True, False, True]),
    ([0.25], [0.5], [-0.75], 1, 1, 1, [True]),
    ([0.25], [0.5], [-0.75], 1, 1, 1, None),
]


class ForwardValueTest(unittest.TestCase):

    def test_hand_computed_cases(self):
        for qs, ks, vs, q, k, d, mask in CASES:
            with self.subTest(q=q, k=k, d=d, mask=mask):
                out = Tensor(qs).attention(
                    Tensor(ks), Tensor(vs), q, k, d, mask
                )
                expected, _ = ref_attention(qs, ks, vs, q, k, d, mask)
                self.assertEqual(out.data, expected)
                self.assertEqual(len(out.data), q * d)

    def test_masked_probabilities_and_output_are_zero(self):
        out, probs = None, None
        result = Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, MASK)
        _, expected_p = ref_attention(Q, K, V, 2, 3, 2, MASK)
        # Key 2 is masked for query 0 and key 1 is masked for query 1.
        self.assertEqual(result.data, ref_attention(Q, K, V, 2, 3, 2, MASK)[0])
        self.assertEqual(expected_p[2], 0.0)
        self.assertEqual(expected_p[4], 0.0)

    def test_single_allowed_key_copies_its_value(self):
        mask = [False, True, False, False, True, False]
        out = Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, mask)
        # Each query attends only to key 1, so every row equals value row 1.
        self.assertEqual(out.data, [0.5, 2.0, 0.5, 2.0])

    def test_gradcheck_queries(self):
        for qs, ks, vs, q, k, d, mask in CASES:
            with self.subTest(q=q, k=k, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * d)]
                passed, err = gradcheck(
                    lambda t, ks=ks, vs=vs, q=q, k=k, d=d, mask=mask,
                    w=weights:
                        t.attention(Tensor(ks), Tensor(vs), q, k, d, mask)
                        .dot(Tensor(w)),
                    qs,
                )
                self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_keys(self):
        for qs, ks, vs, q, k, d, mask in CASES:
            with self.subTest(q=q, k=k, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * d)]
                passed, err = gradcheck(
                    lambda t, qs=qs, vs=vs, q=q, k=k, d=d, mask=mask,
                    w=weights:
                        Tensor(qs).attention(t, Tensor(vs), q, k, d, mask)
                        .dot(Tensor(w)),
                    ks,
                )
                self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_values(self):
        for qs, ks, vs, q, k, d, mask in CASES:
            with self.subTest(q=q, k=k, d=d, mask=mask):
                weights = [(-1.0) ** i * 0.375 for i in range(q * d)]
                passed, err = gradcheck(
                    lambda t, qs=qs, ks=ks, q=q, k=k, d=d, mask=mask,
                    w=weights:
                        Tensor(qs).attention(Tensor(ks), t, q, k, d, mask)
                        .dot(Tensor(w)),
                    vs,
                )
                self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_matches_reference(self):
        g = [0.5, -1.0, 2.0, 0.25]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.attention(key, value, 2, 3, 2, MASK).backward(g)
        _, probs = ref_attention(Q, K, V, 2, 3, 2, MASK)
        dq, dk, dv = ref_attention_grad(Q, K, V, 2, 3, 2, probs, g, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_backward_no_mask_matches_reference(self):
        g = [0.5, -1.0, 2.0, 0.25]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.attention(key, value, 2, 3, 2).backward(g)
        _, probs = ref_attention(Q, K, V, 2, 3, 2)
        dq, dk, dv = ref_attention_grad(Q, K, V, 2, 3, 2, probs, g)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_fully_masked_key_gets_zero_key_value_grad(self):
        # Key 2 is masked for both queries.
        mask = [True, True, False, False, True, False]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        q.attention(key, value, 2, 3, 2, mask).backward(
            [0.5, -1.0, 2.0, 0.25]
        )
        self.assertEqual(key.grad[4:6], [0.0, 0.0])
        self.assertEqual(value.grad[4:6], [0.0, 0.0])

    def test_only_parents_requiring_grad_receive_grads(self):
        q = Tensor(Q, True)
        key = Tensor(K, False)
        value = Tensor(V, True)
        out = q.attention(key, value, 2, 3, 2, MASK)
        out.backward([0.5, -1.0, 2.0, 0.25])
        self.assertIsNotNone(q.grad)
        self.assertIsNone(key.grad)
        self.assertIsNotNone(value.grad)

    def test_no_grad_no_graph(self):
        out = Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, MASK)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0])

    def test_snapshot_survives_mutation_and_replacement(self):
        g = [0.5, -1.0, 2.0, 0.25]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        mask = list(MASK)
        out = q.attention(key, value, 2, 3, 2, mask)
        q.data[0] = 999.0
        q.data = [9.0] * 4
        key.data = [8.0] * 6
        value.data = [7.0] * 6
        mask[0] = False
        mask.append(True)
        out.backward(g)
        expected_y, probs = ref_attention(Q, K, V, 2, 3, 2, MASK)
        self.assertEqual(out.data, expected_y)
        dq, dk, dv = ref_attention_grad(Q, K, V, 2, 3, 2, probs, g, MASK)
        self.assertEqual(q.grad, dq)
        self.assertEqual(key.grad, dk)
        self.assertEqual(value.grad, dv)

    def test_shared_key_and_value_merges_roles(self):
        # The same tensor plays both roles: its grad must equal dK + dV.
        w = [0.7, 0.2, -0.4, 0.9, 0.1, -0.8]
        weights = [0.7, -1.1, 0.4, 0.2]
        passed, err = gradcheck(
            lambda t, qs=Q, weights=weights:
                Tensor(qs).attention(t, t, 2, 3, 2, MASK).dot(Tensor(weights)),
            w,
        )
        self.assertTrue(passed, f"max abs error {err}")

        q = Tensor(Q, True)
        shared = Tensor(w, True)
        g = [0.5, -1.0, 2.0, 0.25]
        q.attention(shared, shared, 2, 3, 2, MASK).backward(g)
        _, probs = ref_attention(Q, w, w, 2, 3, 2, MASK)
        dq, dk, dv = ref_attention_grad(Q, w, w, 2, 3, 2, probs, g, MASK)
        self.assertEqual(shared.grad, [a + b for a, b in zip(dk, dv)])

    def test_all_three_parents_identical_merges_roles(self):
        a = [0.3, -0.5, 1.2, 0.1]
        weights = [0.7, -1.1, 0.4, 0.2]
        passed, err = gradcheck(
            lambda t, weights=weights:
                t.attention(t, t, 2, 2, 2).dot(Tensor(weights)),
            a,
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_repeated_backward_accumulates(self):
        g = [0.5, -1.0, 2.0, 0.25]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.attention(key, value, 2, 3, 2, MASK)
        out.backward(g)
        out.backward(g)
        _, probs = ref_attention(Q, K, V, 2, 3, 2, MASK)
        dq, dk, dv = ref_attention_grad(Q, K, V, 2, 3, 2, probs, g, MASK)
        self.assertEqual(q.grad, [2.0 * x for x in dq])
        self.assertEqual(key.grad, [2.0 * x for x in dk])
        self.assertEqual(value.grad, [2.0 * x for x in dv])

    def test_failed_backward_leaves_existing_grads_unchanged(self):
        g = [0.5, -1.0, 2.0, 0.25]
        q = Tensor(Q, True)
        key = Tensor(K, True)
        value = Tensor(V, True)
        out = q.attention(key, value, 2, 3, 2, MASK)
        out.backward(g)
        saved_q, saved_k, saved_v = list(q.grad), list(key.grad), list(value.grad)
        for bad in (None, 1.0, [1.0], [1.0] * 5, [1.0, float("inf"), 1.0, 1.0]):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    out.backward(bad)
                self.assertEqual(q.grad, saved_q)
                self.assertEqual(key.grad, saved_k)
                self.assertEqual(value.grad, saved_v)
        # A subsequent valid backward still accumulates on the first one.
        out.backward(g)
        _, probs = ref_attention(Q, K, V, 2, 3, 2, MASK)
        dq, dk, dv = ref_attention_grad(Q, K, V, 2, 3, 2, probs, g, MASK)
        self.assertEqual(q.grad, [2.0 * x for x in dq])
        self.assertEqual(key.grad, [2.0 * x for x in dk])
        self.assertEqual(value.grad, [2.0 * x for x in dv])


class ErrorContractTest(unittest.TestCase):

    def test_key_value_not_tensor(self):
        for bad in (None, "x", [0.5], 1, (K,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(bad, Tensor(V), 2, 3, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), bad, 2, 3, 2)

    def test_dims_type(self):
        for bad in (None, 1.0, 0.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), bad, 3, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, bad, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, bad)

    def test_dims_bool_is_type_error(self):
        with self.assertRaises(TypeError):
            Tensor(Q).attention(Tensor(K), Tensor(V), True, 3, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).attention(Tensor(K), Tensor(V), 2, False, 2)
        with self.assertRaises(TypeError):
            Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, True)

    def test_dims_non_positive(self):
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), bad, 3, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, bad, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, bad)

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).attention(Tensor(K), Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(make_bad(data=0.5), Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(Tensor(K), make_bad(data=0.5), 2, 3, 2)

    def test_data_empty_list(self):
        empty = make_bad(data=[])
        with self.assertRaises(ValueError):
            empty.attention(Tensor(K), Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(empty, Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(Tensor(K), empty, 2, 3, 2)

    def test_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.1] * 5).attention(Tensor(K), Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(Tensor([0.1] * 5), Tensor(V), 2, 3, 2)
        with self.assertRaises(ValueError):
            Tensor(Q).attention(Tensor(K), Tensor([0.1] * 7), 2, 3, 2)

    def test_element_type(self):
        for bad in ([0.1, 1, 0.3, 0.4, 0.5, 0.6],
                    [0.1, True, 0.3, 0.4, 0.5, 0.6],
                    [0.1, "x", 0.3, 0.4, 0.5, 0.6],
                    [0.1, None, 0.3, 0.4, 0.5, 0.6]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(make_bad(data=bad), Tensor(V), 2, 3, 2)
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), make_bad(data=bad), 2, 3, 2)

    def test_requires_grad_type(self):
        for role in ("q", "k", "v"):
            with self.subTest(role=role):
                qt = make_bad(requires_grad=1) if role == "q" else Tensor(Q)
                kt = make_bad(requires_grad=1) if role == "k" else Tensor(K)
                vt = make_bad(requires_grad=1) if role == "v" else Tensor(V)
                with self.assertRaises(TypeError):
                    qt.attention(kt, vt, 2, 3, 2)

    def test_non_finite_elements(self):
        for bad in ([0.1, float("inf")] + [0.3] * 4,
                    [float("nan"), 0.2] + [0.3] * 4,
                    [0.1, -float("inf")] + [0.3] * 4):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(make_bad(data=bad), Tensor(V), 2, 3, 2)
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), make_bad(data=bad), 2, 3, 2)

    def test_dims_checked_before_data(self):
        # A non-positive dimension is a ValueError even with malformed data.
        bad = make_bad(data=[0.1, 1])
        with self.assertRaises(ValueError):
            bad.attention(Tensor(K), Tensor(V), 0, 3, 2)

    def test_mask_type(self):
        for bad in (True, False, 1, 1.0, "tt", (True,) * 6, {0: True}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_mask_element_type(self):
        for bad in ([True, 1, True, True, True, True],
                    [True, True, 0.0, True, True, True],
                    [True, None, True, True, True, True],
                    [True, "t", True, True, True, True]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_mask_length(self):
        for bad in ([], [True] * 5, [True] * 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_mask_row_without_true(self):
        for bad in ([False, False, False, True, False, True],
                    [True, True, False, False, False, False],
                    [False] * 6):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor(Q).attention(Tensor(K), Tensor(V), 2, 3, 2, bad)

    def test_non_finite_forward_raises_value_error(self):
        big = [1.7e300] * 4
        with self.assertRaises(ValueError):
            big_t = Tensor(big)
            big_t.attention(Tensor([1.7e300] * 6), Tensor(V), 2, 3, 2)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.q = Tensor(Q, True)
        self.out = self.q.attention(
            Tensor(K, True), Tensor(V, True), 2, 3, 2, MASK
        )

    def test_upstream_omitted(self):
        with self.assertRaises(ValueError):
            self.out.backward()

    def test_upstream_scalar_for_vector_out(self):
        with self.assertRaises(ValueError):
            self.out.backward(1.0)

    def test_upstream_type_errors(self):
        for bad in (None, True, 1, "x", (1.0,) * 4):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_length(self):
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * 3)
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * 5)

    def test_upstream_element_type(self):
        for bad in ([1.0, 1, 1.0, 1.0], [1.0, True, 1.0, 1.0],
                    [1.0, "x", 1.0, 1.0], [1.0, None, 1.0, 1.0]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_non_finite(self):
        with self.assertRaises(ValueError):
            self.out.backward([float("inf"), 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            self.out.backward([1.0, float("nan"), 1.0, 1.0])

    def test_failed_upstream_leaves_no_grad(self):
        for bad in (None, True, 1, 1.0, [1.0],
                    [1.0, float("inf"), 1.0, 1.0]):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.out.backward(bad)
                self.assertIsNone(self.q.grad)


if __name__ == "__main__":
    unittest.main()
