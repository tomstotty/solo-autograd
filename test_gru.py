"""Verifiable contract tests for Tensor.gru.

Single-layer GRU over flattened row-major float lists: input sequence
Tn x I (self), input weights 3H x I (w), hidden weights 3H x H (u), bias
3H (b), initial hidden state H (h0); gate rows ordered z, r, n. Standard
library only; discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def stable_sigmoid(x):
    """Stable sigmoid: 1/(1+exp(-x)) for x >= 0 else exp(x)/(1+exp(x))."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def make_data(n, offset=0):
    """Deterministic moderate floats in [-9/7, 9/7]."""
    return [(((i + offset) * 37 % 19) - 9) / 7.0 for i in range(n)]


def ref_gru(xs, ws, us, bs, h0, Tn, I, H):
    """Independent hand computation in the prescribed operation order.

    Per step t and gate row j (both ascending) the affine pre-activation
    starts from b[j] and accumulates the input contributions in ascending
    index order. The z/r rows then accumulate u*h_prev and take the
    stable sigmoid; the n row accumulates u*(r*h_prev) and takes tanh.
    h = (1 - z)*n + z*h_prev. Returns (out, saved) where out is the Tn*H
    hidden sequence in (t, h) order.
    """
    G = 3 * H
    out = []
    saved = []
    h_prev = list(h0)
    for t in range(Tn):
        a = [0.0] * G
        for j in range(G):
            acc = bs[j]
            for k in range(I):
                acc += ws[j * I + k] * xs[t * I + k]
            a[j] = acc
        for j in range(2 * H):
            acc = a[j]
            for k in range(H):
                acc += us[j * H + k] * h_prev[k]
            a[j] = acc
        gz = [stable_sigmoid(a[k]) for k in range(H)]
        gr = [stable_sigmoid(a[H + k]) for k in range(H)]
        rh = [gr[k] * h_prev[k] for k in range(H)]
        gn = [0.0] * H
        for k in range(H):
            acc = a[2 * H + k]
            for k2 in range(H):
                acc += us[(2 * H + k) * H + k2] * rh[k2]
            gn[k] = math.tanh(acc)
        h_new = [
            (1.0 - gz[k]) * gn[k] + gz[k] * h_prev[k] for k in range(H)
        ]
        saved.append((gz, gr, gn, rh, list(h_prev)))
        out.extend(h_new)
        h_prev = h_new
    return out, saved


def ref_gru_grad(xs, ws, us, Tn, I, H, saved, grad):
    """Chain-rule the recurrence in reverse step order.

    Every sum is accumulated from 0.0 in ascending index order. Returns
    (dx, dw, du, db, dh0).
    """
    G = 3 * H
    dx = [0.0] * (Tn * I)
    dw = [0.0] * (G * I)
    du = [0.0] * (G * H)
    db = [0.0] * G
    dh_next = [0.0] * H
    for t in range(Tn - 1, -1, -1):
        gz, gr, gn, rh, h_prev = saved[t]
        da = [0.0] * G
        dh_prev = [0.0] * H
        for k in range(H):
            dh = 0.0
            dh += grad[t * H + k]
            dh += dh_next[k]
            dn = dh * (1.0 - gz[k])
            dz = dh * (h_prev[k] - gn[k])
            dh_prev[k] += dh * gz[k]
            da[2 * H + k] = dn * (1.0 - gn[k] * gn[k])
            da[k] = (dz * gz[k]) * (1.0 - gz[k])
        drh = [0.0] * H
        for k in range(H):
            da_n = da[2 * H + k]
            for k2 in range(H):
                du[(2 * H + k) * H + k2] += da_n * rh[k2]
                drh[k2] += da_n * us[(2 * H + k) * H + k2]
        for k in range(H):
            dr = drh[k] * h_prev[k]
            dh_prev[k] += drh[k] * gr[k]
            da[H + k] = (dr * gr[k]) * (1.0 - gr[k])
        for j in range(G):
            db[j] += da[j]
            for k in range(I):
                dw[j * I + k] += da[j] * xs[t * I + k]
                dx[t * I + k] += da[j] * ws[j * I + k]
            if j < 2 * H:
                for k in range(H):
                    du[j * H + k] += da[j] * h_prev[k]
                    dh_prev[k] += da[j] * us[j * H + k]
        dh_next = dh_prev
    return dx, dw, du, db, dh_next


def make_operands(Tn, I, H, requires_grad=True):
    """Build the five gru operands with deterministic data."""
    x = Tensor(make_data(Tn * I, 0), requires_grad)
    w = Tensor(make_data(3 * H * I, 3), requires_grad)
    u = Tensor(make_data(3 * H * H, 5), requires_grad)
    b = Tensor(make_data(3 * H, 7), requires_grad)
    h0 = Tensor(make_data(H, 11), requires_grad)
    return x, w, u, b, h0


class GruForwardTest(unittest.TestCase):
    def test_forward_matches_reference(self):
        for Tn, I, H in ((1, 1, 1), (3, 2, 3), (4, 3, 2), (2, 5, 1)):
            x, w, u, b, h0 = make_operands(Tn, I, H)
            out = x.gru(w, u, b, h0, Tn, I, H)
            expected, _ = ref_gru(
                x.data, w.data, u.data, b.data, h0.data, Tn, I, H
            )
            self.assertEqual(out.data, expected)
            self.assertEqual(len(out.data), Tn * H)

    def test_output_is_t_then_h_order(self):
        # Tn=2, I=1, H=1 with zero weights: h_t = (1-z)*n + z*h_prev
        # computed from the bias alone, chained across steps.
        x = Tensor([0.0, 0.0])
        w = Tensor([0.0] * 3)
        u = Tensor([0.0] * 3)
        b = Tensor([0.5, 0.25, -0.5])
        h0 = Tensor([0.0])
        out = x.gru(w, u, b, h0, 2, 1, 1)
        z_g = stable_sigmoid(0.5)
        n_g = math.tanh(-0.5)
        h1 = (1.0 - z_g) * n_g + z_g * 0.0
        h2 = (1.0 - z_g) * n_g + z_g * h1
        self.assertEqual(out.data, [h1, h2])

    def test_no_graph_without_requires_grad(self):
        x, w, u, b, h0 = make_operands(2, 2, 2, requires_grad=False)
        out = x.gru(w, u, b, h0, 2, 2, 2)
        self.assertFalse(out.requires_grad)
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0])

    def test_forward_failure_changes_nothing(self):
        x, w, u, b, h0 = make_operands(2, 2, 2)
        w.data[0] = 1e308
        x.data[0] = 1e308
        before = [list(t.data) for t in (x, w, u, b, h0)]
        with self.assertRaises(ValueError):
            x.gru(w, u, b, h0, 2, 2, 2)
        for tensor, data in zip((x, w, u, b, h0), before):
            self.assertEqual(tensor.data, data)
            self.assertIsNone(tensor.grad)


class GruBackwardTest(unittest.TestCase):
    def test_backward_matches_reference(self):
        for Tn, I, H in ((1, 1, 1), (3, 2, 3), (4, 3, 2)):
            x, w, u, b, h0 = make_operands(Tn, I, H)
            out = x.gru(w, u, b, h0, Tn, I, H)
            grad = make_data(Tn * H, 17)
            out.backward(list(grad))
            _, saved = ref_gru(
                x.data, w.data, u.data, b.data, h0.data, Tn, I, H
            )
            dx, dw, du, db, dh0 = ref_gru_grad(
                x.data, w.data, u.data, Tn, I, H, saved, grad
            )
            self.assertEqual(x.grad, dx)
            self.assertEqual(w.grad, dw)
            self.assertEqual(u.grad, du)
            self.assertEqual(b.grad, db)
            self.assertEqual(h0.grad, dh0)

    def test_gradcheck_each_role(self):
        Tn, I, H = 3, 2, 2
        x, w, u, b, h0 = make_operands(Tn, I, H, requires_grad=False)
        roles = (
            lambda t: t.gru(w, u, b, h0, Tn, I, H).sum(),
            lambda t: x.gru(t, u, b, h0, Tn, I, H).sum(),
            lambda t: x.gru(w, t, b, h0, Tn, I, H).sum(),
            lambda t: x.gru(w, u, t, h0, Tn, I, H).sum(),
            lambda t: x.gru(w, u, b, t, Tn, I, H).sum(),
        )
        datas = (x.data, w.data, u.data, b.data, h0.data)
        for fn, data in zip(roles, datas):
            passed, err = gradcheck(fn, list(data))
            self.assertTrue(passed, f"max abs error {err}")

    def test_partial_requires_grad(self):
        Tn, I, H = 2, 2, 2
        x, w, u, b, h0 = make_operands(Tn, I, H, requires_grad=False)
        w.requires_grad = True
        h0.requires_grad = True
        out = x.gru(w, u, b, h0, Tn, I, H)
        self.assertTrue(out.requires_grad)
        out.backward([1.0] * (Tn * H))
        self.assertIsNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNone(u.grad)
        self.assertIsNone(b.grad)
        self.assertIsNotNone(h0.grad)

    def test_shared_tensor_roles_merge(self):
        # w and u are the same object (I == H so the lengths agree): its
        # grad is the sum of the two role contributions.
        Tn, I, H = 3, 2, 2
        x, _, _, b, h0 = make_operands(Tn, I, H)
        shared = Tensor(make_data(3 * H * I, 23), True)
        out = x.gru(shared, shared, b, h0, Tn, I, H)
        grad = make_data(Tn * H, 29)
        out.backward(list(grad))
        _, saved = ref_gru(
            x.data, shared.data, shared.data, b.data, h0.data, Tn, I, H
        )
        _, dw, du, _, _ = ref_gru_grad(
            x.data, shared.data, shared.data, Tn, I, H, saved, grad
        )
        self.assertEqual(
            shared.grad, [p + q for p, q in zip(dw, du)]
        )

    def test_repeated_backward_replaces_contribution(self):
        Tn, I, H = 2, 2, 2
        g1 = make_data(Tn * H, 31)
        g2 = make_data(Tn * H, 37)
        combined = [p + q for p, q in zip(g1, g2)]

        x, w, u, b, h0 = make_operands(Tn, I, H)
        out = x.gru(w, u, b, h0, Tn, I, H)
        out.backward(list(g1))
        out.backward(list(g2))

        x2, w2, u2, b2, h02 = make_operands(Tn, I, H)
        out2 = x2.gru(w2, u2, b2, h02, Tn, I, H)
        out2.backward(combined)

        for first, second in zip(
            (x, w, u, b, h0), (x2, w2, u2, b2, h02)
        ):
            self.assertEqual(first.grad, second.grad)

    def test_snapshot_ignores_later_mutation(self):
        Tn, I, H = 2, 2, 2
        x, w, u, b, h0 = make_operands(Tn, I, H)
        out = x.gru(w, u, b, h0, Tn, I, H)
        grad = make_data(Tn * H, 41)
        # Mutate every operand after the forward pass; backward must
        # still use the values captured at call time.
        for tensor in (x, w, u, b, h0):
            tensor.data = [99.0] * len(tensor.data)
        out.backward(list(grad))

        x2, w2, u2, b2, h02 = make_operands(Tn, I, H)
        out2 = x2.gru(w2, u2, b2, h02, Tn, I, H)
        out2.backward(list(grad))
        self.assertEqual(out.data, out2.data)
        for first, second in zip(
            (x, w, u, b, h0), (x2, w2, u2, b2, h02)
        ):
            self.assertEqual(first.grad, second.grad)

    def test_failed_backward_changes_no_grad(self):
        # Huge upstream grads overflow the reverse recurrence; the pass
        # must abort with every grad untouched.
        Tn, I, H = 2, 1, 1
        x = Tensor([0.1, 0.1], True)
        w = Tensor([0.1] * 3, True)
        u = Tensor([0.1] * 3, True)
        b = Tensor([0.5, 0.25, -0.5], True)
        h0 = Tensor([0.0], True)
        out = x.gru(w, u, b, h0, Tn, I, H)
        with self.assertRaises(ValueError):
            out.backward([1.5e308] * (Tn * H))
        for tensor in (x, w, u, b, h0):
            self.assertIsNone(tensor.grad)


class GruValidationTest(unittest.TestCase):
    def setUp(self):
        self.Tn, self.I, self.H = 2, 2, 2
        self.operands = make_operands(self.Tn, self.I, self.H)

    def gru(self, x, w, u, b, h0, **kw):
        args = dict(steps=self.Tn, input_size=self.I, hidden_size=self.H)
        args.update(kw)
        return x.gru(w, u, b, h0, **args)

    def test_non_tensor_operands(self):
        x, w, u, b, h0 = self.operands
        for bad in (1.0, [1.0], "w", None, True):
            with self.assertRaises(TypeError):
                self.gru(x, bad, u, b, h0)
            with self.assertRaises(TypeError):
                self.gru(x, w, bad, b, h0)
            with self.assertRaises(TypeError):
                self.gru(x, w, u, bad, h0)
            with self.assertRaises(TypeError):
                self.gru(x, w, u, b, bad)

    def test_data_must_be_nonempty_float_list(self):
        x, w, u, b, h0 = self.operands
        with self.assertRaises(ValueError):
            self.gru(Tensor(1.0), w, u, b, h0)
        with self.assertRaises(ValueError):
            self.gru(x, Tensor([]), u, b, h0)
        with self.assertRaises(TypeError):
            self.gru(x, w, Tensor([1, 2, 3] + [0.5] * 9), b, h0)
        with self.assertRaises(TypeError):
            self.gru(x, w, u, Tensor([True] * 6), h0)
        with self.assertRaises(ValueError):
            self.gru(x, w, u, b, Tensor([float("nan"), 0.5]))
        with self.assertRaises(ValueError):
            self.gru(x, Tensor([float("inf")] + [0.5] * 11), u, b, h0)

    def test_requires_grad_must_be_bool(self):
        x, w, u, b, h0 = self.operands
        u.requires_grad = 1
        with self.assertRaises(TypeError):
            self.gru(x, w, u, b, h0)

    def test_dimension_arguments(self):
        x, w, u, b, h0 = self.operands
        for bad in (True, 2.0, "2", None):
            with self.assertRaises(TypeError):
                self.gru(x, w, u, b, h0, steps=bad)
            with self.assertRaises(TypeError):
                self.gru(x, w, u, b, h0, input_size=bad)
            with self.assertRaises(TypeError):
                self.gru(x, w, u, b, h0, hidden_size=bad)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                self.gru(x, w, u, b, h0, steps=bad)
            with self.assertRaises(ValueError):
                self.gru(x, w, u, b, h0, input_size=bad)
            with self.assertRaises(ValueError):
                self.gru(x, w, u, b, h0, hidden_size=bad)

    def test_length_mismatch(self):
        x, w, u, b, h0 = self.operands
        with self.assertRaises(ValueError):
            self.gru(Tensor([0.5] * 5), w, u, b, h0)
        with self.assertRaises(ValueError):
            self.gru(x, Tensor([0.5] * 11), u, b, h0)
        with self.assertRaises(ValueError):
            self.gru(x, w, Tensor([0.5] * 11), b, h0)
        with self.assertRaises(ValueError):
            self.gru(x, w, u, Tensor([0.5] * 5), h0)
        with self.assertRaises(ValueError):
            self.gru(x, w, u, b, Tensor([0.5] * 3))

    def test_backward_grad_validation(self):
        x, w, u, b, h0 = self.operands
        out = self.gru(x, w, u, b, h0)
        with self.assertRaises(ValueError):
            out.backward()
        with self.assertRaises(ValueError):
            out.backward(1.0)
        with self.assertRaises(ValueError):
            out.backward([1.0])
        with self.assertRaises(TypeError):
            out.backward("grad")
        with self.assertRaises(TypeError):
            out.backward([1.0, 2, 3.0, 4.0])
        with self.assertRaises(TypeError):
            out.backward([1.0, True, 3.0, 4.0])
        with self.assertRaises(ValueError):
            out.backward([1.0, float("nan"), 3.0, 4.0])
        with self.assertRaises(ValueError):
            out.backward([1.0, float("inf"), 3.0, 4.0])
        # A failed backward leaves every grad untouched.
        for tensor in self.operands:
            self.assertIsNone(tensor.grad)


if __name__ == "__main__":
    unittest.main()
