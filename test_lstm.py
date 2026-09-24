"""Verifiable contract tests for Tensor.lstm.

A single-layer LSTM over flattened row-major float lists: input sequence
Tn*I (self), input weights 4H*I, hidden weights 4H*H, bias 4H, initial
hidden/cell states H. Gate order is i, f, g, o. Returns the Tn*H hidden
state sequence, row-major. Standard library only; discovered by
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def sigmoid(value):
    if value >= 0.0:
        e = math.exp(-value)
        return 1.0 / (1.0 + e)
    e = math.exp(value)
    return e / (1.0 + e)


def ref_forward(x, w, u, b, h0, c0, tn, isize, hsize):
    """Independent hand computation in the prescribed operation order.

    For each t and gate row r (i, f, g, o ascending), the affine a starts
    from the bias and accumulates input then hidden terms in ascending
    index order. i/f/o = sigmoid(a), g = tanh(a), c = f*c_prev + i*g,
    h = o*tanh(c). Returns (y, cache) where cache holds the per-step gate
    activations and states needed by the reference backward.

    Every accumulation is an explicit ascending loop starting from 0.0.
    """
    hs = []
    cs = []
    ivs = []
    fvs = []
    gvs = []
    ovs = []
    cbars = []
    h_prev = list(h0)
    c_prev = list(c0)
    for t in range(tn):
        gates = [0.0] * (4 * hsize)
        for r in range(4 * hsize):
            acc = b[r]
            for n in range(isize):
                acc += w[r * isize + n] * x[t * isize + n]
            for n in range(hsize):
                acc += u[r * hsize + n] * h_prev[n]
            gates[r] = acc
        iv = [sigmoid(gates[0 * hsize + m]) for m in range(hsize)]
        fv = [sigmoid(gates[1 * hsize + m]) for m in range(hsize)]
        gv = [math.tanh(gates[2 * hsize + m]) for m in range(hsize)]
        ov = [sigmoid(gates[3 * hsize + m]) for m in range(hsize)]
        c_row = [0.0] * hsize
        cbar = [0.0] * hsize
        h_row = [0.0] * hsize
        for m in range(hsize):
            c_row[m] = fv[m] * c_prev[m] + iv[m] * gv[m]
            cbar[m] = math.tanh(c_row[m])
            h_row[m] = ov[m] * cbar[m]
        ivs.append(iv)
        fvs.append(fv)
        gvs.append(gv)
        ovs.append(ov)
        cbars.append(cbar)
        cs.append(c_row)
        hs.append(h_row)
        h_prev = h_row
        c_prev = c_row
    y = []
    for t in range(tn):
        y += hs[t]
    return y, (hs, cs, ivs, fvs, gvs, ovs, cbars)


def ref_grad(x, w, u, b, h0, c0, tn, isize, hsize, grad):
    """Reference reverse-time chain rule, matching the prescribed order.

    Each buffer accumulates from 0.0 in ascending index order. Returns
    (dx, dw, du, db, dh0, dc0).
    """
    _, cache = ref_forward(x, w, u, b, h0, c0, tn, isize, hsize)
    hs, cs, ivs, fvs, gvs, ovs, cbars = cache
    dw = [0.0] * (4 * hsize * isize)
    du = [0.0] * (4 * hsize * hsize)
    db = [0.0] * (4 * hsize)
    dx = [0.0] * (tn * isize)
    dh0 = [0.0] * hsize
    dc0 = [0.0] * hsize
    dh_next = [0.0] * hsize
    dc_next = [0.0] * hsize
    for t in range(tn - 1, -1, -1):
        h_previous = h0 if t == 0 else hs[t - 1]
        c_previous = c0 if t == 0 else cs[t - 1]
        iv = ivs[t]
        fv = fvs[t]
        gv = gvs[t]
        ov = ovs[t]
        cbar = cbars[t]
        da = [0.0] * (4 * hsize)
        dc_row = [0.0] * hsize
        for m in range(hsize):
            dh_total = grad[t * hsize + m] + dh_next[m]
            do_v = dh_total * cbar[m]
            d_cb = dh_total * ov[m]
            dc = dc_next[m] + d_cb * (1.0 - cbar[m] * cbar[m])
            dc_row[m] = dc
            di = dc * gv[m]
            dg = dc * iv[m]
            df = dc * c_previous[m]
            da[0 * hsize + m] = (di * iv[m]) * (1.0 - iv[m])
            da[1 * hsize + m] = (df * fv[m]) * (1.0 - fv[m])
            da[2 * hsize + m] = dg * (1.0 - gv[m] * gv[m])
            da[3 * hsize + m] = (do_v * ov[m]) * (1.0 - ov[m])
        for r in range(4 * hsize):
            db[r] += da[r]
            for n in range(isize):
                dw[r * isize + n] += da[r] * x[t * isize + n]
            for n in range(hsize):
                du[r * hsize + n] += da[r] * h_previous[n]
        for n in range(isize):
            acc = 0.0
            for r in range(4 * hsize):
                acc += da[r] * w[r * isize + n]
            dx[t * isize + n] = acc
        dh_previous = [0.0] * hsize
        for n in range(hsize):
            acc = 0.0
            for r in range(4 * hsize):
                acc += da[r] * u[r * hsize + n]
            dh_previous[n] = acc
        if t == 0:
            for m in range(hsize):
                dc0[m] = dc_row[m] * fv[m]
            dh0 = dh_previous
        else:
            for m in range(hsize):
                dc_next[m] = dc_row[m] * fv[m]
            dh_next = dh_previous
    return dx, dw, du, db, dh0, dc0


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


def values(count, seed):
    out = []
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((state / 0x7FFFFFFF) * 1.4 - 0.7)
    return out


SHAPES = [(1, 1, 1), (2, 3, 2), (3, 2, 2), (1, 5, 3), (4, 1, 2)]

_DEFAULT = object()
"""Sentinel distinguishing 'use the valid default' from an explicit None."""


def make_case(tn, isize, hsize, grad=False):
    x = values(tn * isize, 1)
    w = values(4 * hsize * isize, 2)
    u = values(4 * hsize * hsize, 3)
    b = values(4 * hsize, 4)
    h0 = values(hsize, 5)
    c0 = values(hsize, 6)
    tensors = [
        Tensor(x, grad), Tensor(w, grad), Tensor(u, grad),
        Tensor(b, grad), Tensor(h0, grad), Tensor(c0, grad),
    ]
    return (x, w, u, b, h0, c0), tensors


class ForwardValueTest(unittest.TestCase):

    def test_hand_computed_cases(self):
        for tn, isize, hsize in SHAPES:
            with self.subTest(tn=tn, isize=isize, hsize=hsize):
                raw, _ = make_case(tn, isize, hsize)
                x, w, u, b, h0, c0 = raw
                out = Tensor(x).lstm(
                    Tensor(w), Tensor(u), Tensor(b),
                    Tensor(h0), Tensor(c0), tn, isize, hsize,
                )
                expected, _ = ref_forward(x, w, u, b, h0, c0, tn, isize, hsize)
                self.assertEqual(out.data, expected)
                self.assertEqual(len(out.data), tn * hsize)

    def test_single_unit_step(self):
        x = [0.4]
        w = [0.1, -0.2, 0.3, -0.4]
        u = [0.5, -0.6, 0.7, -0.8]
        b = [0.0, 0.1, -0.1, 0.2]
        h0 = [0.2]
        c0 = [-0.3]
        out = Tensor(x).lstm(
            Tensor(w), Tensor(u), Tensor(b), Tensor(h0), Tensor(c0),
            1, 1, 1,
        )
        expected, _ = ref_forward(x, w, u, b, h0, c0, 1, 1, 1)
        self.assertEqual(out.data, expected)


class GradCheckTest(unittest.TestCase):

    def test_gradcheck_all_parents(self):
        for tn, isize, hsize in SHAPES:
            raw, _ = make_case(tn, isize, hsize)
            x, w, u, b, h0, c0 = raw
            weights = [(-1.0) ** i * 0.31 for i in range(tn * hsize)]
            roles = {
                "x": lambda t: t.lstm(
                    Tensor(w), Tensor(u), Tensor(b),
                    Tensor(h0), Tensor(c0), tn, isize, hsize),
                "w": lambda t: Tensor(x).lstm(
                    t, Tensor(u), Tensor(b),
                    Tensor(h0), Tensor(c0), tn, isize, hsize),
                "u": lambda t: Tensor(x).lstm(
                    Tensor(w), t, Tensor(b),
                    Tensor(h0), Tensor(c0), tn, isize, hsize),
                "b": lambda t: Tensor(x).lstm(
                    Tensor(w), Tensor(u), t,
                    Tensor(h0), Tensor(c0), tn, isize, hsize),
                "h0": lambda t: Tensor(x).lstm(
                    Tensor(w), Tensor(u), Tensor(b),
                    t, Tensor(c0), tn, isize, hsize),
                "c0": lambda t: Tensor(x).lstm(
                    Tensor(w), Tensor(u), Tensor(b),
                    Tensor(h0), t, tn, isize, hsize),
            }
            for name, fn in roles.items():
                with self.subTest(shape=(tn, isize, hsize), role=name):
                    data = {"x": x, "w": w, "u": u,
                            "b": b, "h0": h0, "c0": c0}[name]
                    passed, err = gradcheck(
                        lambda t, fn=fn: fn(t).dot(Tensor(weights)), data
                    )
                    self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_matches_reference(self):
        for tn, isize, hsize in SHAPES:
            with self.subTest(shape=(tn, isize, hsize)):
                raw, tensors = make_case(tn, isize, hsize, grad=True)
                x, w, u, b, h0, c0 = raw
                xt, wt, ut, bt, h0t, c0t = tensors
                grad = [(-1.0) ** m * 0.5 for m in range(tn * hsize)]
                xt.lstm(
                    wt, ut, bt, h0t, c0t, tn, isize, hsize
                ).backward(grad)
                dx, dw, du, db, dh0, dc0 = ref_grad(
                    x, w, u, b, h0, c0, tn, isize, hsize, grad
                )
                self.assertEqual(xt.grad, dx)
                self.assertEqual(wt.grad, dw)
                self.assertEqual(ut.grad, du)
                self.assertEqual(bt.grad, db)
                self.assertEqual(h0t.grad, dh0)
                self.assertEqual(c0t.grad, dc0)

    def test_only_parents_requiring_grad_receive_grads(self):
        tn, isize, hsize = 2, 3, 2
        raw, _ = make_case(tn, isize, hsize)
        x, w, u, b, h0, c0 = raw
        xt = Tensor(x, True)
        wt = Tensor(w, False)
        ut = Tensor(u, True)
        bt = Tensor(b, False)
        h0t = Tensor(h0, False)
        c0t = Tensor(c0, False)
        out = xt.lstm(wt, ut, bt, h0t, c0t, tn, isize, hsize)
        out.backward([0.5, -1.0, 0.25, -0.75])
        self.assertIsNotNone(xt.grad)
        self.assertIsNotNone(ut.grad)
        self.assertIsNone(wt.grad)
        self.assertIsNone(bt.grad)
        self.assertIsNone(h0t.grad)
        self.assertIsNone(c0t.grad)

    def test_no_grad_no_graph(self):
        tn, isize, hsize = 2, 3, 2
        raw, _ = make_case(tn, isize, hsize)
        x, w, u, b, h0, c0 = raw
        out = Tensor(x).lstm(
            Tensor(w), Tensor(u), Tensor(b), Tensor(h0), Tensor(c0),
            tn, isize, hsize,
        )
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward([1.0] * (tn * hsize))

    def test_snapshot_survives_mutation_and_replacement(self):
        tn, isize, hsize = 2, 3, 2
        raw, tensors = make_case(tn, isize, hsize, grad=True)
        x, w, u, b, h0, c0 = raw
        xt, wt, ut, bt, h0t, c0t = tensors
        out = xt.lstm(wt, ut, bt, h0t, c0t, tn, isize, hsize)
        expected, _ = ref_forward(x, w, u, b, h0, c0, tn, isize, hsize)
        xt.data[0] = 999.0
        xt.data = [9.0] * len(x)
        wt.data = [8.0] * len(w)
        ut.data = [7.0] * len(u)
        bt.data = [6.0] * len(b)
        h0t.data = [5.0] * hsize
        c0t.data = [4.0] * hsize
        grad = [0.5, -1.0, 0.25, -0.75]
        out.backward(grad)
        self.assertEqual(out.data, expected)
        dx, dw, du, db, dh0, dc0 = ref_grad(
            x, w, u, b, h0, c0, tn, isize, hsize, grad
        )
        self.assertEqual(xt.grad, dx)
        self.assertEqual(wt.grad, dw)
        self.assertEqual(ut.grad, du)
        self.assertEqual(bt.grad, db)
        self.assertEqual(h0t.grad, dh0)
        self.assertEqual(c0t.grad, dc0)

    def test_shared_initial_state_merges_roles(self):
        tn, isize, hsize = 2, 3, 2
        raw, _ = make_case(tn, isize, hsize)
        x, w, u, b, _, _ = raw
        hc = values(hsize, 9)
        weights = [(-1.0) ** i * 0.31 for i in range(tn * hsize)]
        passed, err = gradcheck(
            lambda t: Tensor(x).lstm(
                Tensor(w), Tensor(u), Tensor(b), t, t,
                tn, isize, hsize,
            ).dot(Tensor(weights)),
            hc,
        )
        self.assertTrue(passed, f"max abs error {err}")

        shared = Tensor(hc, True)
        Tensor(x).lstm(
            Tensor(w), Tensor(u), Tensor(b), shared, shared,
            tn, isize, hsize,
        ).backward(weights)
        h0t = Tensor(list(hc), True)
        c0t = Tensor(list(hc), True)
        Tensor(x).lstm(
            Tensor(w), Tensor(u), Tensor(b), h0t, c0t,
            tn, isize, hsize,
        ).backward(weights)
        self.assertEqual(
            shared.grad,
            [a + z for a, z in zip(h0t.grad, c0t.grad)],
        )

    def test_shared_weights_merges_roles(self):
        tn, isize, hsize = 2, 2, 2
        wu = values(4 * hsize * isize, 2)
        x = values(tn * isize, 1)
        b = values(4 * hsize, 9)
        hc = values(hsize, 5)
        weights = [0.31 * (-1.0) ** i for i in range(tn * hsize)]
        passed, err = gradcheck(
            lambda t: Tensor(x).lstm(
                t, t, Tensor(b), Tensor(hc), Tensor(hc),
                tn, isize, hsize,
            ).dot(Tensor(weights)),
            wu,
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_repeated_backward_accumulates(self):
        tn, isize, hsize = 2, 3, 2
        raw, tensors = make_case(tn, isize, hsize, grad=True)
        x, w, u, b, h0, c0 = raw
        xt, wt, ut, bt, h0t, c0t = tensors
        out = xt.lstm(wt, ut, bt, h0t, c0t, tn, isize, hsize)
        grad = [0.5, -1.0, 0.25, -0.75]
        out.backward(grad)
        out.backward(grad)
        dx, dw, du, db, dh0, dc0 = ref_grad(
            x, w, u, b, h0, c0, tn, isize, hsize, grad
        )
        self.assertEqual(xt.grad, [2.0 * z for z in dx])
        self.assertEqual(wt.grad, [2.0 * z for z in dw])
        self.assertEqual(ut.grad, [2.0 * z for z in du])
        self.assertEqual(bt.grad, [2.0 * z for z in db])
        self.assertEqual(h0t.grad, [2.0 * z for z in dh0])
        self.assertEqual(c0t.grad, [2.0 * z for z in dc0])

    def test_failed_backward_leaves_existing_grads_unchanged(self):
        tn, isize, hsize = 2, 3, 2
        raw, tensors = make_case(tn, isize, hsize, grad=True)
        xt, wt, ut, bt, h0t, c0t = tensors
        out = xt.lstm(wt, ut, bt, h0t, c0t, tn, isize, hsize)
        grad = [0.5, -1.0, 0.25, -0.75]
        out.backward(grad)
        saved = [list(t.grad) for t in tensors]
        n = tn * hsize
        for bad in (
            None, 1.0, [1.0], [1.0] * (n + 1), [1.0] * (n - 1),
            [1.0, float("inf")] + [1.0] * (n - 2),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    out.backward(bad)
                for tensor, old in zip(tensors, saved):
                    self.assertEqual(tensor.grad, old)


class ErrorContractTest(unittest.TestCase):

    def setUp(self):
        self.tn, self.isize, self.hsize = 2, 3, 2
        raw, _ = make_case(self.tn, self.isize, self.hsize)
        self.x, self.w, self.u, self.b, self.h0, self.c0 = raw

    def _call(self, w=_DEFAULT, u=_DEFAULT, b=_DEFAULT, h0=_DEFAULT,
              c0=_DEFAULT, steps=_DEFAULT, input_size=_DEFAULT,
              hidden_size=_DEFAULT, x=_DEFAULT):
        receiver = Tensor(self.x) if x is _DEFAULT else x
        return receiver.lstm(
            Tensor(self.w) if w is _DEFAULT else w,
            Tensor(self.u) if u is _DEFAULT else u,
            Tensor(self.b) if b is _DEFAULT else b,
            Tensor(self.h0) if h0 is _DEFAULT else h0,
            Tensor(self.c0) if c0 is _DEFAULT else c0,
            self.tn if steps is _DEFAULT else steps,
            self.isize if input_size is _DEFAULT else input_size,
            self.hsize if hidden_size is _DEFAULT else hidden_size,
        )

    def test_operands_not_tensor(self):
        for bad in (None, "x", [0.5], 1, (self.b,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self._call(w=bad)
                with self.assertRaises(TypeError):
                    self._call(u=bad)
                with self.assertRaises(TypeError):
                    self._call(b=bad)
                with self.assertRaises(TypeError):
                    self._call(h0=bad)
                with self.assertRaises(TypeError):
                    self._call(c0=bad)

    def test_dims_type(self):
        for bad in (None, 1.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self._call(steps=bad)
                with self.assertRaises(TypeError):
                    self._call(input_size=bad)
                with self.assertRaises(TypeError):
                    self._call(hidden_size=bad)

    def test_dims_bool_is_type_error(self):
        with self.assertRaises(TypeError):
            self._call(steps=True)
        with self.assertRaises(TypeError):
            self._call(input_size=False)
        with self.assertRaises(TypeError):
            self._call(hidden_size=True)

    def test_dims_non_positive(self):
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._call(steps=bad)
                with self.assertRaises(ValueError):
                    self._call(input_size=bad)
                with self.assertRaises(ValueError):
                    self._call(hidden_size=bad)

    def test_data_not_list_or_empty(self):
        for role in ("x", "w", "u", "b", "h0", "c0"):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    self._call(**{role: make_bad(data=0.5)})
                with self.assertRaises(ValueError):
                    self._call(**{role: make_bad(data=[])})

    def test_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            self._call(x=Tensor([0.1] * (self.tn * self.isize + 1)))
        with self.assertRaises(ValueError):
            self._call(w=Tensor([0.1] * (4 * self.hsize * self.isize + 1)))
        with self.assertRaises(ValueError):
            self._call(u=Tensor([0.1] * (4 * self.hsize * self.hsize + 1)))
        with self.assertRaises(ValueError):
            self._call(b=Tensor([0.1] * (4 * self.hsize + 1)))
        with self.assertRaises(ValueError):
            self._call(h0=Tensor([0.1] * (self.hsize + 1)))
        with self.assertRaises(ValueError):
            self._call(c0=Tensor([0.1] * (self.hsize - 1)))

    def test_element_type(self):
        good = {
            "x": self.x, "w": self.w, "u": self.u,
            "b": self.b, "h0": self.h0, "c0": self.c0,
        }
        for bad in (1, True, "x", None):
            for role in good:
                with self.subTest(bad=bad, role=role):
                    corrupted = list(good[role])
                    corrupted[0] = bad
                    with self.assertRaises(TypeError):
                        self._call(**{role: make_bad(data=corrupted)})

    def test_requires_grad_type(self):
        good = {
            "x": self.x, "w": self.w, "u": self.u,
            "b": self.b, "h0": self.h0, "c0": self.c0,
        }
        for role in good:
            with self.subTest(role=role):
                tensor = Tensor(list(good[role]))
                tensor.requires_grad = 1
                with self.assertRaises(TypeError):
                    self._call(**{role: tensor})

    def test_non_finite_elements(self):
        good = {
            "x": self.x, "w": self.w, "u": self.u,
            "b": self.b, "h0": self.h0, "c0": self.c0,
        }
        for bad in (float("inf"), float("nan"), -float("inf")):
            for role in good:
                with self.subTest(bad=bad, role=role):
                    corrupted = list(good[role])
                    corrupted[0] = bad
                    with self.assertRaises(ValueError):
                        self._call(**{role: make_bad(data=corrupted)})

    def test_non_finite_forward_raises_value_error(self):
        big = [1.7e300] * (self.tn * self.isize)
        with self.assertRaises(ValueError):
            Tensor(big).lstm(
                Tensor([1.7e300] * (4 * self.hsize * self.isize)),
                Tensor(self.u), Tensor(self.b),
                Tensor(self.h0), Tensor(self.c0),
                self.tn, self.isize, self.hsize,
            )


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.tn, self.isize, self.hsize = 2, 3, 2
        raw, tensors = make_case(self.tn, self.isize, self.hsize, grad=True)
        self.tensors = tensors
        self.out = tensors[0].lstm(
            *tensors[1:], self.tn, self.isize, self.hsize
        )
        self.n = self.tn * self.hsize

    def test_upstream_omitted(self):
        with self.assertRaises(ValueError):
            self.out.backward()

    def test_upstream_scalar_for_vector_out(self):
        with self.assertRaises(ValueError):
            self.out.backward(1.0)

    def test_upstream_type_errors(self):
        for bad in (None, True, 1, "x", (1.0,) * self.n):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_length(self):
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * (self.n - 1))
        with self.assertRaises(ValueError):
            self.out.backward([1.0] * (self.n + 1))

    def test_upstream_element_type(self):
        for bad in (1, True, "x", None):
            with self.subTest(bad=bad):
                grad = [1.0] * self.n
                grad[0] = bad
                with self.assertRaises(TypeError):
                    self.out.backward(grad)

    def test_upstream_non_finite(self):
        grad = [1.0] * self.n
        grad[0] = float("inf")
        with self.assertRaises(ValueError):
            self.out.backward(grad)
        grad[0] = float("nan")
        with self.assertRaises(ValueError):
            self.out.backward(grad)

    def test_failed_upstream_leaves_no_grad(self):
        for bad in (None, True, 1, 1.0, [1.0],
                    [1.0, float("inf")] + [1.0] * (self.n - 2)):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.out.backward(bad)
                for tensor in self.tensors:
                    self.assertIsNone(tensor.grad)


if __name__ == "__main__":
    unittest.main()
