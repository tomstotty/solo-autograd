"""Verifiable contract tests for Tensor.log_softmax_batch.

Batched stable log-softmax over a flattened rows x cols float list in
row-major order. Standard library only; discovered by
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def ref_log_softmax_batch(xs, rows, cols):
    """Independent hand computation in the prescribed operation order.

    For each row, in ascending column order: m = max of the row, s
    accumulated from 0.0 as s += exp(x_i - m), then l_i = x_i - m - log(s).
    """
    out = [0.0] * len(xs)
    for q in range(rows):
        base = q * cols
        m = max(xs[base:base + cols])
        s = 0.0
        for j in range(cols):
            s += math.exp(xs[base + j] - m)
        log_s = math.log(s)
        for j in range(cols):
            out[base + j] = xs[base + j] - m - log_s
    return out


def ref_log_softmax_batch_grad(l, rows, cols, g):
    """Per row: G = sum(g_i), then dx_i = g_i - exp(l_i) * G."""
    dx = [0.0] * len(l)
    for q in range(rows):
        base = q * cols
        G = 0.0
        for j in range(cols):
            G += g[base + j]
        for j in range(cols):
            k = base + j
            dx[k] = g[k] - math.exp(l[k]) * G
    return dx


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


CASES = [
    ([0.3, -1.5, 2.0, 1.0, 0.2, -0.4], 2, 3),
    ([2.0, 0.0, 1.0, 0.3, -1.5, 2.0], 2, 3),
    ([0.3, -1.5, 2.0], 1, 3),
    ([-0.0, 0.0, 5.5, -7.25], 2, 2),
    ([5.5, -7.25, 0.125, -0.375, 1.0, 2.0, 3.0, 4.0], 2, 4),
    ([1e-12, -1e-12], 1, 2),
    ([7.0], 1, 1),
    ([1000.0, -1000.0, 1000.0, -1000.0], 2, 2),
]


class ForwardValueTest(unittest.TestCase):

    def test_hand_computed_cases(self):
        for xs, rows, cols in CASES:
            with self.subTest(xs=xs, rows=rows, cols=cols):
                out = Tensor(xs).log_softmax_batch(rows, cols)
                self.assertEqual(
                    out.data, ref_log_softmax_batch(xs, rows, cols)
                )

    def test_matches_log_softmax_per_row(self):
        rows = [[2.0, 0.0, 1.0], [0.3, -1.5, 2.0]]
        xs = rows[0] + rows[1]
        batched = Tensor(xs).log_softmax_batch(2, 3).data
        plain0 = Tensor(rows[0]).log_softmax().data
        plain1 = Tensor(rows[1]).log_softmax().data
        self.assertEqual(batched, plain0 + plain1)

    def test_each_row_softmax_sums_to_one(self):
        for xs, rows, cols in CASES:
            with self.subTest(xs=xs, rows=rows, cols=cols):
                out = Tensor(xs).log_softmax_batch(rows, cols).data
                for q in range(rows):
                    row = out[q * cols:(q + 1) * cols]
                    self.assertAlmostEqual(
                        sum(math.exp(v) for v in row), 1.0, places=12
                    )

    def test_large_logits_stable(self):
        out = Tensor(
            [1000.0, -1000.0, 1000.0, -1000.0]
        ).log_softmax_batch(2, 2)
        self.assertEqual(out.data, [0.0, -2000.0, 0.0, -2000.0])

    def test_single_element_row_is_zero(self):
        out = Tensor([3.25, -1.5]).log_softmax_batch(2, 1)
        self.assertEqual(out.data, [0.0, 0.0])

    def test_output_is_vector(self):
        out = Tensor([0.3, -1.5, 1.0, 0.2]).log_softmax_batch(2, 2)
        self.assertIsInstance(out.data, list)
        self.assertEqual(len(out.data), 4)

    def test_gradcheck(self):
        for xs, rows, cols in CASES:
            with self.subTest(xs=xs, rows=rows, cols=cols):
                w = Tensor(
                    [0.7, -1.3, 2.1, 0.4, -0.9, 1.6, -0.5, 1.2][:len(xs)]
                )
                passed, err = gradcheck(
                    lambda t, r=rows, c=cols, w=w:
                        t.log_softmax_batch(r, c).dot(w),
                    xs,
                )
                self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_matches_reference(self):
        xs = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4]
        g = [0.5, 9.0, -1.5, 2.0, 0.25, 9.0]
        x = Tensor(xs, True)
        x.log_softmax_batch(2, 3).backward(g)
        l = ref_log_softmax_batch(xs, 2, 3)
        self.assertEqual(x.grad, ref_log_softmax_batch_grad(l, 2, 3, g))

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5, 0.1, 0.2]).log_softmax_batch(2, 2)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0])

    def test_snapshot_survives_mutation_and_replacement(self):
        xs = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4]
        g = [1.0, 9.0, 1.0, 1.0, 1.0, 9.0]
        x = Tensor(xs, True)
        out = x.log_softmax_batch(2, 3)
        x.data[0] = 999.0           # in-place mutation after forward
        x.data = [9.0] * 6          # wholesale replacement after forward
        out.backward(g)
        l = ref_log_softmax_batch(xs, 2, 3)
        self.assertEqual(out.data, l)
        self.assertEqual(x.grad, ref_log_softmax_batch_grad(l, 2, 3, g))

    def test_shared_path_accumulates(self):
        xs = [0.3, -1.5, 1.0, 0.2]
        g = [0.5, -1.0, 2.0, 0.25]
        x = Tensor(xs, True)
        out = x.log_softmax_batch(2, 2)
        out.add(out).backward(g)
        l = ref_log_softmax_batch(xs, 2, 2)
        self.assertEqual(
            x.grad, ref_log_softmax_batch_grad(l, 2, 2, [2 * v for v in g])
        )

    def test_repeated_backward_accumulates(self):
        xs = [0.3, -1.5, 1.0, 0.2]
        g = [0.5, -1.0, 2.0, 0.25]
        x = Tensor(xs, True)
        out = x.log_softmax_batch(2, 2)
        out.backward(g)
        out.backward(g)
        l = ref_log_softmax_batch(xs, 2, 2)
        self.assertEqual(
            x.grad, ref_log_softmax_batch_grad(l, 2, 2, [2 * v for v in g])
        )


class ErrorContractTest(unittest.TestCase):

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).log_softmax_batch(1, 1)

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).log_softmax_batch(1, 1)

    def test_data_not_list_checked_first(self):
        # The data precondition is checked before rows and cols.
        with self.assertRaises(ValueError):
            make_bad(data=0.5).log_softmax_batch("x", "x")

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None],
                    [0.3, [0.1]]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).log_softmax_batch(1, len(bad))

    def test_requires_grad_type(self):
        with self.assertRaises(TypeError):
            make_bad(requires_grad=1).log_softmax_batch(1, 1)
        with self.assertRaises(TypeError):
            make_bad(requires_grad="yes").log_softmax_batch(1, 1)

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan"), 0.1],
                    [-float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_bad(data=bad).log_softmax_batch(1, len(bad))

    def test_rows_cols_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (None, 1.0, 0.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.log_softmax_batch(bad, 2)
                with self.assertRaises(TypeError):
                    x.log_softmax_batch(1, bad)

    def test_rows_cols_bool_is_type_error(self):
        x = Tensor([0.3, 0.4])
        with self.assertRaises(TypeError):
            x.log_softmax_batch(True, 2)
        with self.assertRaises(TypeError):
            x.log_softmax_batch(1, False)

    def test_rows_cols_non_positive(self):
        x = Tensor([0.3, 0.4])
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.log_softmax_batch(bad, 2)
                with self.assertRaises(ValueError):
                    x.log_softmax_batch(1, bad)

    def test_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2, 0.3]).log_softmax_batch(2, 2)
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2, 0.3, 0.4, 0.5]).log_softmax_batch(2, 3)

    def test_rows_cols_checked_before_length(self):
        # A bad rows type is a TypeError and a non-positive cols is a
        # ValueError even when the data length cannot match either way.
        with self.assertRaises(TypeError):
            Tensor([0.1, 0.2]).log_softmax_batch(1.0, 2)
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2]).log_softmax_batch(1, 0)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5, 1.0, 0.2], True)
        self.out = self.x.log_softmax_batch(2, 2)

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
            self.out.backward([1.0])
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
            self.out.backward([1.0, float("inf"), 1.0, 1.0])
        with self.assertRaises(ValueError):
            self.out.backward([float("nan"), 1.0, 1.0, 1.0])

    def test_failed_upstream_leaves_no_grad(self):
        for bad in (None, True, 1, 1.0, [1.0], [1.0, float("inf")] * 2,
                    [1.0, 1.0, 1.0, float("nan")]):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.out.backward(bad)
                self.assertIsNone(self.x.grad)


class AtomicityTest(unittest.TestCase):

    def test_forward_failure_leaves_input_untouched(self):
        x = make_bad(data=[0.3, float("inf")], requires_grad=True)
        original = list(x.data)
        with self.assertRaises(ValueError):
            x.log_softmax_batch(1, 2)
        self.assertEqual(x.data, original)
        self.assertIsNone(x.grad)

    def test_non_finite_upstream_is_atomic(self):
        xs = [1000.0, -1000.0]
        x = Tensor(xs, True)
        out = x.log_softmax_batch(1, 2)
        with self.assertRaises(ValueError):
            out.backward([float("inf"), 1.0])
        self.assertIsNone(x.grad)
        g = [1.0, 1.0]
        out.backward(g)
        l = ref_log_softmax_batch(xs, 1, 2)
        self.assertEqual(x.grad, ref_log_softmax_batch_grad(l, 1, 2, g))

    def test_existing_grad_merge_overflow_is_atomic(self):
        xs = [1.0, 0.0]
        x = Tensor(xs, True)
        out = x.log_softmax_batch(1, 2)
        x.grad = [1.7e308, 1.7e308]
        with self.assertRaises(ValueError):
            out.backward([1.7e308, -1.7e308])
        self.assertEqual(x.grad, [1.7e308, 1.7e308])
        x.zero_grad()
        g = [1.0, 1.0]
        out.backward(g)
        l = ref_log_softmax_batch(xs, 1, 2)
        self.assertEqual(x.grad, ref_log_softmax_batch_grad(l, 1, 2, g))


if __name__ == "__main__":
    unittest.main()
