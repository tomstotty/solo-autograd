"""Verifiable contract tests for Tensor.masked_softmax.

Batched masked softmax over a flattened rows x cols float list with a
same-length bool mask. Standard library only; discovered by
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def ref_masked_softmax(xs, mask, rows, cols):
    """Independent hand computation in the prescribed operation order.

    For each row, over the true positions in ascending column order:
    m = max of the true positions, z_i = exp(x_i - m), s = sum(z),
    y_i = z_i / s; false positions are 0.0.
    """
    y = [0.0] * len(xs)
    for q in range(rows):
        base = q * cols
        true_idx = [base + j for j in range(cols) if mask[base + j]]
        m = max(xs[k] for k in true_idx)
        z = [(k, math.exp(xs[k] - m)) for k in true_idx]
        s = 0.0
        for _, z_i in z:
            s += z_i
        for k, z_i in z:
            y[k] = z_i / s
    return y


def ref_masked_softmax_grad(y, mask, rows, cols, g):
    """Per row: d = sum(g_i * y_i) over true positions, dx_i = y_i*(g_i-d)."""
    dx = [0.0] * len(y)
    for q in range(rows):
        base = q * cols
        d = 0.0
        for j in range(cols):
            k = base + j
            if mask[k]:
                d += g[k] * y[k]
        for j in range(cols):
            k = base + j
            if mask[k]:
                dx[k] = y[k] * (g[k] - d)
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
    ([0.3, -1.5, 2.0, 1.0, 0.2, -0.4],
     [True, False, True, True, True, False], 2, 3),
    ([2.0, 0.0, 1.0, 0.3, -1.5, 2.0],
     [True, True, True, True, True, True], 2, 3),
    ([0.3, -1.5, 2.0], [False, True, False], 1, 3),
    ([-0.0, 0.0, 5.5, -7.25], [True, True, False, True], 2, 2),
    ([5.5, -7.25, 0.125, -0.375, 1.0, 2.0, 3.0, 4.0],
     [True, False, True, False, True, True, False, True], 2, 4),
    ([1e-12, -1e-12], [True, True], 1, 2),
    ([7.0], [True], 1, 1),
]


class ForwardValueTest(unittest.TestCase):

    def test_hand_computed_cases(self):
        for xs, mask, rows, cols in CASES:
            with self.subTest(xs=xs, mask=mask):
                out = Tensor(xs).masked_softmax(mask, rows, cols)
                self.assertEqual(
                    out.data, ref_masked_softmax(xs, mask, rows, cols)
                )

    def test_false_positions_are_zero(self):
        out = Tensor([0.3, -1.5, 2.0, 1.0]).masked_softmax(
            [True, False, False, True], 2, 2
        )
        self.assertEqual(out.data[1], 0.0)
        self.assertEqual(out.data[2], 0.0)

    def test_single_true_per_row_is_one(self):
        out = Tensor([0.3, -1.5, 2.0, 1.0]).masked_softmax(
            [False, True, False, True], 2, 2
        )
        self.assertEqual(out.data, [0.0, 1.0, 0.0, 1.0])

    def test_all_true_matches_softmax_rows(self):
        rows = [[2.0, 0.0, 1.0], [0.3, -1.5, 2.0]]
        xs = rows[0] + rows[1]
        mask = [True] * 6
        masked = Tensor(xs).masked_softmax(mask, 2, 3).data
        plain0 = Tensor(rows[0]).softmax().data
        plain1 = Tensor(rows[1]).softmax().data
        self.assertEqual(masked, plain0 + plain1)

    def test_each_true_row_sums_to_one(self):
        for xs, mask, rows, cols in CASES:
            with self.subTest(xs=xs, mask=mask):
                out = Tensor(xs).masked_softmax(mask, rows, cols).data
                for q in range(rows):
                    row = out[q * cols:(q + 1) * cols]
                    self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_large_logits_stable(self):
        out = Tensor(
            [1000.0, -1000.0, 1000.0, -1000.0]
        ).masked_softmax([True, True, True, True], 2, 2)
        self.assertEqual(out.data, [1.0, 0.0, 1.0, 0.0])

    def test_output_is_vector(self):
        out = Tensor([0.3, -1.5, 1.0, 0.2]).masked_softmax(
            [True, True, True, False], 2, 2
        )
        self.assertIsInstance(out.data, list)
        self.assertEqual(len(out.data), 4)

    def test_gradcheck(self):
        for xs, mask, rows, cols in CASES:
            with self.subTest(xs=xs, mask=mask):
                w = Tensor(
                    [0.7, -1.3, 2.1, 0.4, -0.9, 1.6, -0.5, 1.2][:len(xs)]
                )
                passed, err = gradcheck(
                    lambda t, m=mask, r=rows, c=cols, w=w:
                        t.masked_softmax(m, r, c).dot(w),
                    xs,
                )
                self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_matches_reference(self):
        xs = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4]
        mask = [True, False, True, True, True, False]
        g = [0.5, 9.0, -1.5, 2.0, 0.25, 9.0]
        x = Tensor(xs, True)
        x.masked_softmax(mask, 2, 3).backward(g)
        y = ref_masked_softmax(xs, mask, 2, 3)
        self.assertEqual(
            x.grad, ref_masked_softmax_grad(y, mask, 2, 3, g)
        )

    def test_backward_false_positions_zero(self):
        x = Tensor([0.3, -1.5, 2.0, 1.0], True)
        x.masked_softmax([True, False, False, True], 2, 2).backward(
            [1.0, 7.0, 8.0, 3.0]
        )
        self.assertEqual(x.grad[1], 0.0)
        self.assertEqual(x.grad[2], 0.0)

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5, 0.1, 0.2]).masked_softmax(
            [True, True, True, True], 2, 2
        )
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0])

    def test_snapshot_survives_mutation_and_replacement(self):
        xs = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4]
        mask = [True, False, True, True, True, False]
        g = [1.0, 9.0, 1.0, 1.0, 1.0, 9.0]
        x = Tensor(xs, True)
        out = x.masked_softmax(mask, 2, 3)
        x.data[0] = 999.0           # in-place mutation after forward
        x.data = [9.0] * 6          # wholesale replacement after forward
        mask[0] = False             # mask mutation after forward
        mask.append(True)
        out.backward(g)
        y = ref_masked_softmax(xs, [True, False, True, True, True, False],
                               2, 3)
        self.assertEqual(out.data, y)
        self.assertEqual(
            x.grad,
            ref_masked_softmax_grad(
                y, [True, False, True, True, True, False], 2, 3, g
            ),
        )

    def test_shared_path_accumulates(self):
        xs = [0.3, -1.5, 1.0, 0.2]
        mask = [True, True, True, True]
        g = [0.5, -1.0, 2.0, 0.25]
        x = Tensor(xs, True)
        out = x.masked_softmax(mask, 2, 2)
        out.add(out).backward(g)
        y = ref_masked_softmax(xs, mask, 2, 2)
        self.assertEqual(
            x.grad, ref_masked_softmax_grad(y, mask, 2, 2, [2 * v for v in g])
        )

    def test_repeated_backward_accumulates(self):
        xs = [0.3, -1.5, 1.0, 0.2]
        mask = [True, True, True, True]
        g = [0.5, -1.0, 2.0, 0.25]
        x = Tensor(xs, True)
        out = x.masked_softmax(mask, 2, 2)
        out.backward(g)
        out.backward(g)
        y = ref_masked_softmax(xs, mask, 2, 2)
        self.assertEqual(
            x.grad, ref_masked_softmax_grad(y, mask, 2, 2, [2 * v for v in g])
        )


class ErrorContractTest(unittest.TestCase):

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).masked_softmax([True], 1, 1)

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).masked_softmax([], 1, 1)

    def test_data_not_list_checked_first(self):
        # The data precondition is checked before rows, cols and mask.
        with self.assertRaises(ValueError):
            make_bad(data=0.5).masked_softmax("x", "x", "x")

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).masked_softmax(
                        [True] * len(bad), 1, len(bad)
                    )

    def test_requires_grad_type(self):
        with self.assertRaises(TypeError):
            make_bad(requires_grad=1).masked_softmax([True], 1, 1)
        with self.assertRaises(TypeError):
            make_bad(requires_grad="yes").masked_softmax([True], 1, 1)

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan"), 0.1],
                    [-float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_bad(data=bad).masked_softmax(
                        [True] * len(bad), 1, len(bad)
                    )

    def test_rows_cols_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (None, 1.0, 0.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.masked_softmax([True, True], bad, 2)
                with self.assertRaises(TypeError):
                    x.masked_softmax([True, True], 1, bad)

    def test_rows_cols_bool_is_type_error(self):
        x = Tensor([0.3, 0.4])
        with self.assertRaises(TypeError):
            x.masked_softmax([True, True], True, 2)
        with self.assertRaises(TypeError):
            x.masked_softmax([True, True], 1, False)

    def test_rows_cols_non_positive(self):
        x = Tensor([0.3, 0.4])
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.masked_softmax([True, True], bad, 2)
                with self.assertRaises(ValueError):
                    x.masked_softmax([True, True], 1, bad)

    def test_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2, 0.3]).masked_softmax([True] * 3, 2, 2)
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2, 0.3, 0.4, 0.5]).masked_softmax(
                [True] * 5, 2, 3
            )

    def test_mask_type(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in (None, True, False, 1, 1.0, "tt", (True, True),
                    {0: True}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.masked_softmax(bad, 2, 2)

    def test_mask_length(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in ([], [True], [True] * 3, [True] * 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.masked_softmax(bad, 2, 2)

    def test_mask_element_type(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in ([True, 1, True, True], [True, True, 0.0, True],
                    [True, None, True, True], [True, "t", True, True]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.masked_softmax(bad, 2, 2)

    def test_row_without_true(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in ([False, False, True, True],
                    [True, True, False, False],
                    [False, False, False, False]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.masked_softmax(bad, 2, 2)

    def test_rows_cols_checked_before_mask(self):
        # A bad rows type is a TypeError even with a malformed mask, and
        # a non-positive cols is a ValueError even when the mask length
        # cannot match.
        with self.assertRaises(TypeError):
            Tensor([0.1, 0.2]).masked_softmax("x", 1.0, 2)
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2]).masked_softmax([True], 1, 0)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5, 1.0, 0.2], True)
        self.out = self.x.masked_softmax([True, True, True, True], 2, 2)

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
            x.masked_softmax([True, True], 1, 2)
        self.assertEqual(x.data, original)
        self.assertIsNone(x.grad)

    def test_non_finite_upstream_is_atomic(self):
        x = Tensor([1000.0, -1000.0], True)
        out = x.masked_softmax([True, True], 1, 2)
        with self.assertRaises(ValueError):
            out.backward([float("inf"), 1.0])
        self.assertIsNone(x.grad)
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [0.0, 0.0])

    def test_existing_grad_merge_overflow_is_atomic(self):
        x = Tensor([1.0, 0.0], True)
        out = x.masked_softmax([True, True], 1, 2)
        x.grad = [1.7e308, 1.7e308]
        with self.assertRaises(ValueError):
            out.backward([1.7e308, -1.7e308])
        self.assertEqual(x.grad, [1.7e308, 1.7e308])
        x.zero_grad()
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
