"""Verifiable delivery tests for Tensor.binary_cross_entropy_with_logits.

Covers hand-computed and gradcheck-verified forward values (including
+/-1000 logits), every requires-grad combination, upstream grad scaling,
same-tensor/shared-path/repeated backward, snapshot semantics, the full
TypeError/ValueError contract, and atomicity of failed forward/backward
passes. Standard library only.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def _sigmoid(x):
    """Sigmoid via the non-overflowing branch for each sign."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _bce_mean(xs, ys):
    """Independent hand computation: mean of max(x,0) - x*y + log1p(exp(-|x|))."""
    total = 0.0
    for x_i, y_i in zip(xs, ys):
        total += max(x_i, 0.0) - x_i * y_i + math.log1p(math.exp(-abs(x_i)))
    return total / len(xs)


def _bce_dx(xs, ys, g):
    """Hand-computed dL/dx_i = g * (sigmoid(x_i) - y_i) / n."""
    n = len(xs)
    return [g * (_sigmoid(x_i) - y_i) / n for x_i, y_i in zip(xs, ys)]


def _bce_dy(xs, g):
    """Hand-computed dL/dy_i = -g * x_i / n."""
    n = len(xs)
    return [-g * x_i / n for x_i in xs]


class TestForward(unittest.TestCase):
    def test_hand_computed_values(self):
        xs = [0.0, 1.0, -2.0]
        ys = [0.0, 1.0, 0.5]
        out = Tensor(xs).binary_cross_entropy_with_logits(Tensor(ys))
        # Independent decimal expansion of the hand-derived mean.
        self.assertAlmostEqual(out.data, 0.7111122930403802, places=14)
        self.assertEqual(out.data, _bce_mean(xs, ys))
        self.assertIsInstance(out.data, float)

    def test_single_element(self):
        out = Tensor([2.0]).binary_cross_entropy_with_logits(Tensor([1.0]))
        self.assertEqual(out.data, _bce_mean([2.0], [1.0]))
        self.assertAlmostEqual(out.data, math.log1p(math.exp(-2.0)), places=15)

    def test_large_logits_stable(self):
        # Saturated logits: term = max(x,0) - x*y since log1p(exp(-|x|)) = 0.
        out = Tensor([1000.0, -1000.0]).binary_cross_entropy_with_logits(
            Tensor([1.0, 0.0])
        )
        self.assertEqual(out.data, 0.0)
        out = Tensor([1000.0, -1000.0]).binary_cross_entropy_with_logits(
            Tensor([0.0, 1.0])
        )
        self.assertEqual(out.data, 1000.0)
        out = Tensor([1000.0, -1000.0]).binary_cross_entropy_with_logits(
            Tensor([0.5, 0.5])
        )
        self.assertEqual(out.data, 500.0)
        # Matches the stable hand formula exactly.
        self.assertEqual(
            out.data, _bce_mean([1000.0, -1000.0], [0.5, 0.5])
        )

    def test_large_logits_backward_stable(self):
        x = Tensor([1000.0, -1000.0], True)
        y = Tensor([0.25, 0.75], True)
        x.binary_cross_entropy_with_logits(y).backward()
        # sigmoid saturates to exactly 1.0 / 0.0.
        self.assertEqual(x.grad, [(1.0 - 0.25) / 2.0, (0.0 - 0.75) / 2.0])
        self.assertEqual(y.grad, [-1000.0 / 2.0, 1000.0 / 2.0])

    def test_gradcheck_self_side(self):
        passed, err = gradcheck(
            lambda t: t.binary_cross_entropy_with_logits(Tensor([0.25, 0.75])),
            [0.3, -1.2],
        )
        self.assertTrue(passed, err)

    def test_gradcheck_target_side(self):
        passed, err = gradcheck(
            lambda t: Tensor([0.4, -0.9]).binary_cross_entropy_with_logits(t),
            [0.2, 0.6],
        )
        self.assertTrue(passed, err)

    def test_gradcheck_same_tensor_both_sides(self):
        passed, err = gradcheck(
            lambda t: t.binary_cross_entropy_with_logits(t), [0.3, 0.7]
        )
        self.assertTrue(passed, err)


class TestBackward(unittest.TestCase):
    def test_self_only_requires_grad(self):
        x = Tensor([0.0, 1.0, -2.0], True)
        y = Tensor([0.0, 1.0, 0.5])
        out = x.binary_cross_entropy_with_logits(y)
        self.assertTrue(out.requires_grad)
        out.backward()
        self.assertEqual(x.grad, _bce_dx([0.0, 1.0, -2.0], [0.0, 1.0, 0.5], 1.0))
        self.assertIsNone(y.grad)

    def test_target_only_requires_grad(self):
        x = Tensor([0.0, 1.0, -2.0])
        y = Tensor([0.0, 1.0, 0.5], True)
        x.binary_cross_entropy_with_logits(y).backward()
        self.assertIsNone(x.grad)
        self.assertEqual(y.grad, _bce_dy([0.0, 1.0, -2.0], 1.0))

    def test_both_require_grad(self):
        xs = [0.0, 1.0, -2.0]
        ys = [0.0, 1.0, 0.5]
        x = Tensor(xs, True)
        y = Tensor(ys, True)
        x.binary_cross_entropy_with_logits(y).backward()
        self.assertEqual(x.grad, _bce_dx(xs, ys, 1.0))
        self.assertEqual(y.grad, _bce_dy(xs, 1.0))

    def test_neither_requires_grad_builds_no_graph(self):
        out = Tensor([0.5, -0.5]).binary_cross_entropy_with_logits(
            Tensor([0.5, 0.5])
        )
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward()

    def test_finite_float_upstream_scales_grads(self):
        xs = [0.3, -0.7]
        ys = [0.25, 1.0]
        x = Tensor(xs, True)
        y = Tensor(ys, True)
        x.binary_cross_entropy_with_logits(y).backward(2.5)
        self.assertEqual(x.grad, _bce_dx(xs, ys, 2.5))
        self.assertEqual(y.grad, _bce_dy(xs, 2.5))

    def test_same_tensor_on_both_sides(self):
        xs = [0.3, 0.7]
        x = Tensor(xs, True)
        x.binary_cross_entropy_with_logits(x).backward()
        # d/dx_i of bce(x, x) = (sigmoid(x_i) - x_i)/n - x_i/n.
        expected = [
            (_sigmoid(x_i) - x_i) / 2.0 - x_i / 2.0 for x_i in xs
        ]
        self.assertEqual(x.grad, expected)

    def test_shared_path_accumulates(self):
        xs = [0.5, -0.5]
        y1s = [0.0, 1.0]
        y2s = [1.0, 0.0]
        x = Tensor(xs, True)
        t1 = Tensor(y1s, True)
        t2 = Tensor(y2s, True)
        out1 = x.binary_cross_entropy_with_logits(t1)
        out2 = x.binary_cross_entropy_with_logits(t2)
        out1.add(out2).backward()
        expected_x = [
            a + b
            for a, b in zip(_bce_dx(xs, y1s, 1.0), _bce_dx(xs, y2s, 1.0))
        ]
        self.assertEqual(x.grad, expected_x)
        self.assertEqual(t1.grad, _bce_dy(xs, 1.0))
        self.assertEqual(t2.grad, _bce_dy(xs, 1.0))

    def test_repeated_backward_accumulates(self):
        xs = [0.3, -0.4]
        ys = [0.5, 0.5]
        x = Tensor(xs, True)
        y = Tensor(ys, True)
        x.binary_cross_entropy_with_logits(y).backward()
        x.binary_cross_entropy_with_logits(y).backward()
        self.assertEqual(
            x.grad, [2.0 * v for v in _bce_dx(xs, ys, 1.0)]
        )
        self.assertEqual(y.grad, [2.0 * v for v in _bce_dy(xs, 1.0)])

    def test_snapshot_survives_data_mutation(self):
        xs = [0.3, -1.2]
        ys = [0.25, 0.75]
        x = Tensor(xs, True)
        y = Tensor(ys, True)
        out = x.binary_cross_entropy_with_logits(y)
        x.data[0] = 999.0
        y.data[1] = 0.0
        out.backward()
        self.assertEqual(x.grad, _bce_dx(xs, ys, 1.0))
        self.assertEqual(y.grad, _bce_dy(xs, 1.0))

    def test_snapshot_survives_data_replacement(self):
        xs = [0.3, -1.2]
        ys = [0.25, 0.75]
        x = Tensor(xs, True)
        y = Tensor(ys, True)
        out = x.binary_cross_entropy_with_logits(y)
        x.data = [50.0, 60.0]
        y.data = [0.0, 0.0]
        out.backward()
        self.assertEqual(x.grad, _bce_dx(xs, ys, 1.0))
        self.assertEqual(y.grad, _bce_dy(xs, 1.0))


class TestContractErrors(unittest.TestCase):
    def test_target_not_tensor(self):
        x = Tensor([0.5])
        for bad in (None, True, 1, 0.5, "t", [0.5], (0.5,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.binary_cross_entropy_with_logits(bad)

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            Tensor(0.5).binary_cross_entropy_with_logits(Tensor([0.5]))
        with self.assertRaises(ValueError):
            Tensor([0.5]).binary_cross_entropy_with_logits(Tensor(0.5))
        x = Tensor([0.5])
        x.data = "nope"
        with self.assertRaises(ValueError):
            x.binary_cross_entropy_with_logits(Tensor([0.5]))

    def test_data_empty_list(self):
        x = Tensor([0.5])
        x.data = []
        with self.assertRaises(ValueError):
            x.binary_cross_entropy_with_logits(Tensor([0.5]))
        y = Tensor([0.5])
        y.data = []
        with self.assertRaises(ValueError):
            Tensor([0.5]).binary_cross_entropy_with_logits(y)

    def test_element_type_errors(self):
        for bad_element in (True, 1, "x", None):
            with self.subTest(bad=bad_element):
                x = Tensor([0.5, 0.5])
                x.data = [0.5, bad_element]
                with self.assertRaises(TypeError):
                    x.binary_cross_entropy_with_logits(Tensor([0.5, 0.5]))
                y = Tensor([0.5, 0.5])
                y.data = [0.5, bad_element]
                with self.assertRaises(TypeError):
                    Tensor([0.5, 0.5]).binary_cross_entropy_with_logits(y)

    def test_requires_grad_type_errors(self):
        x = Tensor([0.5])
        x.requires_grad = 1
        with self.assertRaises(TypeError):
            x.binary_cross_entropy_with_logits(Tensor([0.5]))
        y = Tensor([0.5])
        y.requires_grad = "yes"
        with self.assertRaises(TypeError):
            Tensor([0.5]).binary_cross_entropy_with_logits(y)

    def test_non_finite_elements(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(bad=bad):
                x = Tensor([0.5, 0.5])
                x.data = [0.5, bad]
                with self.assertRaises(ValueError):
                    x.binary_cross_entropy_with_logits(Tensor([0.5, 0.5]))
                y = Tensor([0.5, 0.5])
                y.data = [0.5, bad]
                with self.assertRaises(ValueError):
                    Tensor([0.5, 0.5]).binary_cross_entropy_with_logits(y)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.5, 0.5]).binary_cross_entropy_with_logits(
                Tensor([0.5, 0.5, 0.5])
            )
        with self.assertRaises(ValueError):
            Tensor([0.5, 0.5, 0.5]).binary_cross_entropy_with_logits(
                Tensor([0.5])
            )

    def test_target_out_of_range(self):
        for bad in (-0.5, 1.5, -1e-9, 1.0 + 1e-9):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Tensor([0.5, 0.5]).binary_cross_entropy_with_logits(
                        Tensor([0.5, bad])
                    )

    def test_upstream_grad_type_errors(self):
        out = Tensor([0.5], True).binary_cross_entropy_with_logits(
            Tensor([0.5])
        )
        for bad in (None, True, 1, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    out.backward(bad)

    def test_upstream_grad_value_errors(self):
        # The output is scalar: a list upstream is a shape mismatch, and
        # non-finite upstream values are rejected.
        for bad in ([1.0], float("inf"), float("-inf"), float("nan")):
            out = Tensor([0.5], True).binary_cross_entropy_with_logits(
                Tensor([0.5])
            )
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    out.backward(bad)


class TestAtomicity(unittest.TestCase):
    def test_forward_overflow_leaves_state_untouched(self):
        # Each term is ~1e308; the running total overflows on the second.
        x = Tensor([1e308, 1e308], True)
        y = Tensor([0.0, 0.0], True)
        x.grad = [1.0, 2.0]
        y.grad = [3.0, 4.0]
        with self.assertRaises(ValueError):
            x.binary_cross_entropy_with_logits(y)
        self.assertEqual(x.data, [1e308, 1e308])
        self.assertEqual(y.data, [0.0, 0.0])
        self.assertEqual(x.grad, [1.0, 2.0])
        self.assertEqual(y.grad, [3.0, 4.0])

    def test_backward_overflow_leaves_whole_graph_untouched(self):
        # g * x_i overflows in the dy path (1e308 * -1e308).
        x = Tensor([-1e308], True)
        t1 = Tensor([0.0], True)
        t2 = Tensor([0.5], True)
        out1 = x.binary_cross_entropy_with_logits(t1)
        out2 = x.binary_cross_entropy_with_logits(t2)
        combined = out1.add(out2)
        combined.backward(1.0)
        before = {
            "x": list(x.grad),
            "t1": list(t1.grad),
            "t2": list(t2.grad),
            "out1": out1.grad,
            "out2": out2.grad,
            "combined": combined.grad,
        }
        with self.assertRaises(ValueError):
            combined.backward(1e308)
        self.assertEqual(x.grad, before["x"])
        self.assertEqual(t1.grad, before["t1"])
        self.assertEqual(t2.grad, before["t2"])
        self.assertEqual(out1.grad, before["out1"])
        self.assertEqual(out2.grad, before["out2"])
        self.assertEqual(combined.grad, before["combined"])

    def test_backward_overflow_with_no_prior_grads(self):
        x = Tensor([-1e308], True)
        y = Tensor([0.0], True)
        out = x.binary_cross_entropy_with_logits(y)
        self.assertEqual(out.data, 0.0)
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_existing_grad_merge_overflow_leaves_grads_untouched(self):
        x = Tensor([1.0], True)
        y = Tensor([0.0], True)
        out = x.binary_cross_entropy_with_logits(y)
        x.grad = [1.5e308]
        # New dx ~ 7.3e307; merging into the existing 1.5e308 overflows.
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertEqual(x.grad, [1.5e308])
        self.assertIsNone(y.grad)


if __name__ == "__main__":
    unittest.main()
