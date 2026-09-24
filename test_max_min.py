"""Tests for Tensor.max / Tensor.min: differentiable extreme reductions."""

import math
import unittest

from autograd import Tensor


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation."""
    return Tensor._make(data, requires_grad, (), None)


class MaxForwardTests(unittest.TestCase):
    def test_scalar_returns_scalar(self):
        out = Tensor(3.5).max()
        self.assertIsInstance(out.data, float)
        self.assertEqual(out.data, 3.5)

    def test_vector_max(self):
        self.assertEqual(Tensor([2.0, 5.0, 3.0, 5.0, 1.0]).max().data, 5.0)

    def test_vector_max_descending_inputs(self):
        self.assertEqual(Tensor([4.0, 3.0, 2.0, 1.0]).max().data, 4.0)

    def test_single_element_vector(self):
        self.assertEqual(Tensor([7.0]).max().data, 7.0)

    def test_negative_values(self):
        self.assertEqual(Tensor([-5.0, -2.0, -9.0]).max().data, -2.0)

    def test_signed_zero_tie_forward(self):
        self.assertEqual(Tensor([-0.0, 0.0]).max().data, -0.0)

    def test_forward_type_errors(self):
        for bad in (True, 1, "x", None, (1.0,), [[1.0]], [1.0, 2]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).max()
        for bad in ([1.0, True], [1.0, 2], [1.0, [2.0]]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).max()

    def test_forward_value_errors(self):
        with self.assertRaises(ValueError):
            _raw_tensor([]).max()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).max()
            with self.assertRaises(ValueError):
                _raw_tensor([1.0, bad]).max()

    def test_non_bool_requires_grad_is_type_error(self):
        with self.assertRaises(TypeError):
            _raw_tensor([1.0, 2.0], requires_grad=1).max()

    def test_failed_forward_leaves_input_unchanged(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        x.data = float("inf")
        with self.assertRaises(ValueError):
            x.max()
        self.assertEqual(x.data, float("inf"))
        self.assertIsNone(x.grad)

    def test_no_grad_has_no_graph(self):
        out = Tensor([2.0, 3.0]).max()
        self.assertFalse(out.requires_grad)
        self.assertEqual(out.data, 3.0)
        with self.assertRaises(ValueError):
            out.backward()

    def test_maximum_still_elementwise(self):
        out = Tensor([1.0, 5.0]).maximum(Tensor([4.0, 2.0]))
        self.assertEqual(out.data, [4.0, 5.0])


class MinForwardTests(unittest.TestCase):
    def test_scalar_returns_scalar(self):
        out = Tensor(3.5).min()
        self.assertIsInstance(out.data, float)
        self.assertEqual(out.data, 3.5)

    def test_vector_min(self):
        self.assertEqual(Tensor([2.0, 5.0, 3.0, 1.0, 1.0]).min().data, 1.0)

    def test_single_element_vector(self):
        self.assertEqual(Tensor([7.0]).min().data, 7.0)

    def test_negative_values(self):
        self.assertEqual(Tensor([-5.0, -2.0, -9.0]).min().data, -9.0)

    def test_forward_type_errors(self):
        for bad in (True, 1, "x", None, (1.0,), [[1.0]], [1.0, 2]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).min()
        for bad in ([1.0, True], [1.0, 2], [1.0, [2.0]]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).min()

    def test_forward_value_errors(self):
        with self.assertRaises(ValueError):
            _raw_tensor([]).min()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).min()
            with self.assertRaises(ValueError):
                _raw_tensor([1.0, bad]).min()

    def test_non_bool_requires_grad_is_type_error(self):
        with self.assertRaises(TypeError):
            _raw_tensor([1.0, 2.0], requires_grad=1).min()

    def test_no_grad_has_no_graph(self):
        out = Tensor([2.0, 3.0]).min()
        self.assertFalse(out.requires_grad)
        self.assertEqual(out.data, 2.0)
        with self.assertRaises(ValueError):
            out.backward()

    def test_minimum_still_elementwise(self):
        out = Tensor([1.0, 5.0]).minimum(Tensor([4.0, 2.0]))
        self.assertEqual(out.data, [1.0, 2.0])


class MaxBackwardTests(unittest.TestCase):
    def test_scalar_default_grad(self):
        x = Tensor(3.5, requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, 1.0)

    def test_scalar_explicit_grad(self):
        x = Tensor(3.5, requires_grad=True)
        x.max().backward(2.0)
        self.assertEqual(x.grad, 2.0)

    def test_unique_winner(self):
        x = Tensor([2.0, 5.0, 3.0], requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, [0.0, 1.0, 0.0])

    def test_winner_at_first_and_last_positions(self):
        x = Tensor([5.0, 1.0, 5.0], requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, [0.5, 0.0, 0.5])

    def test_three_way_tie_splits_evenly(self):
        x = Tensor([4.0, 4.0, 1.0, 4.0], requires_grad=True)
        x.max().backward(3.0)
        self.assertEqual(x.grad, [1.0, 1.0, 0.0, 1.0])

    def test_all_equal_splits_evenly(self):
        x = Tensor([2.0, 2.0, 2.0, 2.0], requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_single_element_vector(self):
        x = Tensor([7.0], requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, [1.0])

    def test_signed_zero_tie_splits(self):
        x = Tensor([-0.0, 0.0, -1.0], requires_grad=True)
        x.max().backward()
        self.assertEqual(x.grad, [0.5, 0.5, 0.0])

    def test_neg_zero_grad_is_accepted(self):
        # -0.0 is a finite float, so it is a legal upstream grad.
        x = Tensor([2.0, 5.0], requires_grad=True)
        x.max().backward(-0.0)
        self.assertEqual(x.grad, [0.0, -0.0])

    def test_grad_type_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).max()
        for bad in (None, True, 1, "x"):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_grad_value_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).max()
        with self.assertRaises(ValueError):
            out.backward([1.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                out.backward(bad)

    def test_failed_backward_leaves_all_grads_untouched(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor(1.0, requires_grad=True)
        out = x.max().mul(y)
        with self.assertRaises(ValueError):
            out.backward(float("inf"))
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([2.0, 5.0, 3.0, 5.0], requires_grad=True)
        out = x.max()
        x.data[0] = 100.0
        x.data[1] = 0.0
        out.backward()
        self.assertEqual(x.grad, [0.0, 0.5, 0.0, 0.5])

    def test_repeated_backward_accumulates(self):
        x = Tensor([2.0, 5.0, 5.0], requires_grad=True)
        out = x.max()
        out.backward()
        out.backward(2.0)
        self.assertEqual(x.grad, [0.0, 1.5, 1.5])

    def test_shared_path_accumulates(self):
        x = Tensor([2.0, 5.0, 5.0], requires_grad=True)
        out = x.max().add(x.max())
        out.backward()
        self.assertEqual(x.grad, [0.0, 1.0, 1.0])


class MinBackwardTests(unittest.TestCase):
    def test_scalar_default_grad(self):
        x = Tensor(3.5, requires_grad=True)
        x.min().backward()
        self.assertEqual(x.grad, 1.0)

    def test_unique_winner(self):
        x = Tensor([2.0, 5.0, 1.0, 3.0], requires_grad=True)
        x.min().backward()
        self.assertEqual(x.grad, [0.0, 0.0, 1.0, 0.0])

    def test_ties_split_evenly(self):
        x = Tensor([1.0, 5.0, 1.0, 1.0], requires_grad=True)
        x.min().backward(6.0)
        self.assertEqual(x.grad, [2.0, 0.0, 2.0, 2.0])

    def test_single_element_vector(self):
        x = Tensor([7.0], requires_grad=True)
        x.min().backward()
        self.assertEqual(x.grad, [1.0])

    def test_signed_zero_tie_splits(self):
        x = Tensor([0.0, -0.0, 1.0], requires_grad=True)
        x.min().backward()
        self.assertEqual(x.grad, [0.5, 0.5, 0.0])

    def test_neg_zero_grad_is_accepted(self):
        # -0.0 is a finite float, so it is a legal upstream grad.
        x = Tensor([2.0, 5.0], requires_grad=True)
        x.min().backward(-0.0)
        self.assertEqual(x.grad, [-0.0, 0.0])

    def test_grad_type_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).min()
        for bad in (None, True, 1, "x"):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_grad_value_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).min()
        with self.assertRaises(ValueError):
            out.backward([1.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                out.backward(bad)

    def test_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([2.0, 1.0, 3.0, 1.0], requires_grad=True)
        out = x.min()
        x.data[1] = 100.0
        x.data[0] = -50.0
        out.backward()
        self.assertEqual(x.grad, [0.0, 0.5, 0.0, 0.5])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 5.0, 1.0], requires_grad=True)
        out = x.min()
        out.backward()
        out.backward()
        self.assertEqual(x.grad, [1.0, 0.0, 1.0])

    def test_shared_path_accumulates(self):
        x = Tensor([1.0, 5.0, 1.0], requires_grad=True)
        out = x.min().add(x.min())
        out.backward()
        self.assertEqual(x.grad, [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
