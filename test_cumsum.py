"""Tests for Tensor.cumsum: differentiable 1D cumulative sum, four modes."""

import math
import unittest

from autograd import Tensor


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation.

    Lets cumsum() itself be exercised against malformed data or a
    non-bool requires_grad flag, which the ordinary constructor rejects.
    """
    return Tensor._make(data, requires_grad, (), None)


class CumsumForwardTests(unittest.TestCase):
    def test_inclusive_forward(self):
        self.assertEqual(Tensor([1.0, 2.0, 3.0]).cumsum().data,
                         [1.0, 3.0, 6.0])

    def test_exclusive_forward(self):
        self.assertEqual(
            Tensor([1.0, 2.0, 3.0]).cumsum(exclusive=True).data,
            [0.0, 1.0, 3.0])

    def test_reverse_inclusive_forward(self):
        self.assertEqual(
            Tensor([1.0, 2.0, 3.0]).cumsum(reverse=True).data,
            [6.0, 5.0, 3.0])

    def test_reverse_exclusive_forward(self):
        self.assertEqual(
            Tensor([1.0, 2.0, 3.0]).cumsum(exclusive=True,
                                           reverse=True).data,
            [5.0, 3.0, 0.0])

    def test_single_element(self):
        self.assertEqual(Tensor([7.0]).cumsum().data, [7.0])
        self.assertEqual(Tensor([7.0]).cumsum(exclusive=True).data, [0.0])
        self.assertEqual(Tensor([7.0]).cumsum(reverse=True).data, [7.0])
        self.assertEqual(
            Tensor([7.0]).cumsum(exclusive=True, reverse=True).data, [0.0])

    def test_negative_elements(self):
        self.assertEqual(Tensor([-2.0, 3.0, -4.0]).cumsum().data,
                         [-2.0, 1.0, -3.0])

    def test_reverse_order_keeps_intermediate_finite(self):
        # Reverse order keeps the partial sums finite where ascending
        # order would overflow.
        values = [1e308, 1e308, -1e308]
        out = Tensor(values).cumsum(reverse=True).data
        self.assertEqual(out, [1e308, 0.0, -1e308])

    def test_partial_sum_overflow_is_value_error(self):
        with self.assertRaises(ValueError):
            Tensor([1e308, 1e308]).cumsum()
        with self.assertRaises(ValueError):
            Tensor([1e308, 1e308]).cumsum(reverse=True)

    def test_exclusive_trailing_partial_sum_overflow_is_value_error(self):
        # The output itself would be finite, but the final accumulated
        # partial sum is not.
        with self.assertRaises(ValueError):
            Tensor([1e308, 1e308]).cumsum(exclusive=True)

    def test_failed_forward_leaves_input_unchanged(self):
        x = Tensor([1e308, 1e308], requires_grad=True)
        with self.assertRaises(ValueError):
            x.cumsum()
        self.assertEqual(x.data, [1e308, 1e308])
        self.assertIsNone(x.grad)

    def test_non_list_and_empty_are_value_errors(self):
        for bad in (1.0, True, 1, "x", None, (1.0,)):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).cumsum()
        with self.assertRaises(ValueError):
            _raw_tensor([]).cumsum()

    def test_non_float_elements_are_type_errors(self):
        for bad in ([True], [1], [1.0, 2], [1.0, [2.0]], [[1.0]]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).cumsum()

    def test_non_finite_elements_are_value_errors(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor([1.0, bad]).cumsum()

    def test_non_bool_requires_grad_is_type_error(self):
        with self.assertRaises(TypeError):
            _raw_tensor([1.0, 2.0], requires_grad=1).cumsum()

    def test_non_bool_flags_are_type_errors(self):
        x = Tensor([1.0, 2.0])
        for bad in (0, 1, 1.0, "x", None):
            with self.assertRaises(TypeError):
                x.cumsum(exclusive=bad)
            with self.assertRaises(TypeError):
                x.cumsum(reverse=bad)

    def test_no_grad_has_no_graph(self):
        out = Tensor([2.0, 3.0]).cumsum()
        self.assertFalse(out.requires_grad)
        # A correctly shaped finite grad is still rejected: no graph.
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0])


class CumsumBackwardTests(unittest.TestCase):
    def test_inclusive_jacobian_columns_summed(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.cumsum().backward([1.0, 1.0, 1.0])
        # J = [[1,0,0],[1,1,0],[1,1,1]]
        self.assertEqual(x.grad, [3.0, 2.0, 1.0])

    def test_exclusive_jacobian_columns_summed(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.cumsum(exclusive=True).backward([1.0, 1.0, 1.0])
        # J = [[0,0,0],[1,0,0],[1,1,0]]
        self.assertEqual(x.grad, [2.0, 1.0, 0.0])

    def test_reverse_inclusive_jacobian_columns_summed(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.cumsum(reverse=True).backward([1.0, 1.0, 1.0])
        # J = [[1,1,1],[0,1,1],[0,0,1]]
        self.assertEqual(x.grad, [1.0, 2.0, 3.0])

    def test_reverse_exclusive_jacobian_columns_summed(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.cumsum(exclusive=True, reverse=True).backward([1.0, 1.0, 1.0])
        # J = [[0,1,1],[0,0,1],[0,0,0]]
        self.assertEqual(x.grad, [0.0, 1.0, 2.0])

    def test_explicit_upstream_grad_selects_rows(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x.cumsum().backward([0.0, 0.0, 2.0])
        self.assertEqual(x.grad, [2.0, 2.0, 2.0])
        y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y.cumsum(reverse=True).backward([2.0, 0.0, 0.0])
        # y[0] = x0 + x1 + x2, so its grad reaches every element.
        self.assertEqual(y.grad, [2.0, 2.0, 2.0])

    def test_single_element(self):
        x = Tensor([3.0], requires_grad=True)
        x.cumsum().backward([2.0])
        self.assertEqual(x.grad, [2.0])
        y = Tensor([3.0], requires_grad=True)
        y.cumsum(exclusive=True).backward([2.0])
        self.assertEqual(y.grad, [0.0])

    def test_grad_value_errors(self):
        out = Tensor([1.0, 2.0, 3.0], requires_grad=True).cumsum()
        # Omitted grad and any scalar (bool/int/float) are ValueError.
        with self.assertRaises(ValueError):
            out.backward()
        for bad in (True, 1, 1.0):
            with self.assertRaises(ValueError):
                out.backward(bad)
        # Wrong length and non-finite lists are ValueError.
        with self.assertRaises(ValueError):
            out.backward([1.0, 2.0])
        with self.assertRaises(ValueError):
            out.backward([1.0, 2.0, 3.0, 4.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                out.backward([1.0, 2.0, bad])

    def test_grad_type_errors(self):
        out = Tensor([1.0, 2.0, 3.0], requires_grad=True).cumsum()
        # None and any non-list container are TypeError.
        for bad in (None, "x", (1.0, 2.0, 3.0), {"a": 1.0}):
            with self.assertRaises(TypeError):
                out.backward(bad)
        # A list with a non-float element (bool/int included) is TypeError.
        for bad in ([1, 2.0, 3.0], [True, 2.0, 3.0], ["x", 2.0, 3.0]):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_nonfinite_partial_sum_is_value_error(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(ValueError):
            x.cumsum().backward([1e308, 1e308])
        self.assertIsNone(x.grad)

    def test_failed_backward_leaves_all_grads_untouched(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        z = Tensor(1.0, requires_grad=True)
        out = x.cumsum().mul(z)
        with self.assertRaises(ValueError):
            out.backward([1e308, 1e308])
        self.assertIsNone(x.grad)
        self.assertIsNone(z.grad)

    def test_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = x.cumsum()
        x.data[0] = 100.0
        out.backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [3.0, 2.0, 1.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        out = x.cumsum()
        out.backward([1.0, 1.0])
        out.backward([1.0, 1.0])
        # J = [[1,0],[1,1]]; each pass contributes [2,1].
        self.assertEqual(x.grad, [4.0, 2.0])

    def test_shared_path_accumulates(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        out = x.cumsum().add(x.cumsum())
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
