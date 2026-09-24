"""Tests for Tensor.cumprod: differentiable 1D prefix product."""

import math
import unittest

from autograd import Tensor


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation.

    Lets cumprod() itself be exercised against malformed data or a
    non-bool requires_grad flag, which the ordinary constructor rejects.
    """
    return Tensor._make(data, requires_grad, (), None)


class CumprodForwardTests(unittest.TestCase):
    def test_prefix_products(self):
        self.assertEqual(Tensor([2.0, 3.0, 4.0]).cumprod().data,
                         [2.0, 6.0, 24.0])

    def test_single_element(self):
        self.assertEqual(Tensor([7.0]).cumprod().data, [7.0])

    def test_zero_prefix_is_sticky(self):
        self.assertEqual(Tensor([2.0, 0.0, 5.0]).cumprod().data,
                         [2.0, 0.0, 0.0])

    def test_negative_elements(self):
        self.assertEqual(Tensor([-2.0, 3.0, -4.0]).cumprod().data,
                         [-2.0, -6.0, 24.0])

    def test_ascending_order_from_one(self):
        # Ascending order keeps the intermediate product finite; the
        # returned prefixes reflect exactly that order.
        values = [1e-308, 1e308, 1e308]
        out = Tensor(values).cumprod().data
        self.assertEqual(out[0], 1e-308)
        self.assertEqual(out[1], 1e-308 * 1e308)
        self.assertEqual(out[2], 9.999999999999998e307)

    def test_partial_product_overflow_is_value_error(self):
        with self.assertRaises(ValueError):
            Tensor([1e308, 10.0]).cumprod()

    def test_failed_forward_leaves_input_unchanged(self):
        x = Tensor([1e308, 10.0], requires_grad=True)
        with self.assertRaises(ValueError):
            x.cumprod()
        self.assertEqual(x.data, [1e308, 10.0])
        self.assertIsNone(x.grad)

    def test_non_list_and_empty_are_value_errors(self):
        for bad in (1.0, True, 1, "x", None, (1.0,)):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).cumprod()
        with self.assertRaises(ValueError):
            _raw_tensor([]).cumprod()

    def test_non_float_elements_are_type_errors(self):
        for bad in ([True], [1], [1.0, 2], [1.0, [2.0]], [[1.0]]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).cumprod()

    def test_non_finite_elements_are_value_errors(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor([1.0, bad]).cumprod()

    def test_non_bool_requires_grad_is_type_error(self):
        with self.assertRaises(TypeError):
            _raw_tensor([1.0, 2.0], requires_grad=1).cumprod()

    def test_no_grad_has_no_graph(self):
        out = Tensor([2.0, 3.0]).cumprod()
        self.assertFalse(out.requires_grad)
        # A correctly shaped finite grad is still rejected: no graph.
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0])


class CumprodBackwardTests(unittest.TestCase):
    def test_jacobian_columns_summed(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        x.cumprod().backward([1.0, 1.0, 1.0])
        # J = [[1,0,0],[3,2,0],[12,8,6]]
        self.assertEqual(x.grad, [16.0, 10.0, 6.0])

    def test_explicit_upstream_grad_selects_rows(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        x.cumprod().backward([0.0, 0.0, 1.0])
        self.assertEqual(x.grad, [12.0, 8.0, 6.0])

    def test_single_element(self):
        x = Tensor([3.0], requires_grad=True)
        x.cumprod().backward([2.0])
        self.assertEqual(x.grad, [2.0])

    def test_zero_element_differentiated_without_division(self):
        # One zero: dy1/dx1 = x0 and dy2/dx1 = x0*x2 stay nonzero,
        # even though y1 and y2 are both zero.
        x = Tensor([2.0, 0.0, 5.0], requires_grad=True)
        x.cumprod().backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 12.0, 0.0])

    def test_two_zero_elements(self):
        # y0 = x0 means dx0 always sees the dy0/dx0 = 1 contribution.
        x = Tensor([0.0, 0.0, 5.0], requires_grad=True)
        x.cumprod().backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0])

    def test_grad_value_errors(self):
        out = Tensor([2.0, 3.0, 4.0], requires_grad=True).cumprod()
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
        out = Tensor([2.0, 3.0, 4.0], requires_grad=True).cumprod()
        # None and any non-list container are TypeError.
        for bad in (None, "x", (1.0, 2.0, 3.0), {"a": 1.0}):
            with self.assertRaises(TypeError):
                out.backward(bad)
        # A list with a non-float element (bool/int included) is TypeError.
        for bad in ([1, 2.0, 3.0], [True, 2.0, 3.0], ["x", 2.0, 3.0]):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_nonfinite_cofactor_is_value_error(self):
        # Forward prefixes are finite, but the cofactor for (i=2, j=0)
        # is 1e308 * 1e308 and overflows.
        x = Tensor([1e-308, 1e308, 1e308], requires_grad=True)
        out = x.cumprod()
        with self.assertRaises(ValueError):
            out.backward([1.0, 0.0, 0.0])
        self.assertIsNone(x.grad)

    def test_nonfinite_term_is_value_error(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        with self.assertRaises(ValueError):
            x.cumprod().backward([0.0, 1e308])
        self.assertIsNone(x.grad)

    def test_failed_backward_leaves_all_grads_untouched(self):
        x = Tensor([1e-308, 1e308, 1e308], requires_grad=True)
        z = Tensor(1.0, requires_grad=True)
        out = x.cumprod().mul(z)
        with self.assertRaises(ValueError):
            out.backward([1.0, 0.0, 0.0])
        self.assertIsNone(x.grad)
        self.assertIsNone(z.grad)

    def test_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        out = x.cumprod()
        x.data[0] = 100.0
        out.backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [16.0, 10.0, 6.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        out = x.cumprod()
        out.backward([1.0, 1.0])
        out.backward([1.0, 1.0])
        # J = [[1,0],[3,2]]; each pass contributes [4,2].
        self.assertEqual(x.grad, [8.0, 4.0])

    def test_shared_path_accumulates(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        out = x.cumprod().add(x.cumprod())
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [8.0, 4.0])


if __name__ == "__main__":
    unittest.main()
