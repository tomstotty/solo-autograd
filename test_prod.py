"""Tests for Tensor.prod: differentiable product reduction."""

import math
import unittest

from autograd import Tensor


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation.

    Lets prod() itself be exercised against malformed data or a
    non-bool requires_grad flag, which the ordinary constructor rejects.
    """
    return Tensor._make(data, requires_grad, (), None)


class ProdForwardTests(unittest.TestCase):
    def test_scalar_returns_scalar(self):
        out = Tensor(3.5).prod()
        self.assertIsInstance(out.data, float)
        self.assertEqual(out.data, 3.5)

    def test_vector_product(self):
        out = Tensor([2.0, 3.0, 4.0]).prod()
        self.assertIsInstance(out.data, float)
        self.assertEqual(out.data, 24.0)

    def test_single_element_vector(self):
        self.assertEqual(Tensor([7.0]).prod().data, 7.0)

    def test_zero_element_gives_zero(self):
        self.assertEqual(Tensor([2.0, 0.0, 5.0]).prod().data, 0.0)

    def test_ascending_index_multiply_from_one(self):
        # Ascending: 1e-308 * 1e308 = 1.0, then * 1e308.
        # Descending order would overflow at 1e308 * 1e308.
        values = [1e-308, 1e308, 1e308]
        expected = 1.0
        for v in values:
            expected *= v
        out = Tensor(values).prod()
        self.assertEqual(out.data, expected)
        self.assertEqual(out.data, 9.999999999999998e307)

    def test_partial_product_overflow_is_value_error(self):
        with self.assertRaises(ValueError):
            Tensor([1e308, 10.0]).prod()

    def test_failed_forward_leaves_input_unchanged(self):
        x = Tensor([1e308, 10.0], requires_grad=True)
        with self.assertRaises(ValueError):
            x.prod()
        self.assertEqual(x.data, [1e308, 10.0])
        self.assertIsNone(x.grad)

    def test_forward_type_errors(self):
        for bad in (True, 1, "x", None, (1.0,), [[1.0]], [1.0, 2]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).prod()
        for bad in ([1.0, True], [1.0, 2], [1.0, [2.0]]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).prod()

    def test_forward_value_errors(self):
        with self.assertRaises(ValueError):
            _raw_tensor([]).prod()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).prod()
            with self.assertRaises(ValueError):
                _raw_tensor([1.0, bad]).prod()

    def test_non_bool_requires_grad_is_type_error(self):
        with self.assertRaises(TypeError):
            _raw_tensor([1.0, 2.0], requires_grad=1).prod()

    def test_no_grad_has_no_graph(self):
        out = Tensor([2.0, 3.0]).prod()
        self.assertFalse(out.requires_grad)
        with self.assertRaises(ValueError):
            out.backward()


class ProdBackwardTests(unittest.TestCase):
    def test_scalar_backward_default_grad(self):
        x = Tensor(3.5, requires_grad=True)
        x.prod().backward()
        self.assertEqual(x.grad, 1.0)

    def test_scalar_backward_explicit_grad(self):
        x = Tensor(3.5, requires_grad=True)
        x.prod().backward(2.0)
        self.assertEqual(x.grad, 2.0)

    def test_vector_backward(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        x.prod().backward()
        self.assertEqual(x.grad, [12.0, 8.0, 6.0])

    def test_vector_backward_with_upstream(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        x.prod().backward(0.5)
        self.assertEqual(x.grad, [6.0, 4.0, 3.0])

    def test_single_element_vector_uses_empty_product(self):
        x = Tensor([7.0], requires_grad=True)
        x.prod().backward()
        self.assertEqual(x.grad, [1.0])

    def test_zero_element_gradients(self):
        # One zero: only the zero element sees a nonzero cofactor.
        x = Tensor([2.0, 0.0, 5.0], requires_grad=True)
        x.prod().backward()
        self.assertEqual(x.grad, [0.0, 10.0, 0.0])
        # Two zeros: every cofactor contains a zero.
        y = Tensor([0.0, 0.0, 5.0], requires_grad=True)
        y.prod().backward()
        self.assertEqual(y.grad, [0.0, 0.0, 0.0])

    def test_backward_grad_type_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).prod()
        for bad in (None, True, 1, "x"):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_backward_grad_value_errors(self):
        out = Tensor([2.0, 3.0], requires_grad=True).prod()
        with self.assertRaises(ValueError):
            out.backward([1.0])
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                out.backward(bad)

    def test_nonfinite_cofactor_is_value_error(self):
        # Forward product is finite (1e308), but the cofactor for
        # element 0 overflows: 1e308 * 1e308.
        x = Tensor([1e-308, 1e308, 1e308], requires_grad=True)
        out = x.prod()
        with self.assertRaises(ValueError):
            out.backward()
        self.assertIsNone(x.grad)

    def test_nonfinite_contribution_is_value_error(self):
        x = Tensor([1e308, 1.0], requires_grad=True)
        out = x.prod()
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertIsNone(x.grad)

    def test_failed_backward_leaves_all_grads_untouched(self):
        x = Tensor([1e-308, 1e308, 1e308], requires_grad=True)
        y = Tensor(1.0, requires_grad=True)
        out = x.prod().mul(y)
        with self.assertRaises(ValueError):
            out.backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        out = x.prod()
        x.data[0] = 100.0
        out.backward()
        self.assertEqual(x.grad, [12.0, 8.0, 6.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        out = x.prod()
        out.backward()
        out.backward()
        self.assertEqual(x.grad, [6.0, 4.0])

    def test_shared_path_accumulates(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        out = x.prod().add(x.prod())
        out.backward()
        self.assertEqual(x.grad, [6.0, 4.0])


if __name__ == "__main__":
    unittest.main()
