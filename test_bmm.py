"""Tests for Tensor.bmm backward grad validation.

bmm always produces a vector (batch * rows * cols), so its backward
grad must be a same-length list of finite floats: an omitted grad or a
bool/int/float scalar is a ValueError, None and other non-list types are
TypeErrors, wrong-length / non-finite lists are ValueErrors, and a list
with a non-float element is a TypeError.
"""

import unittest

from autograd import Tensor


def _bmm(requires_grad_a=True, requires_grad_b=True):
    a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=requires_grad_a)
    b = Tensor([1.0, 1.0, 1.0, 1.0], requires_grad=requires_grad_b)
    return a, b, a.bmm(b, 1, 2, 2, 2)


class BmmBackwardGradValidationTests(unittest.TestCase):
    def test_scalars_and_omitted_are_value_errors(self):
        _, _, out = _bmm()
        with self.assertRaises(ValueError):
            out.backward()
        for bad in (True, 1, 1.0):
            with self.assertRaises(ValueError):
                out.backward(bad)

    def test_none_and_other_non_lists_are_type_errors(self):
        _, _, out = _bmm()
        for bad in (None, "x", (1.0, 1.0, 1.0, 1.0), {0: 1.0}):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_wrong_length_and_non_finite_are_value_errors(self):
        _, _, out = _bmm()
        with self.assertRaises(ValueError):
            out.backward([1.0])
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0, 1.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                out.backward([1.0, 1.0, 1.0, bad])

    def test_non_float_list_elements_are_type_errors(self):
        _, _, out = _bmm()
        for bad in (
            [1, 1.0, 1.0, 1.0],
            [True, 1.0, 1.0, 1.0],
            ["x", 1.0, 1.0, 1.0],
            [None, 1.0, 1.0, 1.0],
        ):
            with self.assertRaises(TypeError):
                out.backward(bad)

    def test_graphless_result_rejects_any_grad(self):
        # Without any requiring parent the result has no graph; the
        # missing-graph ValueError takes precedence over grad validation,
        # so every grad — valid or not — raises ValueError.
        _, _, out = _bmm(False, False)
        self.assertFalse(out.requires_grad)
        for bad in (None, "x", True, 1, 1.0):
            with self.assertRaises(ValueError):
                out.backward(bad)
        with self.assertRaises(ValueError):
            out.backward()
        with self.assertRaises(ValueError):
            out.backward([1.0, 1.0, 1.0, 1.0])

    def test_invalid_grad_changes_no_grad(self):
        a, b, out = _bmm()
        with self.assertRaises(ValueError):
            out.backward(1.0)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        with self.assertRaises(TypeError):
            out.backward(None)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        # A valid pass still works after the failed ones.
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(a.grad, [2.0, 2.0, 2.0, 2.0])
        self.assertEqual(b.grad, [4.0, 4.0, 6.0, 6.0])

    def test_valid_batched_grad(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   requires_grad=True)
        b = Tensor([1.0] * 8, requires_grad=True)
        out = a.bmm(b, 2, 2, 2, 2)
        out.backward([1.0] * 8)
        self.assertEqual(len(a.grad), 8)
        self.assertEqual(len(b.grad), 8)


if __name__ == "__main__":
    unittest.main()
