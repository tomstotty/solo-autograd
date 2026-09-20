"""Tests for Tensor.dot: differentiable inner product of two 1D vectors."""

import math
import unittest

from autograd import Tensor, gradcheck


class TestDotForward(unittest.TestCase):
    def test_basic_value(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        result = a.dot(b)
        self.assertIsInstance(result.data, float)
        self.assertEqual(result.data, 32.0)

    def test_index_order_multiply_then_add_from_zero(self):
        # Ascending fold: ((0 + 1e16) + (-1e16)) + 1.0 == 1.0.
        # A descending (or reodered) accumulation rounds to 0.0 here, so
        # this pins the required ascending multiply-then-add semantics.
        a = Tensor([1e16, -1e16, 1.0])
        b = Tensor([1.0, 1.0, 1.0])
        self.assertEqual(a.dot(b).data, 1.0)

    def test_mixed_signs(self):
        a = Tensor([0.5, -2.0, 1.25])
        b = Tensor([4.0, 3.0, -8.0])
        # 0.5*4 + (-2)*3 + 1.25*(-8) = 2 - 6 - 10
        self.assertEqual(a.dot(b).data, -14.0)

    def test_no_graph_without_grad(self):
        result = Tensor([1.0, 2.0]).dot(Tensor([3.0, 4.0]))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result._parents, ())
        self.assertIsNone(result._backward_fn)

    def test_graph_when_either_side_needs_grad(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], False)
        result = a.dot(b)
        self.assertTrue(result.requires_grad)
        self.assertEqual(result._parents, (a, b))


class TestDotSnapshot(unittest.TestCase):
    def test_backward_uses_snapshots(self):
        a = Tensor([1.0, 2.0, 3.0], True)
        b = Tensor([4.0, 5.0, 6.0], True)
        result = a.dot(b)
        # Mutate the caller's lists after the forward pass; the pending
        # backward must multiply against the frozen snapshots.
        a.data[0] = 100.0
        b.data[:] = [400.0, 500.0, 600.0]
        result.backward()
        self.assertEqual(a.grad, [4.0, 5.0, 6.0])
        self.assertEqual(b.grad, [1.0, 2.0, 3.0])
        self.assertEqual(result.data, 32.0)

    def test_snapshot_is_a_copy(self):
        a_data = [1.0, 2.0]
        b_data = [3.0, 4.0]
        a = Tensor(a_data, True)
        b = Tensor(b_data, True)
        result = a.dot(b)
        a_data.append(99.0)
        b_data.append(99.0)
        result.backward()
        self.assertEqual(a.grad, [3.0, 4.0])
        self.assertEqual(b.grad, [1.0, 2.0])


class TestDotBackward(unittest.TestCase):
    def test_gradients(self):
        a = Tensor([1.0, 2.0, 3.0], True)
        b = Tensor([4.0, 5.0, 6.0], True)
        a.dot(b).backward()
        self.assertEqual(a.grad, [4.0, 5.0, 6.0])
        self.assertEqual(b.grad, [1.0, 2.0, 3.0])

    def test_explicit_upstream_grad(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        a.dot(b).backward(2.5)
        self.assertEqual(a.grad, [7.5, 10.0])
        self.assertEqual(b.grad, [2.5, 5.0])

    def test_only_needed_side_gets_grad(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], False)
        a.dot(b).backward()
        self.assertEqual(a.grad, [3.0, 4.0])
        self.assertIsNone(b.grad)

    def test_alias_same_tensor_merges_contributions(self):
        x = Tensor([1.0, 2.0, 3.0], True)
        result = x.dot(x)
        self.assertEqual(result.data, 14.0)
        result.backward()
        self.assertEqual(x.grad, [2.0, 4.0, 6.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0], True)
        result = x.dot(x)
        result.backward()
        result.backward(2.0)
        self.assertEqual(x.grad, [6.0, 12.0, 18.0])

    def test_shared_path_accumulates(self):
        x = Tensor([1.0, 2.0], True)
        y1 = Tensor([3.0, 4.0], True)
        y2 = Tensor([5.0, 6.0], True)
        loss = x.dot(y1).add(x.dot(y2))
        loss.backward()
        self.assertEqual(x.grad, [8.0, 10.0])
        self.assertEqual(y1.grad, [1.0, 2.0])
        self.assertEqual(y2.grad, [1.0, 2.0])

    def test_composed_with_mul(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        a.dot(b).mul(3.0).backward()
        self.assertEqual(a.grad, [9.0, 12.0])
        self.assertEqual(b.grad, [3.0, 6.0])

    def test_numerical_gradients(self):
        passed, error = gradcheck(
            lambda t: t.dot(Tensor([0.5, -1.5, 2.0])),
            [0.3, -0.7, 1.2],
        )
        self.assertTrue(passed, error)

    def test_numerical_gradients_alias(self):
        passed, error = gradcheck(
            lambda t: t.dot(t), [0.3, -0.7, 1.2]
        )
        self.assertTrue(passed, error)


class TestDotExceptions(unittest.TestCase):
    def test_other_must_be_tensor(self):
        a = Tensor([1.0])
        for bad in (1.0, 1, None, [1.0], "x", True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    a.dot(bad)

    def test_data_must_be_nonempty_list(self):
        vec = Tensor([1.0])
        with self.assertRaises(ValueError):
            vec.dot(Tensor(1.0))
        with self.assertRaises(ValueError):
            Tensor(1.0).dot(vec)
        with self.assertRaises(ValueError):
            vec.dot(Tensor([]))
        with self.assertRaises(ValueError):
            Tensor([]).dot(vec)

    def test_elements_must_be_floats(self):
        valid = Tensor([1.0, 2.0], True)
        for bad_elements in (
            [1, 2],
            [True, False],
            [[1.0], [2.0]],
            [1.0, 2],
        ):
            with self.subTest(bad_elements=bad_elements):
                with self.assertRaises(TypeError):
                    valid.dot(Tensor(bad_elements))
                with self.assertRaises(TypeError):
                    Tensor(bad_elements).dot(valid)

    def test_requires_grad_must_be_bool(self):
        a = Tensor([1.0], True)
        b = Tensor([1.0], True)
        b.requires_grad = "yes"
        with self.assertRaises(TypeError):
            a.dot(b)
        a.requires_grad = 1
        with self.assertRaises(TypeError):
            a.dot(Tensor([1.0], True))

    def test_elements_must_be_finite(self):
        valid = Tensor([1.0])
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    valid.dot(Tensor([value]))
                with self.assertRaises(ValueError):
                    Tensor([value]).dot(valid)

    def test_length_mismatch_no_broadcasting(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0]).dot(Tensor([1.0, 2.0, 3.0]))

    def test_nonfinite_product_aborts(self):
        with self.assertRaises(ValueError):
            Tensor([1e200]).dot(Tensor([1e200]))

    def test_nonfinite_partial_sum_aborts(self):
        with self.assertRaises(ValueError):
            Tensor([1e308, 1e308]).dot(Tensor([1.0, 1.0]))

    def test_forward_failure_leaves_inputs_untouched(self):
        a_data = [1e308, 1e308]
        b_data = [1.0, 1.0]
        a = Tensor(a_data, True)
        b = Tensor(b_data, True)
        with self.assertRaises(ValueError):
            a.dot(b)
        self.assertEqual(a.data, a_data)
        self.assertEqual(b.data, b_data)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)


class TestDotBackwardExceptions(unittest.TestCase):
    def _graph(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        return a, b, a.dot(b)

    def test_explicit_grad_types(self):
        _, _, result = self._graph()
        for bad in (None, True, 1, "x", object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    result.backward(bad)

    def test_explicit_grad_list_is_value_error(self):
        _, _, result = self._graph()
        with self.assertRaises(ValueError):
            result.backward([1.0])

    def test_explicit_grad_nonfinite_is_value_error(self):
        _, _, result = self._graph()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    result.backward(bad)

    def test_backward_failure_is_atomic(self):
        x = Tensor([1e308, 1.0], True)
        y = Tensor([1.0, 1.0], True)
        x.grad = [3.0, 3.0]
        with self.assertRaises(ValueError):
            x.dot(y).backward(1e308)
        # Every grad in the graph keeps its pre-call value.
        self.assertEqual(x.grad, [3.0, 3.0])
        self.assertIsNone(y.grad)

    def test_backward_merge_failure_is_atomic(self):
        # Each branch contribution (1e308) is finite, but merging the two
        # shared-path contributions overflows; nothing may be committed.
        x = Tensor([1.0, 1.0], True)
        y = Tensor([1.0, 1.0], True)
        x.grad = [2.0, 2.0]
        loss = x.dot(y).add(x.dot(y))
        with self.assertRaises(ValueError):
            loss.backward(1e308)
        self.assertEqual(x.grad, [2.0, 2.0])
        self.assertIsNone(y.grad)

    def test_backward_after_failure_can_still_accumulate(self):
        x = Tensor([1e308, 1.0], True)
        y = Tensor([1.0, 1.0], True)
        result = x.dot(y)
        with self.assertRaises(ValueError):
            result.backward(1e308)
        self.assertIsNone(x.grad)
        result.backward(1.0)
        self.assertEqual(x.grad, [1.0, 1.0])
        self.assertEqual(y.grad, [1e308, 1.0])


if __name__ == "__main__":
    unittest.main()
