"""Tests for Tensor.dot: differentiable 1D inner product."""

import math
import unittest

from autograd import Tensor, gradcheck


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation.

    Lets dot() itself be exercised against malformed data or a
    non-bool requires_grad flag, which the ordinary constructor rejects.
    """
    return Tensor._make(data, requires_grad, (), None)


class DotForwardTests(unittest.TestCase):
    def test_basic_value_and_scalar_type(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        out = a.dot(b)
        self.assertIsInstance(out.data, float)
        self.assertEqual(out.data, 32.0)

    def test_ascending_index_multiply_then_add_from_zero(self):
        # Left-associated accumulation from index 0 absorbs the leading 1.0
        # into 1e16 (ULP 2.0) before the cancelling term: result 0.0.
        # Descending order (or math.fsum) would yield 1.0.
        a = Tensor([1.0, 1e16, -1e16])
        b = Tensor([1.0, 1.0, 1.0])
        self.assertEqual(a.dot(b).data, 0.0)

    def test_first_product_nonfinite_is_value_error(self):
        a = Tensor([1e308])
        b = Tensor([10.0])
        with self.assertRaises(ValueError):
            a.dot(b)

    def test_partial_sum_overflow_is_value_error(self):
        # Every individual product is finite, but the running sum overflows.
        a = Tensor([1e308] * 20)
        b = Tensor([1.0] * 20)
        with self.assertRaises(ValueError):
            a.dot(b)

    def test_nonfinite_inputs_are_value_error(self):
        good = Tensor([1.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor([bad]).dot(good)
            with self.assertRaises(ValueError):
                good.dot(_raw_tensor([bad]))

    def test_forward_failure_does_not_mutate_inputs(self):
        a_data = [1.0, 2.0]
        b_data = [1.0, float("nan")]
        a = Tensor(a_data)
        b = _raw_tensor(b_data)
        with self.assertRaises(ValueError):
            a.dot(b)
        self.assertEqual(a.data, [1.0, 2.0])
        self.assertIsNone(a.grad)
        self.assertEqual(b.data[0], 1.0)
        self.assertTrue(math.isnan(b.data[1]))
        self.assertEqual(len(b.data), 2)

    def test_requires_grad_flags_and_parents(self):
        neither = Tensor([1.0]).dot(Tensor([1.0]))
        self.assertFalse(neither.requires_grad)
        self.assertEqual(neither._parents, ())

        only_left = Tensor([1.0], True).dot(Tensor([1.0]))
        self.assertTrue(only_left.requires_grad)
        self.assertEqual(len(only_left._parents), 2)

        only_right = Tensor([1.0]).dot(Tensor([1.0], True))
        self.assertTrue(only_right.requires_grad)

        both = Tensor([1.0], True).dot(Tensor([1.0], True))
        self.assertTrue(both.requires_grad)


class DotValidationTests(unittest.TestCase):
    def test_other_must_be_tensor(self):
        a = Tensor([1.0])
        for bad in (1, 1.0, None, [1.0], (1.0,), "x", True, object()):
            with self.assertRaises(TypeError):
                a.dot(bad)

    def test_other_check_takes_precedence(self):
        # self is a malformed scalar, other is an int: the non-Tensor
        # other must raise TypeError before self's data is inspected.
        with self.assertRaises(TypeError):
            Tensor(1.0).dot(1)

    def test_data_must_be_nonempty_vector(self):
        good = Tensor([1.0])
        for bad in (1.0, [], (1.0,), None):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).dot(good)
            with self.assertRaises(ValueError):
                good.dot(_raw_tensor(bad))

    def test_elements_must_be_floats(self):
        good = Tensor([1.0, 2.0])
        for bad in ([1, 2], [1.0, True], [[1.0], 2.0], [None]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).dot(good)
            with self.assertRaises(TypeError):
                good.dot(_raw_tensor(bad))

    def test_requires_grad_must_be_bool(self):
        good = Tensor([1.0])
        for flag in (1, 0, "yes", None):
            with self.assertRaises(TypeError):
                _raw_tensor([1.0], flag).dot(good)
            with self.assertRaises(TypeError):
                good.dot(_raw_tensor([1.0], flag))

    def test_requires_grad_check_precedes_finite_check(self):
        # _require_nonempty_float_vector checks the requires_grad type
        # before element finiteness.
        good = Tensor([1.0])
        with self.assertRaises(TypeError):
            _raw_tensor([float("nan")], "yes").dot(good)

    def test_length_mismatch_is_value_error_no_broadcast(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([1.0])
        with self.assertRaises(ValueError):
            a.dot(b)
        with self.assertRaises(ValueError):
            b.dot(a)

    def test_backward_without_graph_raises(self):
        out = Tensor([1.0]).dot(Tensor([1.0]))
        with self.assertRaises(ValueError):
            out.backward()


class DotGradientTests(unittest.TestCase):
    def test_default_upstream_is_one(self):
        a = Tensor([1.0, 2.0, 3.0], True)
        b = Tensor([4.0, 5.0, 6.0], True)
        a.dot(b).backward()
        self.assertEqual(a.grad, [4.0, 5.0, 6.0])
        self.assertEqual(b.grad, [1.0, 2.0, 3.0])

    def test_explicit_upstream_scales_gradients(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        a.dot(b).backward(2.5)
        self.assertEqual(a.grad, [7.5, 10.0])
        self.assertEqual(b.grad, [2.5, 5.0])

    def test_only_requiring_parent_receives_grad(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], False)
        out = a.dot(b)
        out.backward()
        self.assertEqual(a.grad, [3.0, 4.0])
        self.assertIsNone(b.grad)

        c = Tensor([1.0, 2.0], False)
        d = Tensor([3.0, 4.0], True)
        out2 = c.dot(d)
        out2.backward()
        self.assertIsNone(c.grad)
        self.assertEqual(d.grad, [1.0, 2.0])

    def test_explicit_upstream_validation(self):
        out = Tensor([1.0], True).dot(Tensor([1.0], True))
        for bad in (None, True, False, 1, 0, "1.0", object()):
            with self.assertRaises(TypeError):
                out.backward(bad)
        for bad in ([1.0], float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                out.backward(bad)

    def test_alias_same_tensor_merges_contributions(self):
        v = Tensor([1.0, 2.0, 3.0], True)
        v.dot(v).backward()
        self.assertEqual(v.grad, [2.0, 4.0, 6.0])

        w = Tensor([1.0, 2.0], True)
        w.dot(w).backward(2.0)
        self.assertEqual(w.grad, [4.0, 8.0])

    def test_repeated_backward_accumulates(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        out = a.dot(b)
        out.backward()
        out.backward(2.0)
        self.assertEqual(a.grad, [9.0, 12.0])   # 3*b
        self.assertEqual(b.grad, [3.0, 6.0])    # 3*a

    def test_shared_leaf_accumulates_across_dots(self):
        a = Tensor([1.0, 2.0], True)
        b1 = Tensor([3.0, 4.0], True)
        b2 = Tensor([5.0, 6.0], True)
        a.dot(b1).add(a.dot(b2)).backward()
        self.assertEqual(a.grad, [8.0, 10.0])   # b1 + b2
        self.assertEqual(b1.grad, [1.0, 2.0])
        self.assertEqual(b2.grad, [1.0, 2.0])

    def test_diamond_same_dot_result_used_twice(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        c = a.dot(b)
        c.add(c).backward()
        self.assertEqual(a.grad, [6.0, 8.0])    # 2*b
        self.assertEqual(b.grad, [2.0, 4.0])    # 2*a

    def test_numerical_gradient_left(self):
        fixed = Tensor([0.5, -1.25, 2.0])
        passed, error = gradcheck(
            lambda v: v.dot(fixed), [0.3, -0.7, 1.25], atol=1e-6
        )
        self.assertTrue(passed, msg=f"max abs error {error}")

    def test_numerical_gradient_right(self):
        fixed = Tensor([0.5, -1.25, 2.0], True)

        def fn(v):
            # The differentiable input is the right operand.
            return fixed.dot(v)

        passed, error = gradcheck(fn, [0.3, -0.7, 1.25], atol=1e-6)
        self.assertTrue(passed, msg=f"max abs error {error}")

    def test_numerical_gradient_alias(self):
        passed, error = gradcheck(
            lambda v: v.dot(v), [0.3, -0.7, 1.25], atol=1e-5
        )
        self.assertTrue(passed, msg=f"max abs error {error}")


class DotSnapshotTests(unittest.TestCase):
    def test_mutating_input_lists_after_forward_is_isolated(self):
        a_data = [1.0, 2.0, 3.0]
        b_data = [4.0, 5.0, 6.0]
        a = Tensor(a_data, True)
        b = Tensor(b_data, True)
        out = a.dot(b)
        a_data[0] = 100.0
        b_data[0] = 200.0
        b_data.append(9.0)
        out.backward()
        self.assertEqual(a.grad, [4.0, 5.0, 6.0])
        self.assertEqual(b.grad, [1.0, 2.0, 3.0])

    def test_replacing_input_data_after_forward_is_isolated(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([3.0, 4.0], True)
        out = a.dot(b)
        a.data = [9.0, 9.0]
        b.data = [9.0, 9.0, 9.0]
        out.backward()
        self.assertEqual(a.grad, [3.0, 4.0])
        self.assertEqual(b.grad, [1.0, 2.0])

    def test_snapshot_with_alias_and_mutation(self):
        data = [1.0, 2.0, 3.0]
        v = Tensor(data, True)
        out = v.dot(v)
        data[0] = 999.0
        data.append(4.0)
        out.backward()
        self.assertEqual(v.grad, [2.0, 4.0, 6.0])

    def test_repeated_backward_uses_same_snapshot(self):
        data_a = [1.0, 2.0]
        data_b = [3.0, 4.0]
        a = Tensor(data_a, True)
        b = Tensor(data_b, True)
        out = a.dot(b)
        out.backward()
        data_a[:] = [0.0, 0.0]
        data_b[:] = [0.0, 0.0]
        out.backward()
        self.assertEqual(a.grad, [6.0, 8.0])
        self.assertEqual(b.grad, [2.0, 4.0])


class DotBackwardAtomicityTests(unittest.TestCase):
    def test_nonfinite_contribution_leaves_prior_grads_untouched(self):
        a = Tensor([1.0, 2.0], True)
        b = Tensor([1e308, 1.0], True)
        out = a.dot(b)
        self.assertTrue(math.isfinite(out.data))
        out.backward(1.0)
        a_grad_before = list(a.grad)
        b_grad_before = list(b.grad)
        with self.assertRaises(ValueError):
            # 1e308 * 1e308 overflows in da[0].
            out.backward(1e308)
        self.assertEqual(a.grad, a_grad_before)
        self.assertEqual(b.grad, b_grad_before)

    def test_nonfinite_merge_leaves_prior_grads_untouched(self):
        a = Tensor([1.0, 1.0], True)
        b = Tensor([6e307, 1.0], True)
        out = a.dot(b)
        out.backward(1.0)
        out.backward(1.0)
        a_grad_before = list(a.grad)   # 1.2e308, still finite
        b_grad_before = list(b.grad)
        self.assertTrue(all(math.isfinite(v) for v in a_grad_before))
        with self.assertRaises(ValueError):
            # Merging a third 6e307 contribution overflows past 1.8e308.
            out.backward(1.0)
        self.assertEqual(a.grad, a_grad_before)
        self.assertEqual(b.grad, b_grad_before)

    def test_failure_in_one_branch_preserves_whole_graph(self):
        big = Tensor([1e308, 1.0], True)
        x = Tensor([1.0, 1.0], True)
        small = Tensor([0.1, 0.1], True)
        y = Tensor([0.1, 0.1], True)
        c1 = big.dot(x)     # non-finite only under a huge upstream g
        c2 = small.dot(y)   # stays finite for the same g
        z = c1.add(c2)
        big.grad = [5.0, 5.0]
        x.grad = [9.0, 9.0]
        small.grad = [7.0, 7.0]
        y.grad = [8.0, 8.0]
        with self.assertRaises(ValueError):
            # dx[0] = 1e308 * 1e308 overflows; the small branch is finite,
            # but the whole pass must abort before any grad is written.
            z.backward(1e308)
        self.assertEqual(big.grad, [5.0, 5.0])
        self.assertEqual(x.grad, [9.0, 9.0])
        self.assertEqual(small.grad, [7.0, 7.0])
        self.assertEqual(y.grad, [8.0, 8.0])
        self.assertIsNone(c1.grad)
        self.assertIsNone(c2.grad)
        self.assertIsNone(z.grad)


if __name__ == "__main__":
    unittest.main()
