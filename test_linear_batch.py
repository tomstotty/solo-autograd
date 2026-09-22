"""Tests for Tensor.linear_batch: batched differentiable affine map."""

import math
import unittest

from autograd import Tensor, gradcheck


def _raw_tensor(data, requires_grad=True):
    """Build a Tensor bypassing the constructor's own validation.

    Lets linear_batch() itself be exercised against malformed data or a
    non-bool requires_grad flag, which the ordinary constructor rejects.
    """
    return Tensor._make(data, requires_grad, (), None)


def _sample():
    # B = 2, n = 2, m = 2.
    x = Tensor([1.0, 2.0, 3.0, 4.0], True)
    w = Tensor([0.5, -1.0, 2.0, 0.25], True)
    b = Tensor([0.1, -0.2], True)
    return x, w, b


class LinearBatchForwardTests(unittest.TestCase):
    def test_basic_value(self):
        x, w, b = _sample()
        out = x.linear_batch(w, b, 2)
        self.assertIsInstance(out.data, list)
        self.assertEqual(out.data, [-1.4, 2.3, -2.4, 6.8])

    def test_batch_size_one_matches_linear(self):
        x = Tensor([1.0, 2.0])
        w = Tensor([0.5, -1.0, 2.0, 0.25])
        b = Tensor([0.1, -0.2])
        self.assertEqual(
            x.linear_batch(w, b, 1).data, x.linear(w, b).data
        )

    def test_accumulation_starts_from_bias(self):
        # Accumulating from b[0] absorbs the bias into 1e16 before the
        # cancelling term: result 0.0. Accumulating from 0.0 and adding
        # the bias at the end would yield 1.0.
        x = Tensor([1e16, -1e16])
        w = Tensor([1.0, 1.0])
        b = Tensor([1.0])
        self.assertEqual(x.linear_batch(w, b, 1).data, [0.0])

    def test_first_product_nonfinite_is_value_error(self):
        x = Tensor([1e308])
        w = Tensor([10.0])
        b = Tensor([0.0])
        with self.assertRaises(ValueError):
            x.linear_batch(w, b, 1)

    def test_partial_sum_overflow_is_value_error(self):
        x = Tensor([1e308, 1e308])
        w = Tensor([1.0, 1.0])
        b = Tensor([0.0])
        with self.assertRaises(ValueError):
            x.linear_batch(w, b, 1)

    def test_nonfinite_inputs_are_value_error(self):
        good = Tensor([1.0])
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _raw_tensor([bad]).linear_batch(good, good, 1)
            with self.assertRaises(ValueError):
                good.linear_batch(_raw_tensor([bad]), good, 1)
            with self.assertRaises(ValueError):
                good.linear_batch(good, _raw_tensor([bad]), 1)

    def test_forward_failure_does_not_mutate_inputs(self):
        x = Tensor([1.0, 2.0])
        w = Tensor([1e308, 1e308])
        b = Tensor([0.0])
        with self.assertRaises(ValueError):
            x.linear_batch(w, b, 1)
        self.assertEqual(x.data, [1.0, 2.0])
        self.assertEqual(w.data, [1e308, 1e308])
        self.assertEqual(b.data, [0.0])
        self.assertIsNone(x.grad)
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)

    def test_requires_grad_flags_and_parents(self):
        neither = Tensor([1.0]).linear_batch(
            Tensor([1.0]), Tensor([1.0]), 1
        )
        self.assertFalse(neither.requires_grad)
        self.assertEqual(neither._parents, ())

        only_x = Tensor([1.0], True).linear_batch(
            Tensor([1.0]), Tensor([1.0]), 1
        )
        self.assertTrue(only_x.requires_grad)
        self.assertEqual(len(only_x._parents), 3)

        only_w = Tensor([1.0]).linear_batch(
            Tensor([1.0], True), Tensor([1.0]), 1
        )
        self.assertTrue(only_w.requires_grad)

        only_b = Tensor([1.0]).linear_batch(
            Tensor([1.0]), Tensor([1.0], True), 1
        )
        self.assertTrue(only_b.requires_grad)


class LinearBatchValidationTests(unittest.TestCase):
    def test_weight_and_bias_must_be_tensors(self):
        x = Tensor([1.0])
        w = Tensor([1.0])
        b = Tensor([1.0])
        for bad in (1, 1.0, None, [1.0], (1.0,), "x", True, object()):
            with self.assertRaises(TypeError):
                x.linear_batch(bad, b, 1)
            with self.assertRaises(TypeError):
                x.linear_batch(w, bad, 1)

    def test_operand_check_takes_precedence(self):
        # The non-Tensor weight must raise TypeError before any data or
        # batch_size inspection.
        with self.assertRaises(TypeError):
            Tensor(1.0).linear_batch(1, Tensor([1.0]), "bad")
        with self.assertRaises(TypeError):
            Tensor(1.0).linear_batch(Tensor([1.0]), 1, "bad")

    def test_data_must_be_nonempty_vector(self):
        good = Tensor([1.0])
        for bad in (1.0, [], (1.0,), None):
            with self.assertRaises(ValueError):
                _raw_tensor(bad).linear_batch(good, good, 1)
            with self.assertRaises(ValueError):
                good.linear_batch(_raw_tensor(bad), good, 1)
            with self.assertRaises(ValueError):
                good.linear_batch(good, _raw_tensor(bad), 1)

    def test_elements_must_be_floats(self):
        good = Tensor([1.0, 2.0])
        for bad in ([1, 2], [1.0, True], [[1.0], 2.0], [None]):
            with self.assertRaises(TypeError):
                _raw_tensor(bad).linear_batch(good, good, 1)
            with self.assertRaises(TypeError):
                good.linear_batch(_raw_tensor(bad), good, 1)
            with self.assertRaises(TypeError):
                good.linear_batch(good, _raw_tensor(bad), 1)

    def test_requires_grad_must_be_bool(self):
        good = Tensor([1.0])
        for flag in (1, 0, "yes", None):
            with self.assertRaises(TypeError):
                _raw_tensor([1.0], flag).linear_batch(good, good, 1)
            with self.assertRaises(TypeError):
                good.linear_batch(_raw_tensor([1.0], flag), good, 1)
            with self.assertRaises(TypeError):
                good.linear_batch(good, _raw_tensor([1.0], flag), 1)

    def test_batch_size_must_be_nonbool_int(self):
        x = Tensor([1.0, 2.0])
        w = Tensor([1.0, 1.0])
        b = Tensor([0.0])
        for bad in (True, False, 1.0, 2.5, "2", None, [2], object()):
            with self.assertRaises(TypeError):
                x.linear_batch(w, b, bad)

    def test_batch_size_must_be_positive(self):
        x = Tensor([1.0, 2.0])
        w = Tensor([1.0, 1.0])
        b = Tensor([0.0])
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError):
                x.linear_batch(w, b, bad)

    def test_batch_size_must_divide_input_length(self):
        x = Tensor([1.0, 2.0, 3.0])
        w = Tensor([1.0])
        b = Tensor([0.0])
        with self.assertRaises(ValueError):
            x.linear_batch(w, b, 2)
        with self.assertRaises(ValueError):
            x.linear_batch(w, b, 4)

    def test_weight_length_must_equal_m_times_n(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        b = Tensor([0.0, 0.0])
        for w_data in ([1.0], [1.0] * 3, [1.0] * 5):
            with self.assertRaises(ValueError):
                x.linear_batch(Tensor(w_data), b, 2)

    def test_backward_without_graph_raises(self):
        out = Tensor([1.0]).linear_batch(Tensor([1.0]), Tensor([1.0]), 1)
        with self.assertRaises(ValueError):
            out.backward()


class LinearBatchGradientTests(unittest.TestCase):
    def test_explicit_upstream_gradients(self):
        x, w, b = _sample()
        x.linear_batch(w, b, 2).backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [4.5, -0.5, 9.5, -2.0])
        self.assertEqual(w.grad, [10.0, 14.0, 14.0, 20.0])
        self.assertEqual(b.grad, [4.0, 6.0])

    def test_only_requiring_parents_receive_grad(self):
        x = Tensor([1.0, 2.0], True)
        w = Tensor([3.0, 4.0], False)
        b = Tensor([0.5], False)
        out = x.linear_batch(w, b, 1)
        out.backward([1.0])
        self.assertEqual(x.grad, [3.0, 4.0])
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)

        x2 = Tensor([1.0, 2.0], False)
        w2 = Tensor([3.0, 4.0], True)
        b2 = Tensor([0.5], True)
        x2.linear_batch(w2, b2, 1).backward([2.0])
        self.assertIsNone(x2.grad)
        self.assertEqual(w2.grad, [2.0, 4.0])
        self.assertEqual(b2.grad, [2.0])

    def test_grad_validation(self):
        x, w, b = _sample()
        out = x.linear_batch(w, b, 2)
        # Omitted grad and a bare float are ValueError for a vector output.
        with self.assertRaises(ValueError):
            out.backward()
        with self.assertRaises(ValueError):
            out.backward(1.0)
        # Wrong-length lists are ValueError.
        for bad in ([1.0, 2.0], [1.0] * 5, []):
            with self.assertRaises(ValueError):
                out.backward(bad)
        # Non-list, non-float grads are TypeError.
        for bad in (None, True, 1, "1.0", (1.0,) * 4, object()):
            with self.assertRaises(TypeError):
                out.backward(bad)
        # Non-float elements are TypeError.
        for bad in ([1, 1.0, 1.0, 1.0], [1.0, True, 1.0, 1.0]):
            with self.assertRaises(TypeError):
                out.backward(bad)
        # Non-finite elements are ValueError.
        for bad in (
            [float("nan"), 1.0, 1.0, 1.0],
            [1.0, float("inf"), 1.0, 1.0],
            [1.0, 1.0, 1.0, float("-inf")],
        ):
            with self.assertRaises(ValueError):
                out.backward(bad)
        self.assertIsNone(x.grad)
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)

    def test_failed_grad_leaves_graph_untouched(self):
        x, w, b = _sample()
        out = x.linear_batch(w, b, 2)
        out.backward([1.0, 1.0, 1.0, 1.0])
        grads_before = (list(x.grad), list(w.grad), list(b.grad))
        with self.assertRaises(ValueError):
            out.backward([1.0, float("nan"), 1.0, 1.0])
        self.assertEqual(x.grad, grads_before[0])
        self.assertEqual(w.grad, grads_before[1])
        self.assertEqual(b.grad, grads_before[2])

    def test_alias_all_three_roles_merge(self):
        # B = 1, n = 1, m = 1: out = v + v*v, d/dv = 1 + 2v.
        v = Tensor([2.0], True)
        v.linear_batch(v, v, 1).backward([1.0])
        self.assertEqual(v.grad, [5.0])

    def test_alias_input_and_weight_merge(self):
        # B = m = 2, n = 2 so len(x) == len(w); b stays separate.
        v = Tensor([1.0, 2.0, 3.0, 4.0], True)
        b = Tensor([0.5, -0.5], True)
        out = v.linear_batch(v, b, 2)
        self.assertEqual(out.data, [5.5, 10.5, 11.5, 24.5])
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(v.grad, [8.0, 12.0, 8.0, 12.0])
        self.assertEqual(b.grad, [2.0, 2.0])

    def test_repeated_backward_accumulates(self):
        x, w, b = _sample()
        out = x.linear_batch(w, b, 2)
        out.backward([1.0, 2.0, 3.0, 4.0])
        out.backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [9.0, -1.0, 19.0, -4.0])
        self.assertEqual(w.grad, [20.0, 28.0, 28.0, 40.0])
        self.assertEqual(b.grad, [8.0, 12.0])

    def test_shared_leaf_accumulates_across_graphs(self):
        x = Tensor([1.0, 2.0], True)
        w1 = Tensor([1.0, 2.0], True)
        w2 = Tensor([3.0, 4.0], True)
        b = Tensor([0.0], True)
        y1 = x.linear_batch(w1, b, 1)
        y2 = x.linear_batch(w2, b, 1)
        y1.add(y2).backward([1.0])
        self.assertEqual(x.grad, [4.0, 6.0])    # w1 + w2 rows
        self.assertEqual(w1.grad, [1.0, 2.0])
        self.assertEqual(w2.grad, [1.0, 2.0])
        self.assertEqual(b.grad, [2.0])

    def test_numerical_gradient_input(self):
        w = Tensor([0.5, -1.0, 2.0, 0.25])
        b = Tensor([0.1, -0.2])
        passed, error = gradcheck(
            lambda v: v.linear_batch(w, b, 2).sum(),
            [0.3, -0.7, 1.25, 0.5],
            atol=1e-6,
        )
        self.assertTrue(passed, msg=f"max abs error {error}")

    def test_numerical_gradient_weight(self):
        x = Tensor([0.3, -0.7, 1.25, 0.5])
        b = Tensor([0.1, -0.2])
        passed, error = gradcheck(
            lambda v: x.linear_batch(v, b, 2).sum(),
            [0.5, -1.0, 2.0, 0.25],
            atol=1e-6,
        )
        self.assertTrue(passed, msg=f"max abs error {error}")

    def test_numerical_gradient_bias(self):
        x = Tensor([0.3, -0.7, 1.25, 0.5])
        w = Tensor([0.5, -1.0, 2.0, 0.25])
        passed, error = gradcheck(
            lambda v: x.linear_batch(w, v, 2).sum(),
            [0.1, -0.2],
            atol=1e-6,
        )
        self.assertTrue(passed, msg=f"max abs error {error}")


class LinearBatchSnapshotTests(unittest.TestCase):
    def test_mutating_input_lists_after_forward_is_isolated(self):
        x_data = [1.0, 2.0]
        w_data = [3.0, 4.0]
        b_data = [0.5]
        x = Tensor(x_data, True)
        w = Tensor(w_data, True)
        b = Tensor(b_data, True)
        out = x.linear_batch(w, b, 1)
        x_data[0] = 100.0
        w_data[0] = 200.0
        b_data[0] = 9.0
        x_data.append(7.0)
        out.backward([1.0])
        self.assertEqual(x.grad, [3.0, 4.0])
        self.assertEqual(w.grad, [1.0, 2.0])
        self.assertEqual(b.grad, [1.0])

    def test_replacing_input_data_after_forward_is_isolated(self):
        x = Tensor([1.0, 2.0], True)
        w = Tensor([3.0, 4.0], True)
        b = Tensor([0.5], True)
        out = x.linear_batch(w, b, 1)
        x.data = [9.0, 9.0]
        w.data = [9.0, 9.0]
        b.data = [9.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [3.0, 4.0])
        self.assertEqual(w.grad, [1.0, 2.0])
        self.assertEqual(b.grad, [1.0])

    def test_repeated_backward_uses_same_snapshot(self):
        x_data = [1.0, 2.0]
        x = Tensor(x_data, True)
        w = Tensor([3.0, 4.0], True)
        b = Tensor([0.5], True)
        out = x.linear_batch(w, b, 1)
        out.backward([1.0])
        x_data[:] = [0.0, 0.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [6.0, 8.0])
        self.assertEqual(w.grad, [2.0, 4.0])
        self.assertEqual(b.grad, [2.0])


class LinearBatchBackwardAtomicityTests(unittest.TestCase):
    def test_nonfinite_contribution_leaves_prior_grads_untouched(self):
        x = Tensor([1.0], True)
        w = Tensor([1e308], True)
        b = Tensor([0.0], True)
        out = x.linear_batch(w, b, 1)
        self.assertTrue(math.isfinite(out.data[0]))
        out.backward([1.0])
        grads_before = (list(x.grad), list(w.grad), list(b.grad))
        with self.assertRaises(ValueError):
            # h * w overflows in dx[0].
            out.backward([1e308])
        self.assertEqual(x.grad, grads_before[0])
        self.assertEqual(w.grad, grads_before[1])
        self.assertEqual(b.grad, grads_before[2])

    def test_nonfinite_accumulation_aborts_before_any_write(self):
        # db[0] accumulates h over q; a huge second-batch grad overflows
        # the running db total mid-pass, after earlier roles advanced.
        x = Tensor([1.0, 1.0], True)
        w = Tensor([1.0], True)
        b = Tensor([0.0], True)
        out = x.linear_batch(w, b, 2)
        out.backward([1.0, 1.0])
        grads_before = (list(x.grad), list(w.grad), list(b.grad))
        with self.assertRaises(ValueError):
            out.backward([1e308, 1e308])
        self.assertEqual(x.grad, grads_before[0])
        self.assertEqual(w.grad, grads_before[1])
        self.assertEqual(b.grad, grads_before[2])

    def test_failure_in_one_branch_preserves_whole_graph(self):
        big = Tensor([1e308], True)
        x1 = Tensor([1.0], True)
        small = Tensor([0.1], True)
        x2 = Tensor([0.2], True)
        b = Tensor([0.0], True)
        c1 = big.linear_batch(x1, b, 1)   # non-finite under a huge upstream
        c2 = small.linear_batch(x2, b, 1)  # stays finite for the same grad
        z = c1.add(c2)
        big.grad = [5.0]
        x1.grad = [9.0]
        small.grad = [7.0]
        x2.grad = [8.0]
        b.grad = [3.0]
        with self.assertRaises(ValueError):
            # dx for big = 1e308 * 1e308 overflows; the whole pass must
            # abort before any grad is written.
            z.backward([1e308, 1e308])
        self.assertEqual(big.grad, [5.0])
        self.assertEqual(x1.grad, [9.0])
        self.assertEqual(small.grad, [7.0])
        self.assertEqual(x2.grad, [8.0])
        self.assertEqual(b.grad, [3.0])
        self.assertIsNone(c1.grad)
        self.assertIsNone(c2.grad)
        self.assertIsNone(z.grad)


if __name__ == "__main__":
    unittest.main()
