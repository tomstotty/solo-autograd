"""Contract tests for Tensor.avg_pool1d.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def _raw(data, requires_grad=True):
    """Build a Tensor bypassing constructor validation."""
    return Tensor._make(data, requires_grad, (), None)


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"expected {exc.__name__}, no error raised")


class AvgPoolForwardTests(unittest.TestCase):
    def test_basic_value_stride_defaults_to_kernel(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool1d(2)
        self.assertEqual(out.data, [1.5, 3.5])

    def test_stride_one_overlapping_windows(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool1d(3, 1)
        self.assertEqual(out.data, [2.0, 3.0])

    def test_kernel_equal_length_gives_one_output(self):
        out = Tensor([2.0, 4.0, 6.0]).avg_pool1d(3)
        self.assertEqual(out.data, [4.0])

    def test_uneven_tail_is_dropped(self):
        # L = (5 - 2) // 2 + 1 = 2; the last element never enters a window.
        out = Tensor([2.0, 4.0, 6.0, 8.0, 100.0]).avg_pool1d(2, 2)
        self.assertEqual(out.data, [3.0, 7.0])

    def test_stride_larger_than_kernel(self):
        # windows at 0 and 3
        out = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).avg_pool1d(2, 3)
        self.assertEqual(out.data, [1.5, 4.5])

    def test_accumulation_starts_from_zero_in_ascending_order(self):
        # 0.0 + 1e16 swallows the five subsequent 1.0 additions under the
        # prescribed ascending accumulation; a different order (e.g. adding
        # the 1.0s before the 1e16) yields a different rounded mean.
        data = [1e16, 1.0, 1.0, 1.0, 1.0, 1.0]
        acc = 0.0
        for value in data:
            acc += value
        out = Tensor(data).avg_pool1d(6)
        self.assertEqual(out.data, [acc / 6])
        self.assertEqual(out.data, [1e16 / 6])

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0]).avg_pool1d(2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class AvgPoolValidationTests(unittest.TestCase):
    def test_data_must_be_nonempty_float_vector(self):
        expect(ValueError, lambda: _raw(1.0, False).avg_pool1d(1))
        expect(ValueError, lambda: _raw([], False).avg_pool1d(1))
        expect(ValueError, lambda: _raw("x", False).avg_pool1d(1))
        expect(TypeError, lambda: _raw([1.0, 2], False).avg_pool1d(1))
        expect(TypeError, lambda: _raw([1.0, True], False).avg_pool1d(1))
        expect(ValueError,
               lambda: _raw([1.0, float("nan")], False).avg_pool1d(1))
        expect(ValueError,
               lambda: _raw([float("inf"), 1.0], False).avg_pool1d(1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw([1.0, 2.0], 1).avg_pool1d(1))
        expect(TypeError, lambda: _raw([1.0, 2.0], "x").avg_pool1d(1))

    def test_kernel_size_must_be_positive_int(self):
        t = Tensor([1.0, 2.0])
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: t.avg_pool1d(bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: t.avg_pool1d(bad))

    def test_stride_must_be_positive_int_when_given(self):
        t = Tensor([1.0, 2.0])
        for bad in (True, False, 1.0, 2.5, "2", 0.0):
            expect(TypeError, lambda bad=bad: t.avg_pool1d(1, bad))
        for bad in (0, -2):
            expect(ValueError, lambda bad=bad: t.avg_pool1d(1, bad))

    def test_none_stride_takes_kernel_size(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool1d(2, None)
        self.assertEqual(out.data, [1.5, 3.5])

    def test_non_positive_output_length_is_value_error(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool1d(3))
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool1d(3, 2))
        # exactly covering works; one input short does not
        Tensor([1.0, 2.0, 3.0]).avg_pool1d(3)

    def test_forward_intermediate_nonfinite_is_value_error(self):
        # Individual means are finite but a pooled window overflows, so the
        # data cannot be constructed directly; assemble a raw tensor.
        huge = sys_float_max()
        x = _raw([huge, huge], True)
        expect(ValueError, lambda: x.avg_pool1d(2))

    def test_failure_leaves_input_untouched(self):
        x = Tensor([1.0, 2.0, 3.0], True)
        expect(ValueError, lambda: x.avg_pool1d(4))
        self.assertEqual(x.data, [1.0, 2.0, 3.0])
        self.assertIsNone(x.grad)


def sys_float_max():
    return 1.7976931348623157e308


class AvgPoolBackwardTests(unittest.TestCase):
    def test_basic_backward_non_overlapping(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool1d(2)
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])

    def test_backward_distinct_grads_overlapping_windows(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], True)
        out = x.avg_pool1d(2, 1)
        self.assertEqual(out.data, [1.5, 2.5, 3.5, 4.5])
        out.backward([1.0, 3.0, 5.0, 7.0])
        # ascending o, i accumulation into dx
        self.assertEqual(
            x.grad,
            [1.0 / 2, (1.0 + 3.0) / 2, (3.0 + 5.0) / 2, (5.0 + 7.0) / 2,
             7.0 / 2],
        )

    def test_backward_stride_larger_than_kernel_zeroes_gaps(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], True)
        x.avg_pool1d(2, 3).backward([2.0, 4.0])
        self.assertEqual(x.grad, [1.0, 1.0, 0.0, 2.0, 2.0, 0.0])

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).avg_pool1d(2)
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # float -> V
        expect(ValueError, lambda: out.backward([1.0]))
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0]))
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward(True))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward(1))
        expect(TypeError, lambda: out.backward((1.0, 1.0)))
        expect(TypeError, lambda: out.backward([1.0, 2]))
        expect(ValueError, lambda: out.backward([1.0, float("nan")]))
        expect(ValueError, lambda: out.backward([float("inf"), 1.0]))

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool1d(2, 1)
        x.data = [100.0, 100.0, 100.0, 100.0]
        out.backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [0.5, 1.0, 1.0, 0.5])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool1d(2)
        out.backward([1.0, 1.0])
        out.backward([2.0, 4.0])
        self.assertEqual(x.grad, [1.5, 1.5, 2.5, 2.5])

    def test_backward_on_separate_graphs_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.avg_pool1d(2).backward([1.0, 1.0])
        x.avg_pool1d(2).backward([1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 1.0, 1.0, 1.0])

    def test_shared_input_paths_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        # Two overlapping-window pools both read x within a single graph;
        # their contributions merge at x.
        y = x.avg_pool1d(2, 1).add(x.avg_pool1d(2, 1))
        self.assertEqual(y.data, [3.0, 5.0, 7.0])
        y.backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 2.0, 2.0, 1.0])

    def test_nonfinite_backward_intermediate_leaves_grads_untouched(self):
        huge = sys_float_max()
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], True)
        out = x.avg_pool1d(3, 1)
        # each grad/k is finite (~6e307) but middle positions receive three
        # contributions, overflowing on accumulation
        expect(ValueError,
               lambda: out.backward([huge, huge, huge]))
        self.assertIsNone(x.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor([1.0, 2.0], True)
        x.grad = [1.0, float("inf")]
        out = x.avg_pool1d(2)
        expect(ValueError, lambda: out.backward([1.0, 1.0]))
        # failed pass does not touch the existing grad
        self.assertEqual(x.grad, [1.0, float("inf")])

    def test_gradcheck_matches_numerical_derivatives(self):
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1]
        passed, error = gradcheck(
            lambda d: d.avg_pool1d(3, 2).sum(), data, eps=1e-6, atol=1e-5
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.avg_pool1d(2, 1).sum(), data[:5], eps=1e-6, atol=1e-5
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
