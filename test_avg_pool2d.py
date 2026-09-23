"""Contract tests for Tensor.avg_pool2d.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

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


def sys_float_max():
    return 1.7976931348623157e308


class AvgPool2dForwardTests(unittest.TestCase):
    def test_basic_value_stride_defaults_to_kernel(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool2d(2, 2, 2)
        self.assertEqual(out.data, [2.5])

    def test_stride_one_overlapping_windows(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        out = out.avg_pool2d(3, 3, 2, 1)
        self.assertEqual(out.data, [3.0, 4.0, 6.0, 7.0])

    def test_padding_zero_matches_legacy_behavior(self):
        data = [float(i) for i in range(1, 17)]
        self.assertEqual(
            Tensor(data).avg_pool2d(4, 4, 2).data,
            Tensor(data).avg_pool2d(4, 4, 2, None, 0).data,
        )
        self.assertEqual(
            Tensor(data).avg_pool2d(4, 4, 2).data, [3.5, 5.5, 11.5, 13.5]
        )

    def test_padding_adds_zero_border(self):
        # 2x2 image, K=2, S=2, P=1 -> OH=OW=(2+2-2)//2+1=2
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool2d(2, 2, 2, 2, 1)
        self.assertEqual(out.data, [1.0 / 4, 2.0 / 4, 3.0 / 4, 4.0 / 4])

    def test_padding_with_stride_one(self):
        # 2x2 image, K=3, S=1, P=1 -> OH=OW=(2+2-3)//1+1=2
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool2d(2, 2, 3, 1, 1)
        self.assertEqual(
            out.data,
            [
                (1.0 + 2.0 + 3.0 + 4.0) / 9,
                (1.0 + 2.0 + 3.0 + 4.0) / 9,
                (1.0 + 2.0 + 3.0 + 4.0) / 9,
                (1.0 + 2.0 + 3.0 + 4.0) / 9,
            ],
        )

    def test_fully_out_of_range_window_is_zero(self):
        # K=1, S=1, P=1 on a 1x1 image -> OH=OW=3; corner windows see no
        # valid tap and yield 0.0.
        out = Tensor([5.0]).avg_pool2d(1, 1, 1, 1, 1)
        self.assertEqual(
            out.data,
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_denominator_stays_kernel_squared_with_padding(self):
        # The padded tap counts as 0.0 in the sum but the denominator is
        # still K*K.
        out = Tensor([4.0]).avg_pool2d(1, 1, 2, 1, 1)
        # OH=OW=(1+2-2)//1+1=2; windows anchored at (-1,-1),(-1,0),(0,-1),(0,0)
        self.assertEqual(out.data, [1.0, 1.0, 1.0, 1.0])

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool2d(2, 2, 2, 2, 1)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class AvgPool2dValidationTests(unittest.TestCase):
    def test_data_must_be_nonempty_float_vector(self):
        expect(ValueError, lambda: _raw(1.0, False).avg_pool2d(1, 1, 1))
        expect(ValueError, lambda: _raw([], False).avg_pool2d(1, 1, 1))
        expect(ValueError, lambda: _raw("x", False).avg_pool2d(1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, 2], False).avg_pool2d(1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, True], False).avg_pool2d(1, 1, 1))
        expect(ValueError,
               lambda: _raw([1.0, float("nan")], False).avg_pool2d(1, 1, 1))
        expect(ValueError,
               lambda: _raw([float("inf"), 1.0], False).avg_pool2d(1, 1, 1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw([1.0], 1).avg_pool2d(1, 1, 1))
        expect(TypeError, lambda: _raw([1.0], "x").avg_pool2d(1, 1, 1))

    def test_input_length_must_equal_height_times_width(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool2d(2, 2, 1))
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool2d(1, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: t.avg_pool2d(bad, 1, 1))
            expect(TypeError, lambda bad=bad: t.avg_pool2d(1, bad, 1))
            expect(TypeError, lambda bad=bad: t.avg_pool2d(1, 1, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: t.avg_pool2d(bad, 1, 1))
            expect(ValueError, lambda bad=bad: t.avg_pool2d(1, bad, 1))
            expect(ValueError, lambda bad=bad: t.avg_pool2d(1, 1, bad))

    def test_stride_must_be_positive_int_when_given(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "2", 0.0):
            expect(TypeError, lambda bad=bad: t.avg_pool2d(1, 1, 1, bad))
        for bad in (0, -2):
            expect(ValueError, lambda bad=bad: t.avg_pool2d(1, 1, 1, bad))

    def test_none_stride_takes_kernel_size(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).avg_pool2d(2, 2, 2, None)
        self.assertEqual(out.data, [2.5])

    def test_padding_must_be_non_negative_int(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "1", None, [1]):
            expect(TypeError, lambda bad=bad: t.avg_pool2d(1, 1, 1, 1, bad))
        for bad in (-1, -3):
            expect(ValueError, lambda bad=bad: t.avg_pool2d(1, 1, 1, 1, bad))

    def test_non_positive_output_dimensions_are_value_error(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool2d(1, 2, 3))
        expect(ValueError, lambda: Tensor([1.0, 2.0]).avg_pool2d(2, 1, 3))
        # padding can rescue an otherwise empty output
        Tensor([1.0, 2.0]).avg_pool2d(1, 2, 3, 1, 1)

    def test_forward_intermediate_nonfinite_is_value_error(self):
        huge = sys_float_max()
        x = _raw([huge, huge, huge, huge], True)
        expect(ValueError, lambda: x.avg_pool2d(2, 2, 2))

    def test_failure_leaves_input_untouched(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        expect(ValueError, lambda: x.avg_pool2d(2, 2, 3))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(x.grad)


class AvgPool2dBackwardTests(unittest.TestCase):
    def test_basic_backward_non_overlapping(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.avg_pool2d(2, 2, 2).backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_backward_with_padding_ignores_out_of_range_taps(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool2d(2, 2, 2, 2, 1)
        self.assertEqual(out.data, [0.25, 0.5, 0.75, 1.0])
        out.backward([1.0, 2.0, 3.0, 4.0])
        # each input cell is covered by exactly one window
        self.assertEqual(x.grad, [0.25, 0.5, 0.75, 1.0])

    def test_backward_overlapping_windows_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.avg_pool2d(2, 2, 2, 1).backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).avg_pool2d(2, 2, 2, 2, 1)
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # float -> V
        expect(ValueError, lambda: out.backward([1.0]))
        expect(ValueError, lambda: out.backward([1.0] * 5))
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward(True))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward(1))
        expect(TypeError, lambda: out.backward((1.0,) * 4))
        expect(TypeError, lambda: out.backward([1.0, 1.0, 1.0, 2]))
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, float("nan")]))
        expect(ValueError,
               lambda: out.backward([float("inf"), 1.0, 1.0, 1.0]))

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool2d(2, 2, 2, 2, 1)
        x.data = [100.0, 100.0, 100.0, 100.0]
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.avg_pool2d(2, 2, 2)
        out.backward([1.0])
        out.backward([2.0])
        self.assertEqual(x.grad, [0.75, 0.75, 0.75, 0.75])

    def test_backward_on_separate_graphs_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.avg_pool2d(2, 2, 2).backward([1.0])
        x.avg_pool2d(2, 2, 2).backward([1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])

    def test_shared_input_paths_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        y = x.avg_pool2d(2, 2, 2, 2, 1).add(x.avg_pool2d(2, 2, 2, 2, 1))
        y.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])

    def test_nonfinite_backward_intermediate_leaves_grads_untouched(self):
        huge = sys_float_max()
        x = Tensor([1.0] * 25, True)
        out = x.avg_pool2d(5, 5, 3, 1)
        # each grad/(K*K) is finite (~2e307) but the center cell receives
        # nine contributions, overflowing on accumulation
        expect(ValueError, lambda: out.backward([huge] * 9))
        self.assertIsNone(x.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.grad = [1.0, float("inf"), 1.0, 1.0]
        out = x.avg_pool2d(2, 2, 2)
        expect(ValueError, lambda: out.backward([1.0]))
        # failed pass does not touch the existing grad
        self.assertEqual(x.grad, [1.0, float("inf"), 1.0, 1.0])

    def test_gradcheck_matches_numerical_derivatives(self):
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]
        passed, error = gradcheck(
            lambda d: d.avg_pool2d(3, 3, 2, 1).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.avg_pool2d(3, 3, 2, 1, 1).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.avg_pool2d(3, 3, 3, 3, 1).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
