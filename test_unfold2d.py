"""Contract tests for Tensor.unfold2d.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

import unittest

from autograd import Tensor, gradcheck


def _raw(data, requires_grad=False):
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


class Unfold2dForwardTests(unittest.TestCase):
    def test_basic_window(self):
        # C=1, 2x2 image, K=2, S=1 -> OH=OW=1, output is the window
        out = Tensor([1.0, 2.0, 3.0, 4.0]).unfold2d(1, 2, 2, 2)
        self.assertEqual(out.data, [1.0, 2.0, 3.0, 4.0])

    def test_stride_one_overlapping_windows(self):
        # 3x3, K=2, S=1 -> OH=OW=2; flattened in (or, oc, kr, kc) order
        data = [float(i) for i in range(9)]
        out = Tensor(data).unfold2d(1, 3, 3, 2, 1)
        self.assertEqual(out.data, [
            0.0, 1.0, 3.0, 4.0,
            1.0, 2.0, 4.0, 5.0,
            3.0, 4.0, 6.0, 7.0,
            4.0, 5.0, 7.0, 8.0,
        ])

    def test_channel_ordering(self):
        # Two channels; within one window the channels stay adjacent:
        # ascending (or, oc, c, kr, kc).
        data = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        out = Tensor(data).unfold2d(2, 2, 2, 1)
        self.assertEqual(
            out.data,
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        )

    def test_stride_two_non_overlapping(self):
        data = [float(i) for i in range(16)]
        out = Tensor(data).unfold2d(1, 4, 4, 2, 2)
        self.assertEqual(out.data, [
            0.0, 1.0, 4.0, 5.0,
            2.0, 3.0, 6.0, 7.0,
            8.0, 9.0, 12.0, 13.0,
            10.0, 11.0, 14.0, 15.0,
        ])

    def test_padding_fills_out_of_range_taps_with_zero(self):
        # 1x1 image, K=1, S=1, P=1 -> OH=OW=3
        out = Tensor([5.0]).unfold2d(1, 1, 1, 1, 1, 1)
        self.assertEqual(
            out.data,
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_padding_with_kernel(self):
        # 2x2, K=2, S=2, P=1 -> OH=OW=2; each window sees exactly one
        # corner of the image and three padded zeros.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).unfold2d(1, 2, 2, 2, 2, 1)
        self.assertEqual(out.data, [
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            4.0, 0.0, 0.0, 0.0,
        ])

    def test_dilation_spaces_taps(self):
        # 3x3, K=2, S=1, D=2 -> OH=OW=1, the four corners
        data = [float(i) for i in range(9)]
        out = Tensor(data).unfold2d(1, 3, 3, 2, 1, 0, 2)
        self.assertEqual(out.data, [0.0, 2.0, 6.0, 8.0])

    def test_dilation_with_stride_and_padding(self):
        # 4x4, K=2, S=2, P=1, D=2: E=3; OH=OW=(4+2-3)//2+1=2
        data = [float(i) for i in range(16)]
        out = Tensor(data).unfold2d(1, 4, 4, 2, 2, 1, 2)
        self.assertEqual(out.data, [
            0.0, 0.0, 0.0, 5.0,
            0.0, 0.0, 5.0, 7.0,
            0.0, 5.0, 0.0, 13.0,
            5.0, 7.0, 13.0, 15.0,
        ])

    def test_output_length(self):
        out = Tensor([1.0] * 24).unfold2d(3, 2, 4, 2, 1)
        # OH=1, OW=3, C=3, K*K=4 -> 36
        self.assertEqual(len(out.data), 1 * 3 * 3 * 4)

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).unfold2d(1, 2, 2, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, 1.0]))


class Unfold2dValidationTests(unittest.TestCase):
    def test_data_must_be_nonempty_float_vector(self):
        expect(ValueError, lambda: _raw(1.0, False).unfold2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw([], False).unfold2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw("x", False).unfold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, 2], False).unfold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, True], False).unfold2d(1, 1, 1, 1))
        expect(ValueError,
               lambda: _raw([1.0, float("nan")], False).unfold2d(1, 1, 2, 1))
        expect(ValueError,
               lambda: _raw([float("inf"), 1.0], False).unfold2d(1, 1, 2, 1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw([1.0], 1).unfold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0], "x").unfold2d(1, 1, 1, 1))

    def test_input_length_must_match_c_h_w(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0]).unfold2d(1, 1, 1, 1))
        expect(ValueError, lambda: Tensor([1.0, 2.0]).unfold2d(1, 2, 2, 1))
        expect(ValueError, lambda: Tensor([1.0]).unfold2d(2, 1, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "1", None, [1]):
            expect(TypeError, lambda bad=bad: t.unfold2d(bad, 1, 1, 1))
            expect(TypeError, lambda bad=bad: t.unfold2d(1, bad, 1, 1))
            expect(TypeError, lambda bad=bad: t.unfold2d(1, 1, bad, 1))
            expect(TypeError, lambda bad=bad: t.unfold2d(1, 1, 1, bad))
            expect(TypeError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, bad))
            expect(TypeError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, 1, 0, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: t.unfold2d(bad, 1, 1, 1))
            expect(ValueError, lambda bad=bad: t.unfold2d(1, bad, 1, 1))
            expect(ValueError, lambda bad=bad: t.unfold2d(1, 1, bad, 1))
            expect(ValueError, lambda bad=bad: t.unfold2d(1, 1, 1, bad))
            expect(ValueError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, bad))
            expect(ValueError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, 1, 0, bad))

    def test_padding_must_be_non_negative_int(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "0", None, [0]):
            expect(TypeError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, 1, bad))
        for bad in (-1, -3):
            expect(ValueError, lambda bad=bad: t.unfold2d(1, 1, 1, 1, 1, bad))

    def test_non_positive_output_dimensions_are_value_error(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0]).unfold2d(1, 1, 2, 3))
        expect(ValueError, lambda: Tensor([1.0, 2.0]).unfold2d(1, 2, 1, 3))
        # dilation can push the window beyond the image
        expect(ValueError, lambda: Tensor([1.0, 2.0, 3.0, 4.0])
               .unfold2d(1, 2, 2, 2, 1, 0, 3))
        # padding can rescue an otherwise empty output
        Tensor([1.0, 2.0]).unfold2d(1, 1, 2, 3, 1, 1)

    def test_failure_leaves_input_untouched(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        expect(ValueError, lambda: x.unfold2d(1, 2, 2, 3))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(x.grad)


class Unfold2dBackwardTests(unittest.TestCase):
    def test_basic_backward_single_window(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.unfold2d(1, 2, 2, 2).backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [1.0, 2.0, 3.0, 4.0])

    def test_backward_overlapping_windows_accumulate(self):
        # 3x3, K=2, S=1: corner cells covered once, edge cells twice,
        # the center four times.
        data = [float(i) for i in range(9)]
        x = Tensor(data, True)
        out = x.unfold2d(1, 3, 3, 2, 1)
        out.backward([1.0] * len(out.data))
        self.assertEqual(
            x.grad,
            [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        )

    def test_backward_padding_drops_out_of_range_taps(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.unfold2d(1, 2, 2, 2, 2, 1)
        out.backward([float(i) for i in range(16)])
        # each image cell appears in exactly one window
        self.assertEqual(x.grad, [3.0, 6.0, 9.0, 12.0])

    def test_backward_channels_do_not_mix(self):
        x = Tensor([1.0] * 8, True)
        out = x.unfold2d(2, 2, 2, 1)
        out.backward([float(i) for i in range(8)])
        # window order 0..3, each holding (c0, c1)
        self.assertEqual(x.grad, [0.0, 2.0, 4.0, 6.0,
                                  1.0, 3.0, 5.0, 7.0])

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).unfold2d(1, 2, 2, 2)
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # float -> V
        expect(ValueError, lambda: out.backward(True))
        expect(ValueError, lambda: out.backward(1))
        expect(ValueError, lambda: out.backward([1.0]))
        expect(ValueError, lambda: out.backward([1.0] * 5))
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward((1.0,) * 4))
        expect(TypeError, lambda: out.backward([1.0, 1.0, 1.0, 2]))
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, float("nan")]))
        expect(ValueError,
               lambda: out.backward([float("inf"), 1.0, 1.0, 1.0]))

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.unfold2d(1, 2, 2, 2)
        x.data = [100.0, 100.0, 100.0, 100.0]
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 1.0, 1.0, 1.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.unfold2d(1, 2, 2, 2)
        out.backward([1.0, 2.0, 3.0, 4.0])
        out.backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [2.0, 4.0, 6.0, 8.0])

    def test_backward_on_separate_graphs_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.unfold2d(1, 2, 2, 2).backward([1.0, 1.0, 1.0, 1.0])
        x.unfold2d(1, 2, 2, 2).backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [2.0, 2.0, 2.0, 2.0])

    def test_shared_input_paths_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        y = x.unfold2d(1, 2, 2, 2).add(x.unfold2d(1, 2, 2, 2))
        y.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [2.0, 2.0, 2.0, 2.0])

    def test_nonfinite_backward_intermediate_leaves_grads_untouched(self):
        huge = sys_float_max()
        x = Tensor([1.0] * 9, True)
        out = x.unfold2d(1, 3, 3, 2, 1)
        # the center cells receive four contributions, overflowing
        expect(ValueError, lambda: out.backward([huge] * 16))
        self.assertIsNone(x.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.grad = [1.0, float("inf"), 1.0, 1.0]
        out = x.unfold2d(1, 2, 2, 2)
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(x.grad, [1.0, float("inf"), 1.0, 1.0])

    def test_gradcheck_matches_numerical_derivatives(self):
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]
        passed, error = gradcheck(
            lambda d: d.unfold2d(1, 3, 3, 2, 1).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.unfold2d(1, 3, 3, 2, 1, 1, 1).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.unfold2d(2, 2, 2, 2, 2, 1, 1).sum(),
            [0.1 * k - 0.3 for k in range(8)], eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = gradcheck(
            lambda d: d.unfold2d(1, 3, 3, 2, 1, 0, 2).sum(), data, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
