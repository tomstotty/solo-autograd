"""Contract tests for Tensor.fold2d.

fold2d is the scatter adjoint of unfold2d: the unfolded
(or, oc, c, kr, kc) tensor is scattered back into the C*H*W image,
accumulating overlapping windows and dropping taps outside the image.

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


def unfold(data, C, H, W, K, S=1, P=0, D=1):
    return Tensor(data).unfold2d(C, H, W, K, S, P, D).data


class Fold2dForwardTests(unittest.TestCase):
    def test_basic_window(self):
        # C=1, 2x2 image, K=2, S=1 -> OH=OW=1; one window folds verbatim
        out = Tensor([1.0, 2.0, 3.0, 4.0]).fold2d(1, 2, 2, 2)
        self.assertEqual(out.data, [1.0, 2.0, 3.0, 4.0])

    def test_stride_one_overlapping_windows_accumulate(self):
        # 3x3, K=2, S=1 -> OH=OW=2; corner cells covered once, edge cells
        # twice, the center cell four times.
        flat = unfold([float(i) for i in range(9)], 1, 3, 3, 2, 1)
        out = Tensor(flat).fold2d(1, 3, 3, 2, 1)
        self.assertEqual(
            out.data,
            [0.0, 1.0 + 1.0, 2.0,
             3.0 + 3.0, 4.0 * 4, 5.0 + 5.0,
             6.0, 7.0 + 7.0, 8.0],
        )

    def test_channel_ordering(self):
        # Two channels, K=1, OH=OW=2: the unfolded window holds (c0, c1)
        # taps adjacently in ascending (or, oc, c, kr, kc) order; folding
        # scatters them back apart into the per-channel images.
        data = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
        out = Tensor(data).fold2d(2, 2, 2, 1)
        self.assertEqual(out.data, [1.0, 2.0, 3.0, 4.0,
                                   10.0, 20.0, 30.0, 40.0])

    def test_stride_two_non_overlapping(self):
        flat = unfold([float(i) for i in range(16)], 1, 4, 4, 2, 2)
        out = Tensor(flat).fold2d(1, 4, 4, 2, 2)
        self.assertEqual(out.data, [float(i) for i in range(16)])

    def test_padding_drops_out_of_range_taps(self):
        # 1x1 image, K=1, S=1, P=1 -> OH=OW=3; only the center window's
        # tap lands in the image.
        flat = unfold([5.0], 1, 1, 1, 1, 1, 1)
        out = Tensor(flat).fold2d(1, 1, 1, 1, 1, 1)
        self.assertEqual(out.data, [5.0])

    def test_padding_with_kernel(self):
        # 2x2, K=2, S=2, P=1 -> OH=OW=2; each window sees one image corner.
        flat = [0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 2.0, 0.0,
                0.0, 3.0, 0.0, 0.0,
                4.0, 0.0, 0.0, 0.0]
        out = Tensor(flat).fold2d(1, 2, 2, 2, 2, 1)
        self.assertEqual(out.data, [1.0, 2.0, 3.0, 4.0])

    def test_dilation_spaces_taps(self):
        # 3x3, K=2, S=1, D=2 -> OH=OW=1, the four corners
        flat = unfold([float(i) for i in range(9)], 1, 3, 3, 2, 1, 0, 2)
        out = Tensor(flat).fold2d(1, 3, 3, 2, 1, 0, 2)
        self.assertEqual(
            out.data,
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 8.0],
        )

    def test_dilation_with_stride_and_padding(self):
        # 4x4, K=2, S=2, P=1, D=2: E=3; OH=OW=2. With S=D=2 the windows'
        # in-range taps only ever hit rows/cols {1, 3}; cell (1,1) is
        # covered by all four windows, edge cells twice.
        flat = unfold([float(i) for i in range(16)], 1, 4, 4, 2, 2, 1, 2)
        out = Tensor(flat).fold2d(1, 4, 4, 2, 2, 1, 2)
        self.assertEqual(out.data, [
            0.0, 0.0, 0.0, 0.0,
            0.0, 20.0, 0.0, 14.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 26.0, 0.0, 15.0,
        ])

    def test_output_length(self):
        flat = [1.0] * (1 * 3 * 3 * 4)
        out = Tensor(flat).fold2d(3, 2, 4, 2, 1)
        # OH=1, OW=3 -> image is C*H*W = 24
        self.assertEqual(len(out.data), 3 * 2 * 4)

    def test_fold_is_inverse_of_unfold_for_unit_coverage(self):
        # With S=K and no padding/dilation every cell is covered exactly
        # once, so fold(unfold(x)) == x.
        data = [0.1 * k - 1.3 for k in range(12)]
        flat = unfold(data, 3, 2, 2, 2, 2)
        out = Tensor(flat).fold2d(3, 2, 2, 2, 2)
        self.assertEqual(out.data, data)

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).fold2d(1, 2, 2, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, 1.0]))


class Fold2dValidationTests(unittest.TestCase):
    def test_data_must_be_nonempty_float_vector(self):
        expect(ValueError, lambda: _raw(1.0, False).fold2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw([], False).fold2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw("x", False).fold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, 2], False).fold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, True], False).fold2d(1, 1, 1, 1))
        # K=1, S=1 on a 1x1 image expects one input element; a list whose
        # only element is non-finite fails the data precondition.
        expect(ValueError,
               lambda: _raw([float("nan")], False).fold2d(1, 1, 1, 1))
        expect(ValueError,
               lambda: _raw([float("inf")], False).fold2d(1, 1, 1, 1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw([1.0], 1).fold2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0], "x").fold2d(1, 1, 1, 1))

    def test_input_length_must_match_oh_ow_c_kk(self):
        # 2x2 image, K=2, S=1 -> OH=OW=1, expect 4 inputs
        expect(ValueError, lambda: Tensor([1.0, 2.0]).fold2d(1, 2, 2, 2))
        expect(ValueError, lambda: Tensor([1.0] * 5).fold2d(1, 2, 2, 2))
        # 1x1 image, K=1 -> expect 1 input
        expect(ValueError, lambda: Tensor([1.0, 2.0]).fold2d(1, 1, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "1", None, [1]):
            expect(TypeError, lambda bad=bad: t.fold2d(bad, 1, 1, 1))
            expect(TypeError, lambda bad=bad: t.fold2d(1, bad, 1, 1))
            expect(TypeError, lambda bad=bad: t.fold2d(1, 1, bad, 1))
            expect(TypeError, lambda bad=bad: t.fold2d(1, 1, 1, bad))
            expect(TypeError, lambda bad=bad: t.fold2d(1, 1, 1, 1, bad))
            expect(TypeError, lambda bad=bad: t.fold2d(1, 1, 1, 1, 1, 0, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: t.fold2d(bad, 1, 1, 1))
            expect(ValueError, lambda bad=bad: t.fold2d(1, bad, 1, 1))
            expect(ValueError, lambda bad=bad: t.fold2d(1, 1, bad, 1))
            expect(ValueError, lambda bad=bad: t.fold2d(1, 1, 1, bad))
            expect(ValueError, lambda bad=bad: t.fold2d(1, 1, 1, 1, bad))
            expect(ValueError, lambda bad=bad: t.fold2d(1, 1, 1, 1, 1, 0, bad))

    def test_padding_must_be_non_negative_int(self):
        t = Tensor([1.0])
        for bad in (True, False, 1.0, 2.5, "0", None, [0]):
            expect(TypeError, lambda bad=bad: t.fold2d(1, 1, 1, 1, 1, bad))
        for bad in (-1, -3):
            expect(ValueError, lambda bad=bad: t.fold2d(1, 1, 1, 1, 1, bad))

    def test_non_positive_output_dimensions_are_value_error(self):
        # 1x2 image, K=3 -> OH = (1-3)//1+1 = -1
        expect(ValueError, lambda: Tensor([1.0]).fold2d(1, 1, 2, 3))
        expect(ValueError, lambda: Tensor([1.0]).fold2d(1, 2, 1, 3))
        # dilation can push the window beyond the image: 2x2, K=2, D=3
        # -> E=4, OH = (2-4)//1+1 = -1
        expect(ValueError,
               lambda: Tensor([1.0]).fold2d(1, 2, 2, 2, 1, 0, 3))
        # padding can rescue an otherwise empty output: 1x2, K=3, P=1
        # -> OH=1, OW=2, expecting OH*OW*C*K*K = 18 inputs
        Tensor([1.0] * 18).fold2d(1, 1, 2, 3, 1, 1)

    def test_failure_leaves_input_untouched(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        expect(ValueError, lambda: x.fold2d(1, 2, 2, 3))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(x.grad)


class Fold2dBackwardTests(unittest.TestCase):
    def test_basic_backward_single_window(self):
        # one window, all taps in range -> grad gathered verbatim
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.fold2d(1, 2, 2, 2).backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [1.0, 2.0, 3.0, 4.0])

    def test_backward_gathers_in_unfold_order(self):
        # 3x3, K=2, S=1; the adjoint of the overlapping scatter gathers
        # the unfolded windows from the image grad.
        data = [float(i) for i in range(16)]
        x = Tensor(data, True)
        out = x.fold2d(1, 3, 3, 2, 1)
        image_grad = [10.0 + i for i in range(9)]
        out.backward(image_grad)
        self.assertEqual(
            x.grad,
            unfold(image_grad, 1, 3, 3, 2, 1),
        )

    def test_backward_padding_fills_out_of_range_taps_with_zero(self):
        # 2x2, K=2, S=2, P=1 -> OH=OW=2; each window has one in-range tap;
        # the other three grad slots must come back as 0.0.
        x = Tensor([float(i) for i in range(16)], True)
        out = x.fold2d(1, 2, 2, 2, 2, 1)
        out.backward([3.0, 6.0, 9.0, 12.0])
        self.assertEqual(
            x.grad,
            [0.0, 0.0, 0.0, 3.0,
             0.0, 0.0, 6.0, 0.0,
             0.0, 9.0, 0.0, 0.0,
             12.0, 0.0, 0.0, 0.0],
        )

    def test_backward_channels_do_not_mix(self):
        x = Tensor([float(i) for i in range(8)], True)
        out = x.fold2d(2, 2, 2, 1)
        image_grad = [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
        out.backward(image_grad)
        # unfolded order: window slots hold (c0 tap, c1 tap) adjacently
        self.assertEqual(x.grad, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).fold2d(1, 2, 2, 2)
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
        expect(ValueError,
               lambda: out.backward([1.0, 1.0, 1.0, float("nan")]))
        expect(ValueError,
               lambda: out.backward([float("inf"), 1.0, 1.0, 1.0]))

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.fold2d(1, 2, 2, 2)
        x.data = [100.0, 100.0, 100.0, 100.0]
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 1.0, 1.0, 1.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = x.fold2d(1, 2, 2, 2)
        out.backward([1.0, 2.0, 3.0, 4.0])
        out.backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [2.0, 4.0, 6.0, 8.0])

    def test_backward_on_separate_graphs_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.fold2d(1, 2, 2, 2).backward([1.0, 1.0, 1.0, 1.0])
        x.fold2d(1, 2, 2, 2).backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [2.0, 2.0, 2.0, 2.0])

    def test_shared_input_paths_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        y = x.fold2d(1, 2, 2, 2).add(x.fold2d(1, 2, 2, 2))
        y.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [2.0, 2.0, 2.0, 2.0])

    def test_nonfinite_forward_intermediate_leaves_grads_untouched(self):
        # Two huge contributions scatter onto the same image cell and
        # overflow while accumulating; the forward must abort before any
        # result tensor exists. 3x3, K=2, S=1, OH=OW=2: unfolded slots
        # 3 (window(0,0) tap(1,1)) and 6 (window(0,1) tap(1,0)) both land
        # on the center image cell (1,1).
        huge = sys_float_max()
        flat = [0.0] * 16
        flat[3] = huge
        flat[6] = huge
        x = Tensor(flat, True)
        expect(ValueError, lambda: x.fold2d(1, 3, 3, 2, 1))
        self.assertIsNone(x.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.grad = [1.0, float("inf"), 1.0, 1.0]
        out = x.fold2d(1, 2, 2, 2)
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(x.grad, [1.0, float("inf"), 1.0, 1.0])

    def test_fold_unfold_roundtrip_grad(self):
        # fold(unfold(x)) is the identity on non-overlapping windows, so
        # its gradcheck must match numerical derivatives.
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9]

        def fn(d):
            flat = d.unfold2d(2, 2, 2, 2, 2)
            return flat.fold2d(2, 2, 2, 2, 2).sum()

        passed, error = gradcheck(fn, data, eps=1e-6, atol=1e-5)
        self.assertTrue(passed, f"error={error}")

    def test_gradcheck_matches_numerical_derivatives(self):
        # Overlapping fold: each output is a fixed linear sum of inputs.
        flat = [0.1 * k - 0.7 for k in range(16)]
        passed, error = gradcheck(
            lambda d: d.fold2d(1, 3, 3, 2, 1).sum(), flat, eps=1e-6,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        # Padded folded windows drop three of every four taps.
        passed, error = gradcheck(
            lambda d: d.fold2d(1, 2, 2, 2, 2, 1).sum(),
            [0.2 * k - 1.1 for k in range(16)], eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        # Dilated taps.
        passed, error = gradcheck(
            lambda d: d.fold2d(1, 3, 3, 2, 1, 0, 2).sum(),
            [0.15 * k - 0.4 for k in range(4)], eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
