"""Contract tests for Tensor.bilinear_resize2d.

Bilinear 2D resizing: a single-channel H x W image (row-major) is
resized to OH x OW. With align_corners=True the source row of output
row or is or*(H-1)/(OH-1) (0.0 when OH == 1); with align_corners=False
it is (or + 0.5)*H/OH - 0.5, and analogously for columns. Source
coordinates are truncated to the legal range, then split into floor
and capped upper indices with fractional interpolation weights.
Only the standard library is used; discovered via
``python -m unittest discover``.
"""

import unittest

from autograd import Tensor, jacobiancheck


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


IMG4 = [1.0, 2.0, 3.0, 4.0]
IMG3 = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]


class BilinearResize2dForwardTests(unittest.TestCase):
    def test_identity_size_returns_input(self):
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 2, 2)
        self.assertEqual(out.data, IMG4)
        out = Tensor(IMG3).bilinear_resize2d(3, 3, 3, 3, True)
        self.assertEqual(out.data, IMG3)

    def test_align_corners_pins_endpoints(self):
        # 2 rows -> 4 rows: rows are 0, 1/3, 2/3, 1.
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 4, 4, True)
        top = out.data[0:4]
        bottom = out.data[12:16]
        for got, want in zip(top, (1.0, 4.0 / 3, 5.0 / 3, 2.0)):
            self.assertAlmostEqual(got, want)
        for got, want in zip(bottom, (3.0, 10.0 / 3, 11.0 / 3, 4.0)):
            self.assertAlmostEqual(got, want)
        self.assertEqual(out.data[0], 1.0)
        self.assertEqual(out.data[15], 4.0)

    def test_align_corners_false_spacing(self):
        # 2 rows -> 4 rows: positions -0.25, 0.25, 0.75, 1.25, truncated
        # at both ends, and column positions are the same.
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 4, 4)
        self.assertEqual(
            out.data,
            [1.0, 1.25, 1.75, 2.0,
             1.5, 1.75, 2.25, 2.5,
             2.5, 2.75, 3.25, 3.5,
             3.0, 3.25, 3.75, 4.0],
        )

    def test_align_corners_true_single_output_row_pins_zero(self):
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 1, 3, True)
        # Row pinned to 0; columns 0, 0.5, 1.
        self.assertEqual(out.data, [1.0, 1.5, 2.0])

    def test_align_corners_false_single_output_centers(self):
        # r = 0.5*2/1 - 0.5 = 0.5 and c = 0.5 -> the four-pixel mean.
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 1, 1)
        self.assertEqual(out.data, [2.5])

    def test_single_pixel_image(self):
        out = Tensor([0.7]).bilinear_resize2d(1, 1, 2, 3)
        self.assertEqual(out.data, [0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        out = Tensor([0.7]).bilinear_resize2d(1, 1, 3, 1, True)
        self.assertEqual(out.data, [0.7, 0.7, 0.7])

    def test_downsample_3x3_to_2x2(self):
        # Unaligned positions are 0.25 and 0.75 on both axes.
        out = Tensor([float(v) for v in range(1, 10)]).bilinear_resize2d(
            3, 3, 2, 2
        )
        # (0.25, 0.25) -> 0.5625*1 + 0.1875*2 + 0.1875*4 + 0.0625*5 = 2.0
        self.assertAlmostEqual(out.data[0], 2.0)
        self.assertAlmostEqual(out.data[1], 3.5)
        self.assertAlmostEqual(out.data[2], 6.5)
        self.assertAlmostEqual(out.data[3], 8.0)

    def test_rectangular_resize(self):
        # 2 rows x 4 cols -> 3 rows x 2 cols, unaligned.
        img = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        out = Tensor(img).bilinear_resize2d(2, 4, 3, 2)
        # Row positions -1/6 (truncated to 0), 0.5, 7/6 (truncated to 1);
        # col positions 0.5 and 2.5.
        self.assertAlmostEqual(out.data[0], 1.5)
        self.assertAlmostEqual(out.data[1], 3.5)
        self.assertAlmostEqual(out.data[2], 3.5)
        self.assertAlmostEqual(out.data[3], 5.5)
        self.assertAlmostEqual(out.data[4], 5.5)
        self.assertAlmostEqual(out.data[5], 7.5)
        self.assertEqual(len(out.data), 6)

    def test_no_graph_without_requires_grad(self):
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 1, 1)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class BilinearResize2dValidationTests(unittest.TestCase):
    def test_data_must_be_nonempty_float_vector(self):
        expect(ValueError, lambda: _raw(1.0).bilinear_resize2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw([]).bilinear_resize2d(1, 1, 1, 1))
        expect(ValueError, lambda: _raw("x").bilinear_resize2d(1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1, 2]).bilinear_resize2d(1, 2, 1, 1))
        expect(TypeError,
               lambda: _raw([1.0, True]).bilinear_resize2d(1, 2, 1, 1))
        expect(ValueError,
               lambda: _raw([float("nan"), 1.0]).bilinear_resize2d(
                   1, 2, 1, 1))
        expect(ValueError,
               lambda: _raw([float("inf"), 1.0]).bilinear_resize2d(
                   1, 2, 1, 1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw(IMG4, 1).bilinear_resize2d(2, 2, 1, 1))
        expect(TypeError, lambda: _raw(IMG4, "x").bilinear_resize2d(
            2, 2, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(bad, 2, 1, 1))
            expect(TypeError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, bad, 1, 1))
            expect(TypeError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, 2, bad, 1))
            expect(TypeError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, 2, 1, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(bad, 2, 1, 1))
            expect(ValueError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, bad, 1, 1))
            expect(ValueError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, 2, bad, 1))
            expect(ValueError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, 2, 1, bad))

    def test_align_corners_must_be_bool(self):
        for bad in (1, 0, 1.0, None, "x", (True,)):
            expect(TypeError, lambda bad=bad: Tensor(IMG4)
                   .bilinear_resize2d(2, 2, 1, 1, bad))

    def test_input_length_must_equal_height_times_width(self):
        expect(ValueError, lambda: Tensor([1.0, 2.0, 3.0])
               .bilinear_resize2d(2, 2, 1, 1))
        expect(ValueError, lambda: Tensor([1.0]).bilinear_resize2d(2, 2, 1, 1))

    def test_failure_leaves_input_untouched(self):
        x = Tensor(IMG4, True)
        expect(ValueError, lambda: x.bilinear_resize2d(2, 2, 0, 1))
        self.assertEqual(x.data, IMG4)
        self.assertIsNone(x.grad)


class BilinearResize2dBackwardTests(unittest.TestCase):
    def test_center_grad_is_quarter_each(self):
        x = Tensor(IMG4, True)
        x.bilinear_resize2d(2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_identity_routes_grad_through(self):
        x = Tensor(IMG4, True)
        x.bilinear_resize2d(2, 2, 2, 2).backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(x.grad, [1.0, 2.0, 3.0, 4.0])

    def test_downstream_grads_sum_to_one_per_output(self):
        x = Tensor([float(v) for v in range(1, 10)], True)
        out = x.bilinear_resize2d(3, 3, 2, 2)
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(x.grad), 4.0)
        self.assertEqual(
            x.grad,
            [0.5625, 0.375, 0.5625, 0.375, 0.25, 0.375,
             0.5625, 0.375, 0.5625],
        )

    def test_backward_grad_validation(self):
        out = Tensor(IMG4, True).bilinear_resize2d(2, 2, 2, 2)
        expect(ValueError, out.backward)
        expect(ValueError, lambda: out.backward(1.0))
        expect(ValueError, lambda: out.backward(True))
        expect(ValueError, lambda: out.backward(1))
        expect(ValueError, lambda: out.backward([1.0]))
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward((1.0, 1.0, 1.0, 1.0)))
        expect(TypeError, lambda: out.backward([1, 1, 1, 1]))
        expect(TypeError, lambda: out.backward([True] * 4))
        expect(TypeError, lambda: out.backward(["x"] * 4))
        expect(ValueError, lambda: out.backward([float("nan")] * 4))
        expect(ValueError, lambda: out.backward([float("inf")] * 4))

    def test_validation_runs_on_graphless_result(self):
        out = Tensor(IMG4).bilinear_resize2d(2, 2, 2, 2)
        self.assertFalse(out.requires_grad)
        for bad, error in (
            (None, TypeError),
            ("x", TypeError),
            ((1.0, 1.0, 1.0, 1.0), TypeError),
            (True, ValueError),
            (1, ValueError),
            (1.0, ValueError),
        ):
            expect(error, lambda bad=bad: out.backward(bad))
        expect(ValueError, out.backward)
        expect(ValueError, lambda: out.backward([1.0, 1.0, 1.0, 1.0]))

    def test_invalid_grad_changes_no_grad(self):
        x = Tensor(IMG4, True)
        out = x.bilinear_resize2d(2, 2, 1, 1)
        expect(ValueError, lambda: out.backward(1.0))
        self.assertIsNone(x.grad)
        expect(TypeError, lambda: out.backward(None))
        self.assertIsNone(x.grad)
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor(IMG4, True)
        out = x.bilinear_resize2d(2, 2, 1, 1)
        x.data = [100.0, 200.0, 300.0, 400.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])

    def test_repeated_backward_accumulates(self):
        x = Tensor(IMG4, True)
        out = x.bilinear_resize2d(2, 2, 1, 1)
        out.backward([1.0])
        out.backward([1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])

    def test_separate_graphs_accumulate(self):
        x = Tensor(IMG4, True)
        x.bilinear_resize2d(2, 2, 1, 1).backward([1.0])
        x.bilinear_resize2d(2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])

    def test_nonfinite_backward_leaves_grads_untouched(self):
        huge = sys_float_max()
        # A 1x1 image replicated to 2x2 gives every tap weight 1.0, so
        # backpropagating four huge grads overflows and aborts the pass.
        x = Tensor([1.0], True)
        out = x.bilinear_resize2d(1, 1, 2, 2)
        expect(ValueError, lambda: out.backward([huge, huge, huge, huge]))
        self.assertIsNone(x.grad)
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [4.0])


class BilinearResize2dGradcheckTests(unittest.TestCase):
    def test_jacobian_matches_numerical(self):
        for align_corners in (False, True):
            for oh, ow in ((5, 4), (2, 2), (1, 1), (3, 7), (7, 3)):
                passed, error = jacobiancheck(
                    lambda d, oh=oh, ow=ow, ac=align_corners:
                        d.bilinear_resize2d(3, 3, oh, ow, ac),
                    IMG3, eps=1e-6, atol=1e-4,
                )
                self.assertTrue(
                    passed,
                    f"align_corners={align_corners} {oh}x{ow} error={error}",
                )

    def test_jacobian_rectangular_image(self):
        img = [0.5 * v - 1.7 for v in range(1, 9)]
        passed, error = jacobiancheck(
            lambda d: d.bilinear_resize2d(2, 4, 5, 3),
            img, eps=1e-6, atol=1e-4,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = jacobiancheck(
            lambda d: d.bilinear_resize2d(2, 4, 5, 3, True),
            img, eps=1e-6, atol=1e-4,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
