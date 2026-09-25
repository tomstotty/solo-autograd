"""Contract tests for Tensor.grid_sample2d.

Bilinear 2D grid sampling: a single-channel H x W image (row-major) is
sampled at OH*OW normalized (gx, gy) coordinate pairs (also row-major)
mapped via x = (gx + 1) * (W - 1) / 2 and y = (gy + 1) * (H - 1) / 2.
Only the standard library is used; discovered via
``python -m unittest discover``.
"""

import unittest

from autograd import Tensor, jacobiancheck


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


IMG3 = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]


class GridSample2dForwardTests(unittest.TestCase):
    def test_normalized_corners_hit_exact_pixels(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0]),
            2, 2, 2, 2,
        )
        # (-1,-1)->top-left, (1,1)->bottom-right, (1,-1)->top-right,
        # (-1,1)->bottom-left.
        self.assertEqual(out.data, [1.0, 4.0, 2.0, 3.0])

    def test_center_bilaterally_averages_four_pixels(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [2.5])

    def test_fractional_coord_interpolates(self):
        # gx=-0.5 -> x=0.25 (wx=(0.75,0.25)); gy=-1 -> y=0 (top row).
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-0.5, -1.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [1.25])

    def test_out_of_range_coord_zeros_all_taps(self):
        # gx=-3 -> x=-2 (i=-2, both columns out) regardless of gy.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-3.0, -1.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [0.0])

    def test_partially_out_of_range_keeps_in_range_weight(self):
        # gx=-2 -> x=-0.5: i=-1, only column 0 survives with wx[1]=0.5;
        # gy=-1 pins the top row, so 0.5 * pixel(0,0) = 0.5.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-2.0, -1.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [0.5])

    def test_single_pixel_image(self):
        # H=W=1 collapses both scales to 0; the lone pixel is always picked.
        out = Tensor([0.7]).grid_sample2d(
            Tensor([0.3, -0.2, 0.9, 0.1]), 1, 1, 2, 1
        )
        self.assertEqual(out.data, [0.7, 0.7])

    def test_output_length_is_out_height_times_out_width(self):
        grid = [-0.9, -0.8, -0.1, 0.2, 0.4, 0.3, 0.8, -0.7,
                0.1, 0.6, -0.6, 0.9]
        out = Tensor(IMG3).grid_sample2d(Tensor(grid), 3, 3, 3, 2)
        self.assertEqual(len(out.data), 6)

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 2, 2, 1, 1
        )
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class GridSample2dValidationTests(unittest.TestCase):
    def test_grid_must_be_a_tensor(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        for bad in ([0.0, 0.0], (0.0, 0.0), None, 1, 1.0, "x"):
            expect(TypeError, lambda bad=bad: x.grid_sample2d(
                bad, 2, 2, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        grid = Tensor([0.0, 0.0])
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: x.grid_sample2d(
                grid, bad, 2, 1, 1))
            expect(TypeError, lambda bad=bad: x.grid_sample2d(
                grid, 2, bad, 1, 1))
            expect(TypeError, lambda bad=bad: x.grid_sample2d(
                grid, 2, 2, bad, 1))
            expect(TypeError, lambda bad=bad: x.grid_sample2d(
                grid, 2, 2, 1, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: x.grid_sample2d(
                grid, bad, 2, 1, 1))
            expect(ValueError, lambda bad=bad: x.grid_sample2d(
                grid, 2, bad, 1, 1))
            expect(ValueError, lambda bad=bad: x.grid_sample2d(
                grid, 2, 2, bad, 1))
            expect(ValueError, lambda bad=bad: x.grid_sample2d(
                grid, 2, 2, 1, bad))

    def test_image_data_must_be_nonempty_float_vector(self):
        grid = Tensor([0.0, 0.0])
        expect(ValueError, lambda: _raw(1.0, False).grid_sample2d(
            grid, 1, 1, 1, 1))
        expect(ValueError, lambda: _raw([], False).grid_sample2d(
            grid, 1, 1, 1, 1))
        expect(ValueError, lambda: _raw("x", False).grid_sample2d(
            grid, 1, 1, 1, 1))
        expect(TypeError, lambda: _raw([1.0, 2], False).grid_sample2d(
            grid, 1, 2, 1, 1))
        expect(TypeError, lambda: _raw([1.0, True], False).grid_sample2d(
            grid, 1, 2, 1, 1))
        expect(ValueError,
               lambda: _raw([1.0, float("nan")], False).grid_sample2d(
                   grid, 1, 2, 1, 1))
        expect(ValueError,
               lambda: _raw([float("inf"), 1.0], False).grid_sample2d(
                   grid, 1, 2, 1, 1))

    def test_grid_data_must_be_nonempty_float_vector(self):
        x = Tensor([1.0])
        expect(ValueError, lambda: x.grid_sample2d(
            _raw(1.0, False), 1, 1, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            _raw([], False), 1, 1, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            _raw("x", False), 1, 1, 1, 1))
        expect(TypeError, lambda: x.grid_sample2d(
            _raw([0.0, 1], False), 1, 1, 1, 1))
        expect(TypeError, lambda: x.grid_sample2d(
            _raw([0.0, True], False), 1, 1, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            _raw([0.0, float("nan")], False), 1, 1, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            _raw([float("inf"), 0.0], False), 1, 1, 1, 1))

    def test_requires_grad_must_be_bool(self):
        grid = Tensor([0.0, 0.0])
        expect(TypeError, lambda: _raw([1.0, 2.0, 3.0, 4.0], 1)
               .grid_sample2d(grid, 2, 2, 1, 1))
        expect(TypeError, lambda: Tensor([1.0, 2.0, 3.0, 4.0])
               .grid_sample2d(_raw([0.0, 0.0], 1), 2, 2, 1, 1))

    def test_image_length_must_equal_height_times_width(self):
        grid = Tensor([0.0, 0.0])
        expect(ValueError, lambda: Tensor([1.0, 2.0, 3.0]).grid_sample2d(
            grid, 2, 2, 1, 1))
        expect(ValueError, lambda: Tensor([1.0]).grid_sample2d(
            grid, 2, 2, 1, 1))

    def test_grid_length_must_equal_two_times_output_size(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        expect(ValueError, lambda: x.grid_sample2d(
            Tensor([0.0]), 2, 2, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            Tensor([0.0, 0.0, 0.0]), 2, 2, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(
            Tensor([0.0] * 6), 2, 2, 1, 1))

    def test_failure_leaves_inputs_untouched(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0], True)
        expect(ValueError, lambda: x.grid_sample2d(g, 2, 2, 1, 1))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(g.data, [0.0])
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)


class GridSample2dBackwardTests(unittest.TestCase):
    def test_image_and_grid_grads_at_center(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        # dgx = 0.5 * [0.5*(2-1) + 0.5*(4-3)] = 0.5
        # dgy = 1.0 * [0.5*(3-1) + 0.5*(4-2)] = 1.0
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_corner_uses_fixed_forward_floor_branch(self):
        # At an exact pixel corner the floor branch is pinned: only that
        # pixel receives image grad; grid grad is the fixed-branch
        # analytical value (not the one-sided numerical kink value).
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([-1.0, -1.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_only_image_requires_grad(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], False)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        self.assertEqual(out.data, [2.5])
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        self.assertIsNone(g.grad)

    def test_only_grid_requires_grad(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], False)
        g = Tensor([0.0, 0.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(g.grad, [0.5, 1.0])
        self.assertIsNone(x.grad)

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).grid_sample2d(
            Tensor([0.0, 0.0], True), 2, 2, 1, 1)
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # float -> V
        expect(ValueError, lambda: out.backward(True))
        expect(ValueError, lambda: out.backward(1))
        expect(ValueError, lambda: out.backward([1.0, 1.0]))  # length 2
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward((1.0,)))
        expect(TypeError, lambda: out.backward([2]))
        expect(TypeError, lambda: out.backward([True]))
        expect(TypeError, lambda: out.backward(["x"]))
        expect(ValueError,
               lambda: out.backward([float("nan")]))
        expect(ValueError,
               lambda: out.backward([float("inf")]))

    def test_graphless_result_rejects_any_grad(self):
        # The missing-graph ValueError takes precedence over grad
        # validation: every grad, valid or not, raises ValueError.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 2, 2, 1, 1)
        self.assertFalse(out.requires_grad)
        for bad in (None, "x", (1.0,), True, 1, 1.0):
            expect(ValueError, lambda bad=bad: out.backward(bad))
        expect(ValueError, out.backward)
        expect(ValueError, lambda: out.backward([1.0]))

    def test_invalid_grad_changes_no_grad(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        expect(ValueError, lambda: out.backward(1.0))
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)
        expect(TypeError, lambda: out.backward(None))
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)
        # A valid pass still works after the failed ones.
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        x.data = [100.0, 200.0, 300.0, 400.0]
        g.data = [9.0, 9.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        out.backward([1.0])
        out.backward([1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(g.grad, [1.0, 2.0])

    def test_separate_graphs_accumulate(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(g.grad, [1.0, 2.0])

    def test_shared_input_roles_are_merged(self):
        # The same length-2 tensor plays both the image (H*W=2) and the
        # grid (2*OH*OW=2); its two roles are summed into one grad.
        t = Tensor([2.0, 3.0], True)
        t.grid_sample2d(t, 1, 2, 1, 1).backward([1.0])
        self.assertEqual(t.grad, [-1.5, 0.5])

    def test_nonfinite_backward_leaves_grads_untouched(self):
        huge = sys_float_max()
        # All four points sit at the 3x3 center (scale_y = 1); the
        # per-point dgy product g * (0.5*3 + 0.5*4) = 3.5 * huge is
        # non-finite, so the pass aborts. The failed pass changes no grad.
        x = Tensor([float(v) for v in range(1, 10)], True)
        g = Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
        out = x.grid_sample2d(g, 3, 3, 2, 2)
        expect(ValueError, lambda: out.backward([huge] * 4))
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)
        # A subsequent valid pass still works.
        out.backward([1.0, 1.0, 1.0, 1.0])
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(g.grad)


class GridSample2dGradcheckTests(unittest.TestCase):
    def test_image_jacobian_matches_numerical(self):
        # Interior, non-integer mapped coordinates avoid the bilinear
        # floor kinks so central differences agree with the fixed-floor
        # analytic gradients.
        grid = [-0.7, 0.3, 0.2, -0.4, 0.45, 0.28]
        passed, error = jacobiancheck(
            lambda d: d.grid_sample2d(Tensor(grid, True), 3, 3, 3, 1),
            IMG3, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_grid_jacobian_matches_numerical(self):
        grid = [-0.7, 0.3, 0.2, -0.4, 0.45, 0.28]
        passed, error = jacobiancheck(
            lambda gr: Tensor(IMG3, True).grid_sample2d(gr, 3, 3, 3, 1),
            grid, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobian_with_border_straddling_samples(self):
        # Coordinates map just outside [-1, 1] so one tap per axis is
        # out of range and contributes 0.0.
        img = IMG3
        grid = [-1.05, -1.05, 1.05, -0.95]
        passed, error = jacobiancheck(
            lambda d: d.grid_sample2d(Tensor(grid, True), 3, 3, 2, 1),
            img, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"image error={error}")
        passed, error = jacobiancheck(
            lambda gr: Tensor(img, True).grid_sample2d(gr, 3, 3, 2, 1),
            grid, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"grid error={error}")

    def test_single_pixel_degenerate_jacobian(self):
        # With H=W=1 both scales are 0, so grid coordinates cannot move
        # the sample: every grid grad is exactly 0.
        passed, error = jacobiancheck(
            lambda d: d.grid_sample2d(
                Tensor([0.3, -0.2, 0.9, 0.1], True), 1, 1, 2, 1),
            [0.7], eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"image error={error}")
        passed, error = jacobiancheck(
            lambda gr: Tensor([0.7], True).grid_sample2d(gr, 1, 1, 2, 1),
            [0.3, -0.2, 0.9, 0.1], eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"grid error={error}")


if __name__ == "__main__":
    unittest.main()
