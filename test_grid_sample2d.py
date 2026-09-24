"""Contract tests for Tensor.grid_sample2d.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

import unittest

from autograd import Tensor, gradcheck, jacobiancheck


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


class GridSample2dForwardTests(unittest.TestCase):
    def test_aligned_corners_hit_exact_pixels(self):
        img = [1.0, 2.0, 3.0, 4.0]
        grid = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        out = Tensor(img).grid_sample2d(Tensor(grid), 2, 2, 2, 2)
        self.assertEqual(out.data, [1.0, 2.0, 3.0, 4.0])

    def test_center_is_average_of_four_neighbours(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [2.5])

    def test_bilinear_weights_along_x(self):
        # gx=-0.5 -> x = 0.25 -> col 0 weight 0.75, col 1 weight 0.25;
        # gy=-1 pins row 0.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-0.5, -1.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [1.25])

    def test_bilinear_weights_along_y(self):
        # gy=-0.5 -> y = 0.25 -> row 0 weight 0.75, row 1 weight 0.25;
        # gx=-1 pins col 0: 0.75*1 + 0.25*3 = 1.5.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-1.0, -0.5]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [1.5])

    def test_integer_coordinate_inside_image(self):
        nine = [float(i) for i in range(1, 10)]
        # gx=1 -> x=(W-1)=2 -> col 2, gy=-1 -> row 0 -> value 3
        out = Tensor(nine).grid_sample2d(
            Tensor([1.0, -1.0]), 3, 3, 1, 1
        )
        self.assertEqual(out.data, [3.0])
        # gx=-1 -> col 0, gy=0 -> y=1 -> row 1 -> value 4
        out = Tensor(nine).grid_sample2d(
            Tensor([-1.0, 0.0]), 3, 3, 1, 1
        )
        self.assertEqual(out.data, [4.0])

    def test_partially_out_of_range_still_samples_in_range_taps(self):
        # gx=-1.5 -> x=-0.25: col -1 is out of range, col 0 keeps
        # weight 0.75; gy=-1 pins row 0.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-1.5, -1.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [0.75])

    def test_fully_out_of_range_point_is_zero(self):
        # gx=-3.1 -> x=-1.05, both sampled columns out of range.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-3.1, 0.0]), 2, 2, 1, 1
        )
        self.assertEqual(out.data, [0.0])

    def test_one_by_one_image(self):
        # scale_x = scale_y = 0; every finite point maps to the sole pixel
        # whenever the (0, 0) tap is in range, which is i=j=0, i.e.
        # gx, gy in [-3, 1].
        out = Tensor([7.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 1, 1, 1, 1
        )
        self.assertEqual(out.data, [7.0])

    def test_grid_is_row_major_pairs(self):
        # Points map in the same order the (gx, gy) pairs are stored.
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([-1.0, -1.0, 0.0, 0.0]), 2, 2, 1, 2
        )
        self.assertEqual(out.data, [1.0, 2.5])

    def test_output_length_is_out_height_times_out_width(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2, 2, 1, 3
        )
        self.assertEqual(out.data, [2.5, 2.5, 2.5])

    def test_no_graph_without_requires_grad(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0]).grid_sample2d(
            Tensor([0.0, 0.0]), 2, 2, 1, 1
        )
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, lambda: out.backward([1.0]))


class GridSample2dValidationTests(unittest.TestCase):
    def test_grid_must_be_tensor(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        expect(TypeError, lambda: x.grid_sample2d([0.0, 0.0], 2, 2, 1, 1))
        expect(TypeError, lambda: x.grid_sample2d("x", 2, 2, 1, 1))
        expect(TypeError, lambda: x.grid_sample2d(None, 2, 2, 1, 1))

    def test_dimensions_must_be_positive_ints(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        grid = Tensor([0.0, 0.0])
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda b=bad: x.grid_sample2d(grid, b, 2, 1, 1))
            expect(TypeError, lambda b=bad: x.grid_sample2d(grid, 2, b, 1, 1))
            expect(TypeError, lambda b=bad: x.grid_sample2d(grid, 2, 2, b, 1))
            expect(TypeError, lambda b=bad: x.grid_sample2d(grid, 2, 2, 1, b))
        for bad in (0, -1, -3):
            expect(ValueError, lambda b=bad: x.grid_sample2d(grid, b, 2, 1, 1))
            expect(ValueError, lambda b=bad: x.grid_sample2d(grid, 2, b, 1, 1))
            expect(ValueError, lambda b=bad: x.grid_sample2d(grid, 2, 2, b, 1))
            expect(ValueError, lambda b=bad: x.grid_sample2d(grid, 2, 2, 1, b))

    def test_input_data_must_be_float_list(self):
        grid = _raw([0.0, 0.0])
        expect(ValueError, lambda: _raw(1.0).grid_sample2d(grid, 2, 2, 1, 1))
        expect(ValueError, lambda: _raw([]).grid_sample2d(grid, 2, 2, 1, 1))
        expect(ValueError, lambda: _raw("x").grid_sample2d(grid, 2, 2, 1, 1))
        expect(
            TypeError,
            lambda: _raw([1.0, 2, 3.0, 4.0]).grid_sample2d(grid, 2, 2, 1, 1),
        )
        expect(
            TypeError,
            lambda: _raw([1.0, True, 3.0, 4.0]).grid_sample2d(
                grid, 2, 2, 1, 1
            ),
        )

    def test_grid_data_must_be_float_list(self):
        x = _raw([1.0, 2.0, 3.0, 4.0])
        expect(ValueError, lambda: x.grid_sample2d(_raw(1.0), 2, 2, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(_raw([]), 2, 2, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(_raw("x"), 2, 2, 1, 1))
        expect(TypeError, lambda: x.grid_sample2d(_raw([1, 0.0]), 2, 2, 1, 1))
        expect(
            TypeError,
            lambda: x.grid_sample2d(_raw([False, 0.0]), 2, 2, 1, 1),
        )

    def test_requires_grad_must_be_bool(self):
        grid = _raw([0.0, 0.0])
        expect(
            TypeError,
            lambda: _raw([1.0, 2.0, 3.0, 4.0], 1).grid_sample2d(
                grid, 2, 2, 1, 1
            ),
        )
        expect(
            TypeError,
            lambda: _raw([1.0, 2.0, 3.0, 4.0], "x").grid_sample2d(
                grid, 2, 2, 1, 1
            ),
        )
        x = _raw([1.0, 2.0, 3.0, 4.0])
        expect(
            TypeError,
            lambda: x.grid_sample2d(_raw([0.0, 0.0], 1), 2, 2, 1, 1),
        )

    def test_data_must_be_finite(self):
        grid = _raw([0.0, 0.0])
        expect(
            ValueError,
            lambda: _raw([1.0, 2.0, 3.0, float("nan")]).grid_sample2d(
                grid, 2, 2, 1, 1
            ),
        )
        expect(
            ValueError,
            lambda: _raw([1.0, 2.0, float("inf"), 4.0]).grid_sample2d(
                grid, 2, 2, 1, 1
            ),
        )
        x = _raw([1.0, 2.0, 3.0, 4.0])
        expect(
            ValueError,
            lambda: x.grid_sample2d(_raw([float("nan"), 0.0]), 2, 2, 1, 1),
        )
        expect(
            ValueError,
            lambda: x.grid_sample2d(_raw([0.0, float("-inf")]), 2, 2, 1, 1),
        )

    def test_lengths_must_match_shapes(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        expect(ValueError, lambda: x.grid_sample2d(Tensor([0.0, 0.0]), 1, 1, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(Tensor([0.0, 0.0]), 2, 3, 1, 1))
        expect(ValueError, lambda: x.grid_sample2d(Tensor([0.0]), 2, 2, 1, 1))
        expect(
            ValueError,
            lambda: x.grid_sample2d(Tensor([0.0, 0.0, 0.0, 0.0]), 2, 2, 1, 1),
        )
        expect(
            ValueError,
            lambda: x.grid_sample2d(
                Tensor([0.0, 0.0, 0.0, 0.0]), 2, 2, 2, 2
            ),
        )

    def test_failure_leaves_inputs_untouched(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        expect(ValueError, lambda: x.grid_sample2d(g, 2, 2, 1, 2))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(g.data, [0.0, 0.0])
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)


class GridSample2dBackwardTests(unittest.TestCase):
    def test_corner_point_gradients(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([-1.0, -1.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0, 0.0])
        # At the boundary the forward floor branch is fixed: moving gx
        # blends col 1 into col 0, so dgx = scale_x*(2-1) = 0.5 and
        # dgy = scale_y*(3-1) = 1.0.
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_center_gradients(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        # dgx = scale_x * sum_v wy[v] * (col1_v - col0_v)
        #     = 0.5 * 0.5 * ((2 - 1) + (4 - 3)) = 0.5
        # dgy = scale_y * sum_u wx[u] * (row1_u - row0_u)
        #     = 0.5 * 0.5 * ((3 - 1) + (4 - 2)) = 1.0
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_off_center_gradients(self):
        # Point (gx, gy) = (-0.5, -1.0): x=0.25, y=0, a=0.25, b=0.
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([-0.5, -1.0], True)
        x.grid_sample2d(g, 2, 2, 1, 1).backward([2.0])
        self.assertEqual(x.grad, [1.5, 0.5, 0.0, 0.0])
        # dgx = 2 * scale_x * (2 - 1) = 1.0; the boundary dgy blends
        # row 1 into row 0: 2 * scale_y * (0.75*3 + 0.25*4
        # - 0.75*1 - 0.25*2) = 2.0.
        self.assertEqual(g.grad, [1.0, 2.0])

    def test_out_of_range_point_has_zero_gradients(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([3.0, 3.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        self.assertEqual(out.data, [0.0])
        out.backward([1.0])
        self.assertEqual(x.grad, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(g.grad, [0.0, 0.0])

    def test_multiple_points_accumulate_into_input(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([-1.0, -1.0, 1.0, 1.0], True)
        x.grid_sample2d(g, 2, 2, 1, 2).backward([1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0, 1.0])
        # Boundary grid grads: point (-1,-1) gives [0.5, 1.0]; point
        # (1,1) sits next to pixel 4 with the other neighbours out of
        # range, giving dgx = -4*scale_x = -2.0 and dgy = -2.0.
        self.assertEqual(g.grad, [0.5, 1.0, -2.0, -2.0])

    def test_one_by_one_image_backward(self):
        x = Tensor([7.0], True)
        g = Tensor([0.0, 0.0], True)
        x.grid_sample2d(g, 1, 1, 1, 1).backward([2.0])
        self.assertEqual(x.grad, [2.0])
        # Both scales are zero, so grid coordinates carry no gradient.
        self.assertEqual(g.grad, [0.0, 0.0])

    def test_only_grid_requires_grad(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        self.assertIs(x.requires_grad, False)
        g = Tensor([0.0, 0.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        out.backward([1.0])
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_same_tensor_in_both_roles_merges_gradients(self):
        # A length-2 list acts as both a 1x2 image and the single grid
        # point (gx, gy) = (1, 2). x = (gx+1)*(W-1)/2 = 1 -> col 1 with
        # a = 0; y = (gy+1)*(H-1)/2 = 0 -> row 0. Only the (v=0, u=0)
        # tap at pixel (0, 1) is in range, weighted 1*1.
        t = Tensor([1.0, 2.0], True)
        out = t.grid_sample2d(t, 1, 2, 1, 1)
        self.assertEqual(out.data, [2.0])
        out.backward([1.0])
        # Image role: d_input = [0, 1]. Grid role: scale_y = 0 so
        # dgy = 0; dgx = 1 * 2 * wy0 * (-1) * scale_x = -1.0. The two
        # roles are merged by Tensor identity into [-1.0, 1.0].
        self.assertEqual(t.grad, [-1.0, 1.0])

    def test_backward_grad_validation(self):
        out = Tensor([1.0, 2.0, 3.0, 4.0], True).grid_sample2d(
            Tensor([0.0, 0.0], True), 2, 2, 1, 1
        )
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # scalar float -> V
        expect(ValueError, lambda: out.backward(1))    # int -> V
        expect(ValueError, lambda: out.backward(True))  # bool -> V
        expect(ValueError, lambda: out.backward([1.0, 2.0]))  # length -> V
        expect(ValueError, lambda: out.backward([float("nan")]))
        expect(ValueError, lambda: out.backward([float("inf")]))
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward((1.0,)))
        expect(TypeError, lambda: out.backward([1]))

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([0.0, 0.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        x.data = [9.0, 9.0, 9.0, 9.0]
        g.data = [5.0, 5.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [0.25, 0.25, 0.25, 0.25])
        # Grid grads depend on the snapshotted image/positions, not the
        # mutated grid values.
        self.assertEqual(g.grad, [0.5, 1.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        g = Tensor([-1.0, -1.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        out.backward([1.0])
        out.backward([2.0])
        self.assertEqual(x.grad, [3.0, 0.0, 0.0, 0.0])
        # Repeated backward recomputes with the merged upstream 3.0:
        # grid grads are 3.0 * [0.5, 1.0].
        self.assertEqual(g.grad, [1.5, 3.0])

    def test_backward_on_separate_graphs_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.grid_sample2d(Tensor([-1.0, -1.0]), 2, 2, 1, 1).backward([1.0])
        x.grid_sample2d(Tensor([-1.0, -1.0]), 2, 2, 1, 1).backward([1.0])
        self.assertEqual(x.grad, [2.0, 0.0, 0.0, 0.0])

    def test_nonfinite_backward_intermediate_leaves_grads_untouched(self):
        huge = sys_float_max()
        x = Tensor([huge, 1.0, 1.0, 1.0], True)
        g = Tensor([-1.0, -1.0], True)
        out = x.grid_sample2d(g, 2, 2, 1, 1)
        # huge * huge * 1 overflows while computing the grid gradient.
        expect(ValueError, lambda: out.backward([huge]))
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        x.grad = [1.0, float("inf"), 1.0, 1.0]
        out = x.grid_sample2d(Tensor([-1.0, -1.0]), 2, 2, 1, 1)
        expect(ValueError, lambda: out.backward([1.0]))
        self.assertEqual(x.grad, [1.0, float("inf"), 1.0, 1.0])

    def test_gradcheck_matches_numerical_image_derivatives(self):
        data = [0.3, -1.2, 2.4, 0.7]
        for gx, gy in ((0.2, -0.4), (-1.0, -1.0), (1.0, 1.0), (-3.1, 0.0)):
            grid = [gx, gy]
            passed, error = gradcheck(
                lambda d, grid=grid: d.grid_sample2d(
                    Tensor(grid), 2, 2, 1, 1
                ).sum(),
                data,
                eps=1e-6,
                atol=1e-5,
            )
            self.assertTrue(passed, f"grid={grid} error={error}")

    def test_gradcheck_matches_numerical_grid_derivatives(self):
        data = [0.3, -1.2, 2.4, 0.7]
        for grid in ([0.2, -0.4], [-0.9, 0.3]):
            passed, error = gradcheck(
                lambda g: Tensor(data).grid_sample2d(
                    g, 2, 2, 1, 1
                ).sum(),
                list(grid),
                eps=1e-6,
                atol=1e-5,
            )
            self.assertTrue(passed, f"grid={grid} error={error}")

    def test_jacobiancheck_multi_point_image(self):
        image = [0.31, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]
        grid = [-0.3, 0.1, 0.6, -0.7, 0.2, 0.9, -0.9, -0.1]
        passed, error = jacobiancheck(
            lambda d: d.grid_sample2d(Tensor(grid), 3, 3, 2, 2),
            image,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobiancheck_multi_point_grid(self):
        image = [0.31, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]
        grid = [-0.3, 0.1, 0.6, -0.7, 0.2, 0.9, -0.9, -0.1]
        passed, error = jacobiancheck(
            lambda g: Tensor(image).grid_sample2d(g, 3, 3, 2, 2),
            grid,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobiancheck_with_out_of_range_points(self):
        image = [0.31, -1.2, 2.4, 0.7, -0.5, 3.1, 1.1, -0.9, 0.2]
        grid = [-0.3, 0.1, 5.0, -7.0, 0.2, 0.9, -0.9, -0.1]
        passed, error = jacobiancheck(
            lambda d: d.grid_sample2d(Tensor(grid), 3, 3, 2, 2),
            image,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")
        passed, error = jacobiancheck(
            lambda g: Tensor(image).grid_sample2d(g, 3, 3, 2, 2),
            grid,
            atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
