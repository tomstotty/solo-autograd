"""Contract tests for Tensor.affine_grid2d.

A 2x3 affine matrix [a00, a01, a02, a10, a11, a12] maps normalized
sampling coordinates (x, y) in [-1, 1] (ascending row/column order, a
degenerate axis collapsing to 0.0) to output pairs
gx = a00*x + a01*y + a02, gy = a10*x + a11*y + a12, returned
interleaved. Only the standard library is used; discovered via
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


IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
THETA = [0.5, 0.25, 0.1, -0.5, 2.0, -0.3]


class AffineGrid2dForwardTests(unittest.TestCase):
    def test_identity_returns_normalized_corners(self):
        out = Tensor(IDENTITY).affine_grid2d(2, 2)
        self.assertEqual(
            out.data, [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        )

    def test_general_matrix_applies_per_point(self):
        out = Tensor(THETA).affine_grid2d(2, 2)
        self.assertEqual(
            out.data,
            [-0.65, -1.8, 0.35, -2.8, -0.15, 2.2, 0.85, 1.2],
        )

    def test_translation_shifts_every_point(self):
        out = Tensor([1.0, 0.0, 0.5, 0.0, 1.0, -0.25]).affine_grid2d(2, 1)
        # x collapses to 0.0; y is -1.0 then 1.0.
        self.assertEqual(out.data, [0.5, -1.25, 0.5, 0.75])

    def test_single_row_collapses_y_to_zero(self):
        out = Tensor(IDENTITY).affine_grid2d(1, 3)
        self.assertEqual(
            out.data, [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )

    def test_single_column_collapses_x_to_zero(self):
        out = Tensor(IDENTITY).affine_grid2d(3, 1)
        self.assertEqual(
            out.data, [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        )

    def test_single_point_is_origin(self):
        out = Tensor(IDENTITY).affine_grid2d(1, 1)
        self.assertEqual(out.data, [0.0, 0.0])

    def test_output_length_is_two_times_rows_times_cols(self):
        out = Tensor(IDENTITY).affine_grid2d(3, 4)
        self.assertEqual(len(out.data), 24)

    def test_no_graph_without_requires_grad(self):
        out = Tensor(IDENTITY).affine_grid2d(2, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class AffineGrid2dValidationTests(unittest.TestCase):
    def test_data_must_be_a_six_element_list(self):
        expect(ValueError, lambda: _raw(1.0, False).affine_grid2d(1, 1))
        expect(ValueError, lambda: _raw("x", False).affine_grid2d(1, 1))
        expect(ValueError, lambda: _raw([], False).affine_grid2d(1, 1))
        expect(ValueError, lambda: Tensor([0.0] * 5).affine_grid2d(1, 1))
        expect(ValueError, lambda: Tensor([0.0] * 7).affine_grid2d(1, 1))

    def test_data_elements_must_be_floats(self):
        expect(TypeError, lambda: _raw(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0], False).affine_grid2d(1, 1))
        expect(TypeError, lambda: _raw(
            [1.0, 0.0, 0.0, 0.0, 1.0, True], False).affine_grid2d(1, 1))
        expect(TypeError, lambda: _raw(
            [1.0, 0.0, 0.0, 0.0, 1.0, "0"], False).affine_grid2d(1, 1))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw(list(IDENTITY), 1)
               .affine_grid2d(1, 1))

    def test_data_elements_must_be_finite(self):
        expect(ValueError, lambda: _raw(
            [1.0, 0.0, 0.0, 0.0, 1.0, float("nan")], False)
            .affine_grid2d(1, 1))
        expect(ValueError, lambda: _raw(
            [float("inf"), 0.0, 0.0, 0.0, 1.0, 0.0], False)
            .affine_grid2d(1, 1))

    def test_rows_and_cols_must_be_positive_ints(self):
        theta = Tensor(IDENTITY)
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: theta.affine_grid2d(bad, 2))
            expect(TypeError, lambda bad=bad: theta.affine_grid2d(2, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: theta.affine_grid2d(bad, 2))
            expect(ValueError, lambda bad=bad: theta.affine_grid2d(2, bad))

    def test_nonfinite_forward_result_rejected(self):
        huge = sys_float_max()
        # gx = huge * 1.0 + 0.0 + huge overflows to inf.
        expect(ValueError, lambda: Tensor(
            [huge, 0.0, huge, 0.0, 1.0, 0.0]).affine_grid2d(2, 2))

    def test_failure_leaves_input_untouched(self):
        theta = Tensor(list(IDENTITY), True)
        expect(ValueError, lambda: theta.affine_grid2d(0, 2))
        self.assertEqual(theta.data, IDENTITY)
        self.assertIsNone(theta.grad)


class AffineGrid2dBackwardTests(unittest.TestCase):
    def test_grad_is_outer_product_of_upstream_and_coords(self):
        theta = Tensor(list(IDENTITY), True)
        theta.affine_grid2d(2, 2).backward(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        # coords (-1,-1), (1,-1), (-1,1), (1,1); u = 1,3,5,7; v = 2,4,6,8.
        self.assertEqual(
            theta.grad, [4.0, 8.0, 16.0, 4.0, 8.0, 20.0]
        )

    def test_single_point_passes_translation_through(self):
        # The lone point sits at the origin, so only the bias terms see
        # the upstream grad.
        theta = Tensor(list(IDENTITY), True)
        theta.affine_grid2d(1, 1).backward([3.0, -2.0])
        self.assertEqual(theta.grad, [0.0, 0.0, 3.0, 0.0, 0.0, -2.0])

    def test_grad_does_not_depend_on_theta_values(self):
        theta = Tensor(THETA, True)
        theta.affine_grid2d(2, 2).backward(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        self.assertEqual(
            theta.grad, [4.0, 8.0, 16.0, 4.0, 8.0, 20.0]
        )

    def test_backward_grad_validation(self):
        out = Tensor(list(IDENTITY), True).affine_grid2d(1, 2)
        expect(ValueError, out.backward)               # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))  # float -> V
        expect(ValueError, lambda: out.backward(True))
        expect(ValueError, lambda: out.backward(1))
        expect(ValueError, lambda: out.backward([1.0]))         # short
        expect(ValueError, lambda: out.backward([1.0] * 6))     # long
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward((1.0, 1.0)))
        expect(TypeError, lambda: out.backward([1.0, 1.0, 1.0, 2]))
        expect(TypeError, lambda: out.backward([1.0, 1.0, 1.0, True]))
        expect(TypeError, lambda: out.backward([1.0, 1.0, 1.0, "x"]))
        expect(ValueError,
               lambda: out.backward([1.0, 1.0, 1.0, float("nan")]))
        expect(ValueError,
               lambda: out.backward([float("inf"), 1.0, 1.0, 1.0]))

    def test_validation_runs_on_graphless_result(self):
        out = Tensor(IDENTITY).affine_grid2d(1, 2)
        self.assertFalse(out.requires_grad)
        for bad, error in (
            (None, TypeError),
            ("x", TypeError),
            ((1.0, 1.0), TypeError),
            (True, ValueError),
            (1, ValueError),
            (1.0, ValueError),
        ):
            expect(error, lambda bad=bad: out.backward(bad))
        expect(ValueError, out.backward)
        # A valid finite list passes validation, then hits the missing
        # graph error.
        expect(ValueError, lambda: out.backward([1.0, 1.0]))

    def test_invalid_grad_changes_no_grad(self):
        theta = Tensor(list(IDENTITY), True)
        out = theta.affine_grid2d(1, 1)
        expect(ValueError, lambda: out.backward(1.0))
        self.assertIsNone(theta.grad)
        expect(TypeError, lambda: out.backward(None))
        self.assertIsNone(theta.grad)
        # A valid pass still works after the failed ones.
        out.backward([1.0, 2.0])
        self.assertEqual(theta.grad, [0.0, 0.0, 1.0, 0.0, 0.0, 2.0])

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        theta = Tensor(list(IDENTITY), True)
        out = theta.affine_grid2d(2, 2)
        theta.data = [9.0] * 6
        out.backward([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(
            theta.grad, [4.0, 8.0, 16.0, 4.0, 8.0, 20.0]
        )

    def test_repeated_backward_accumulates(self):
        theta = Tensor(list(IDENTITY), True)
        out = theta.affine_grid2d(1, 1)
        out.backward([1.0, 2.0])
        out.backward([1.0, 2.0])
        self.assertEqual(theta.grad, [0.0, 0.0, 2.0, 0.0, 0.0, 4.0])

    def test_separate_graphs_accumulate(self):
        theta = Tensor(list(IDENTITY), True)
        theta.affine_grid2d(1, 1).backward([1.0, 2.0])
        theta.affine_grid2d(1, 1).backward([1.0, 2.0])
        self.assertEqual(theta.grad, [0.0, 0.0, 2.0, 0.0, 0.0, 4.0])

    def test_nonfinite_backward_leaves_grad_untouched(self):
        huge = sys_float_max()
        # Two points both contribute u = huge to dtheta[2], so the
        # running sum overflows and the pass aborts with no grad written.
        theta = Tensor(list(IDENTITY), True)
        out = theta.affine_grid2d(1, 2)
        expect(ValueError, lambda: out.backward([huge, 0.0, huge, 0.0]))
        self.assertIsNone(theta.grad)
        # A subsequent valid pass still works.
        out.backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(theta.grad, [2.0, 0.0, 4.0, 2.0, 0.0, 6.0])


class AffineGrid2dGradcheckTests(unittest.TestCase):
    def test_jacobian_matches_numerical(self):
        passed, error = jacobiancheck(
            lambda d: d.affine_grid2d(3, 2),
            THETA, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_degenerate_axes_jacobian_matches_numerical(self):
        passed, error = jacobiancheck(
            lambda d: d.affine_grid2d(1, 1),
            THETA, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
