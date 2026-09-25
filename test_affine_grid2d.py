"""Contract tests for Tensor.affine_grid2d.

Affine grid generation: the six entries
[a00, a01, a02, a10, a11, a12] map an ascending (r, c) grid of
normalized coordinates (x = 2*c/(cols-1) - 1, y = 2*r/(rows-1) - 1,
with a singleton axis collapsing to 0.0) to interleaved
(gx, gy) pairs with gx = a00*x + a01*y + a02 and
gy = a10*x + a11*y + a12. Only the standard library is used;
discovered via ``python -m unittest discover``.
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


class AffineGrid2dForwardTests(unittest.TestCase):
    def test_identity_2x3_grid(self):
        out = Tensor(IDENTITY).affine_grid2d(2, 3)
        # r=0: y=-1; r=1: y=1. cols give x = -1, 0, 1.
        self.assertEqual(out.data, [
            -1.0, -1.0, 0.0, -1.0, 1.0, -1.0,
            -1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
        ])

    def test_general_affine_maps_coordinates(self):
        # gx = x + 2y + 3, gy = 4x + 5y + 6.
        theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = Tensor(theta).affine_grid2d(2, 2)
        self.assertEqual(out.data, [
            0.0, -3.0,    # (-1,-1): gx=0,  gy=-3
            2.0, 5.0,     # ( 1,-1): gx=2,  gy=5
            4.0, 7.0,     # (-1, 1): gx=4,  gy=7
            6.0, 15.0,    # ( 1, 1): gx=6,  gy=15
        ])

    def test_singleton_rows_pins_y_to_zero(self):
        # gx = x + 2y + 3, gy = 4x + 5y + 6 with y = 0.
        theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = Tensor(theta).affine_grid2d(1, 3)
        self.assertEqual(out.data, [
            2.0, 2.0,     # (-1, 0)
            3.0, 6.0,     # ( 0, 0)
            4.0, 10.0,    # ( 1, 0)
        ])

    def test_singleton_cols_pins_x_to_zero(self):
        theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = Tensor(theta).affine_grid2d(3, 1)
        self.assertEqual(out.data, [
            1.0, 1.0,     # (0,-1)
            3.0, 6.0,     # (0, 0)
            5.0, 11.0,    # (0, 1)
        ])

    def test_single_point_uses_translation_only(self):
        theta = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        out = Tensor(theta).affine_grid2d(1, 1)
        self.assertEqual(out.data, [4.0, 7.0])

    def test_output_length_is_two_times_rows_times_cols(self):
        out = Tensor(IDENTITY).affine_grid2d(3, 2)
        self.assertEqual(len(out.data), 12)

    def test_no_graph_without_requires_grad(self):
        out = Tensor(IDENTITY).affine_grid2d(2, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)


class AffineGrid2dValidationTests(unittest.TestCase):
    def test_dimensions_must_be_positive_ints(self):
        for bad in (True, False, 1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: Tensor(IDENTITY)
                   .affine_grid2d(bad, 2))
            expect(TypeError, lambda bad=bad: Tensor(IDENTITY)
                   .affine_grid2d(2, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: Tensor(IDENTITY)
                   .affine_grid2d(bad, 2))
            expect(ValueError, lambda bad=bad: Tensor(IDENTITY)
                   .affine_grid2d(2, bad))

    def test_data_must_be_six_element_list(self):
        expect(ValueError, lambda: _raw(1.0, False).affine_grid2d(2, 2))
        expect(ValueError, lambda: _raw([], False).affine_grid2d(2, 2))
        expect(ValueError, lambda: _raw("x", False).affine_grid2d(2, 2))
        expect(ValueError, lambda: _raw([1.0] * 5, False)
               .affine_grid2d(2, 2))
        expect(ValueError, lambda: _raw([1.0] * 7, False)
               .affine_grid2d(2, 2))

    def test_data_elements_must_be_floats(self):
        expect(TypeError, lambda: _raw([0] * 6, False).affine_grid2d(2, 2))
        expect(TypeError,
               lambda: _raw([1.0, 2.0, 3.0, 4.0, 5.0, 6], False)
               .affine_grid2d(2, 2))
        expect(TypeError,
               lambda: _raw([1.0, 1.0, True, 1.0, 1.0, 1.0], False)
               .affine_grid2d(2, 2))
        expect(TypeError,
               lambda: _raw(["x"] + [1.0] * 5, False)
               .affine_grid2d(2, 2))

    def test_data_elements_must_be_finite(self):
        expect(ValueError,
               lambda: _raw([float("nan")] + [0.0] * 5, False)
               .affine_grid2d(2, 2))
        expect(ValueError,
               lambda: _raw([0.0] * 5 + [float("inf")], False)
               .affine_grid2d(2, 2))
        expect(ValueError,
               lambda: _raw([float("-inf")] + [0.0] * 5, False)
               .affine_grid2d(2, 2))

    def test_requires_grad_must_be_bool(self):
        expect(TypeError, lambda: _raw(IDENTITY, 1).affine_grid2d(2, 2))
        expect(TypeError, lambda: _raw(IDENTITY, 0).affine_grid2d(2, 2))

    def test_nonfinite_forward_raises_value_error(self):
        huge = sys_float_max()
        # At (x, y) = (1, 1): gx = huge + huge overflows.
        theta = [huge, huge, 0.0, 0.0, 0.0, 0.0]
        expect(ValueError, lambda: Tensor(theta).affine_grid2d(2, 2))

    def test_failure_leaves_input_untouched(self):
        x = Tensor(IDENTITY, True)
        expect(ValueError, lambda: x.affine_grid2d(2, 0))
        self.assertEqual(x.data, IDENTITY)
        self.assertIsNone(x.grad)
        bad = _raw([1.0] * 5, True)
        expect(ValueError, lambda: bad.affine_grid2d(2, 2))
        self.assertIsNone(bad.grad)


class AffineGrid2dBackwardTests(unittest.TestCase):
    def test_identity_2x2_all_ones(self):
        theta = Tensor(IDENTITY, True)
        theta.affine_grid2d(2, 2).backward([1.0] * 8)
        # Points: (-1,-1),(1,-1),(-1,1),(1,1).
        # sum x = sum y = 0; translation receives the count 4.
        self.assertEqual(theta.grad, [0.0, 0.0, 4.0, 0.0, 0.0, 4.0])

    def test_single_row_general_grad(self):
        # rows=1 -> y = 0; points (-1, 0), (1, 0).
        # Upstream pairs: (u0, v0) = (1, 2), (u1, v1) = (3, 4).
        theta = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], True)
        theta.affine_grid2d(1, 2).backward([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(theta.grad, [
            1.0 * -1.0 + 3.0 * 1.0,   # da00 = 2
            0.0,                      # da01
            4.0,                      # da02 = u0 + u1
            2.0 * -1.0 + 4.0 * 1.0,   # da10 = 2
            0.0,                      # da11
            6.0,                      # da12 = v0 + v1
        ])

    def test_single_point_gradient(self):
        theta = Tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], True)
        theta.affine_grid2d(1, 1).backward([2.0, 3.0])
        # x = y = 0: only translations receive (u, v).
        self.assertEqual(theta.grad, [0.0, 0.0, 2.0, 0.0, 0.0, 3.0])

    def test_backward_grad_validation(self):
        out = Tensor(IDENTITY, True).affine_grid2d(2, 2)
        expect(ValueError, out.backward)                  # omitted -> V
        expect(ValueError, lambda: out.backward(1.0))     # float -> V
        expect(ValueError, lambda: out.backward(True))
        expect(ValueError, lambda: out.backward(1))
        expect(ValueError, lambda: out.backward([1.0] * 7))   # length 7
        expect(ValueError, lambda: out.backward([1.0] * 9))   # length 9
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward("x"))
        expect(TypeError, lambda: out.backward(tuple([1.0] * 8)))
        expect(TypeError, lambda: out.backward([1] * 8))
        expect(TypeError, lambda: out.backward([True] * 8))
        expect(TypeError, lambda: out.backward(["x"] * 8))
        expect(ValueError, lambda: out.backward([float("nan")] * 8))
        expect(ValueError, lambda: out.backward([float("inf")] * 8))

    def test_graphless_result_rejects_any_grad(self):
        out = Tensor(IDENTITY).affine_grid2d(2, 2)
        self.assertFalse(out.requires_grad)
        # No graph: every grad, valid or not, is a ValueError.
        for bad in (None, "x", tuple([1.0] * 8), True, 1, 1.0):
            expect(ValueError, lambda bad=bad: out.backward(bad))
        expect(ValueError, out.backward)
        expect(ValueError, lambda: out.backward([1.0] * 8))

    def test_invalid_grad_changes_no_grad(self):
        theta = Tensor(IDENTITY, True)
        out = theta.affine_grid2d(2, 2)
        expect(ValueError, lambda: out.backward(1.0))
        self.assertIsNone(theta.grad)
        expect(TypeError, lambda: out.backward(None))
        self.assertIsNone(theta.grad)
        # A valid pass still works after the failed ones.
        out.backward([1.0] * 8)
        self.assertEqual(theta.grad, [0.0, 0.0, 4.0, 0.0, 0.0, 4.0])

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        theta = Tensor(IDENTITY, True)
        out = theta.affine_grid2d(2, 2)
        theta.data = [9.0] * 6
        out.backward([1.0] * 8)
        self.assertEqual(theta.grad, [0.0, 0.0, 4.0, 0.0, 0.0, 4.0])

    def test_repeated_backward_accumulates(self):
        theta = Tensor(IDENTITY, True)
        out = theta.affine_grid2d(2, 2)
        out.backward([1.0] * 8)
        out.backward([1.0] * 8)
        self.assertEqual(theta.grad, [0.0, 0.0, 8.0, 0.0, 0.0, 8.0])

    def test_separate_graphs_accumulate(self):
        theta = Tensor(IDENTITY, True)
        theta.affine_grid2d(2, 2).backward([1.0] * 8)
        theta.affine_grid2d(2, 2).backward([1.0] * 8)
        self.assertEqual(theta.grad, [0.0, 0.0, 8.0, 0.0, 0.0, 8.0])

    def test_nonfinite_backward_leaves_grads_untouched(self):
        huge = sys_float_max()
        # da02 sums nine copies of `huge`, which overflows; the failed
        # pass changes no grad.
        theta = Tensor(IDENTITY, True)
        out = theta.affine_grid2d(3, 3)
        expect(ValueError, lambda: out.backward([huge] * 18))
        self.assertIsNone(theta.grad)
        # A subsequent valid pass still works.
        out.backward([1.0] * 18)
        self.assertEqual(theta.grad, [0.0, 0.0, 9.0, 0.0, 0.0, 9.0])


class AffineGrid2dGradcheckTests(unittest.TestCase):
    def test_jacobian_matches_numerical(self):
        theta = [0.7, -0.3, 0.2, 0.5, 0.1, -0.4]
        passed, error = jacobiancheck(
            lambda t: t.affine_grid2d(3, 2),
            theta, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobian_singleton_rows(self):
        theta = [1.2, -0.8, 0.3, -0.4, 0.6, 0.1]
        passed, error = jacobiancheck(
            lambda t: t.affine_grid2d(1, 4),
            theta, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobian_singleton_cols(self):
        theta = [1.2, -0.8, 0.3, -0.4, 0.6, 0.1]
        passed, error = jacobiancheck(
            lambda t: t.affine_grid2d(4, 1),
            theta, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_jacobian_single_point(self):
        theta = [1.2, -0.8, 0.3, -0.4, 0.6, 0.1]
        passed, error = jacobiancheck(
            lambda t: t.affine_grid2d(1, 1),
            theta, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


if __name__ == "__main__":
    unittest.main()
