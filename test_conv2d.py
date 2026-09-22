"""Contract tests for Tensor.conv2d, including dilated cross-correlation.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

import json
import math
import os
import subprocess
import sys
import unittest

from autograd import Tensor, gradcheck


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOGRAD_PATH = os.path.join(REPO_DIR, "autograd.py")


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


# A 3x3 image (1..9) and a 2x2 kernel [[1,2],[3,4]], both row-major.
X3 = [float(v) for v in range(1, 10)]
K2 = [1.0, 2.0, 3.0, 4.0]
# A 5x5 image (1..25) row-major.
X5 = [float(v) for v in range(1, 26)]


def sys_float_max():
    return 1.7976931348623157e308


class Conv2dConstructionTests(unittest.TestCase):
    def test_defaults_match_dilation_one_stride_one_padding_zero(self):
        out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2)
        self.assertEqual(out.data, [37.0, 47.0, 67.0, 77.0])
        # explicitly passing the defaults must be identical
        explicit = Tensor(X3).conv2d(
            Tensor(K2), 3, 3, 2, stride=1, padding=0, dilation=1
        )
        self.assertEqual(explicit.data, out.data)
        self.assertEqual(len(out.data), 2 * 2)

    def test_no_graph_without_requires_grad(self):
        out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)

    def test_graph_built_when_either_parent_requires_grad(self):
        out = Tensor(X3, True).conv2d(Tensor(K2), 3, 3, 2)
        self.assertIs(out.requires_grad, True)
        self.assertEqual(len(out._parents), 2)
        self.assertIsNotNone(out._backward_fn)

    def test_output_is_length_oh_times_ow_flat_tensor(self):
        # 3x3, K=2, stride=2 -> OH=OW=(3-2)//2+1=1
        out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2, stride=2)
        self.assertEqual(out.data, [37.0])
        # 5x5, K=2, stride=2 -> 2x2 output (only top-left windows used)
        out = Tensor(X5).conv2d(Tensor(K2), 5, 5, 2, stride=2)
        self.assertEqual(len(out.data), 4)


class Conv2dForwardTests(unittest.TestCase):
    def test_stride_two_with_padding(self):
        # H=W=3, K=2, S=2, P=1 -> OH=OW=(3+2-2)//2+1=2
        out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2, 2, 1)
        self.assertEqual(out.data, [4.0, 18.0, 36.0, 77.0])

    def test_dilation_basic_single_output(self):
        # K=2, D=2 -> E=3; 3x3 -> OH=OW=1; corners only:
        # 1*1 + 3*2 + 7*3 + 9*4 = 64
        out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2, dilation=2)
        self.assertEqual(out.data, [64.0])

    def test_dilation_with_stride_and_padding(self):
        # 5x5, K=2, D=2 (E=3), S=2, P=1 -> OH=OW=(5+2-3)//2+1=3
        out = Tensor(X5).conv2d(Tensor(K2), 5, 5, 2, 2, 1, 2)
        self.assertEqual(
            out.data,
            [28.0, 57.0, 27.0, 82.0, 152.0, 66.0, 34.0, 55.0, 19.0],
        )

    def test_dilation_two_equals_spaced_kernel(self):
        # K=3, D=2 on a 5x5 image yields a single output over the 3x3
        # lattice with spacing 2; identity kernel picks the four... here
        # only q=4 (center of the 3x3 filter) is 1.0, so the result is the
        # image center x[12] = 13.0.
        k = [0.0] * 9
        k[4] = 1.0
        out = Tensor(X5).conv2d(Tensor(k), 5, 5, 3, dilation=2)
        self.assertEqual(out.data, [13.0])

    def test_dilation_one_preserves_old_results_with_padding(self):
        # 3x3, K=2, S=1, P=1 -> 4x4 output; same result whether dilation
        # is omitted or explicitly 1.
        default_out = Tensor(X3).conv2d(Tensor(K2), 3, 3, 2, 1, 1)
        explicit_out = Tensor(X3).conv2d(
            Tensor(K2), 3, 3, 2, 1, 1, 1
        )
        self.assertEqual(len(default_out.data), 16)
        self.assertEqual(explicit_out.data, default_out.data)

    def test_accumulation_runs_from_zero_in_ascending_order(self):
        # K=3, D=2, 5x5 -> one output, 9 products in (ir, ic) ascending
        # order. The first product (ir=0, ic=0) is 1e16 and the other
        # eight are 1.0; the swallowed additions distinguish the mandated
        # order from any order summing the 1.0s first.
        x = [0.0] * 25
        for j in (0, 2, 4, 10, 12, 14, 20, 22, 24):
            x[j] = 1.0
        x[0] = 1e16
        out = Tensor(x).conv2d(Tensor([1.0] * 9), 5, 5, 3, dilation=2)
        self.assertEqual(out.data, [1e16])

    def test_forward_snapshots_both_operands(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        self.assertEqual(out.data, [64.0])
        # later caller-side mutation changes neither stored forward data
        # nor a pending backward pass
        x.data = [100.0] * 9
        k.data = [0.0] * 4
        self.assertEqual(out.data, [64.0])
        out.backward([1.0])
        # dx/dk computed from the snapshots, not the mutated data:
        # dx corners 1,2,3,4; dk = [1,3,7,9]
        self.assertEqual(
            x.grad, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0]
        )
        self.assertEqual(k.grad, [1.0, 3.0, 7.0, 9.0])


class Conv2dValidationTests(unittest.TestCase):
    def test_kernel_must_be_tensor(self):
        expect(TypeError, lambda: Tensor(X3).conv2d(K2, 3, 3, 2))

    def test_data_must_be_nonempty_finite_float_vectors(self):
        good_x = Tensor(X3)
        good_k = Tensor(K2)
        expect(ValueError,
               lambda: _raw(1.0, False).conv2d(good_k, 3, 3, 2))
        expect(ValueError,
               lambda: _raw([], False).conv2d(good_k, 3, 3, 2))
        expect(TypeError,
               lambda: _raw([1] * 9, False).conv2d(good_k, 3, 3, 2))
        expect(ValueError,
               lambda: good_x.conv2d(_raw([], False), 3, 3, 2))
        expect(TypeError,
               lambda: good_x.conv2d(_raw([1.0] * 4 + [True], False),
                                     3, 3, 2))

    def test_dimensions_must_be_positive_ints(self):
        x, k = Tensor(X3), Tensor(K2)
        for name, args in (
            ("height", (True, 3, 2, 1, 0, 1)),
            ("width", (3, True, 2, 1, 0, 1)),
            ("kernel_size", (3, 3, True, 1, 0, 1)),
            ("stride", (3, 3, 2, True, 0, 1)),
        ):
            with self.subTest(name):
                expect(TypeError,
                       lambda args=args: x.conv2d(k, *args))
        for bad_h in (0, -1):
            expect(ValueError, lambda bad_h=bad_h: x.conv2d(
                k, bad_h, 3, 2, 1, 0, 1))
        for bad_s in (0, -2):
            expect(ValueError, lambda bad_s=bad_s: x.conv2d(
                k, 3, 3, 2, bad_s, 0, 1))
        for bad in (1.0, 2.5, "2", None, [2]):
            expect(TypeError, lambda bad=bad: x.conv2d(
                k, 3, 3, 2, bad, 0, 1))

    def test_padding_must_be_non_negative_int(self):
        x, k = Tensor(X3), Tensor(K2)
        for bad in (True, False, 1.0, "0", None):
            expect(TypeError, lambda bad=bad: x.conv2d(
                k, 3, 3, 2, 1, bad, 1))
        expect(ValueError, lambda: x.conv2d(k, 3, 3, 2, 1, -1, 1))

    def test_dilation_must_be_non_bool_positive_int(self):
        x, k = Tensor(X3), Tensor(K2)
        for bad in (True, False, 1.0, 2.0, "2", None, [2]):
            expect(TypeError, lambda bad=bad: x.conv2d(
                k, 3, 3, 2, 1, 0, bad))
        for bad in (0, -1, -3):
            expect(ValueError, lambda bad=bad: x.conv2d(
                k, 3, 3, 2, 1, 0, bad))

    def test_input_and_kernel_lengths_must_match(self):
        x, k = Tensor(X3), Tensor(K2)
        expect(ValueError, lambda: x.conv2d(k, 4, 3, 2))
        expect(ValueError, lambda: x.conv2d(
            Tensor([1.0] * 9), 3, 3, 2))

    def test_non_positive_output_dimensions_are_value_error(self):
        x, k = Tensor(X3), Tensor(K2)
        # a 4x4 kernel (16 weights) does not fit a 3x3 image
        big_k = Tensor([1.0] * 16)
        expect(ValueError, lambda: x.conv2d(big_k, 3, 3, 4))
        # stride drops the only window: H=1,K=2,S=2 gives (-1)//2+1=0
        expect(ValueError, lambda: Tensor([1.0]).conv2d(
            Tensor([1.0] * 4), 1, 1, 2, 2))
        # dilation expands the receptive field past the image:
        # K=2, D=3 -> E=4 > 3
        expect(ValueError, lambda: x.conv2d(
            k, 3, 3, 2, 1, 0, 3))
        # height survives (3-3//1+1=1) but width does not (2-3//1+1=0)
        expect(ValueError, lambda: Tensor(X3[:6]).conv2d(
            k, 3, 2, 2, 1, 0, 2))

    def test_boundary_output_sizes_just_fit(self):
        x, k = Tensor(X3), Tensor(K2)
        # H=2, K=2, S=2 -> (0)//2+1 = 1: exactly one window
        out = Tensor(X3[:6]).conv2d(k, 2, 3, 2, 2)
        self.assertEqual(out.data, [37.0])
        # dilated field exactly covered, no padding: E=3 == H=3
        out = x.conv2d(k, 3, 3, 2, 1, 0, 2)
        self.assertEqual(out.data, [64.0])
        # padding makes a dilated conv fit with room around it
        out = x.conv2d(k, 3, 3, 2, 1, 1, 2)
        self.assertEqual(len(out.data), 9)

    def test_forward_nonfinite_product_raises_value_error(self):
        x = _raw([float("inf")] + X3[1:], False)
        expect(ValueError, lambda: x.conv2d(Tensor(K2), 3, 3, 2))

    def test_forward_nonfinite_accumulation_raises_value_error(self):
        # every product finite (~3.2e308) but adding two overflows
        huge = sys_float_max()
        x = _raw([huge, huge] + [0.0] * 7, False)
        k = _raw([1.0, 1.0, 0.0, 0.0], False)
        expect(ValueError, lambda: x.conv2d(k, 3, 3, 2))

    def test_failure_leaves_inputs_untouched(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        # dilation expands the K=2 field to E=4, wider than the 3x3 image
        expect(ValueError, lambda: x.conv2d(k, 3, 3, 2, 1, 0, 3))
        self.assertEqual(x.data, X3)
        self.assertEqual(k.data, K2)
        self.assertIsNone(x.grad)
        self.assertIsNone(k.grad)


class Conv2dBackwardTests(unittest.TestCase):
    def test_basic_gradients(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        x.conv2d(k, 3, 3, 2).backward([1.0] * 4)
        self.assertEqual(
            x.grad,
            [1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        )
        self.assertEqual(k.grad, [12.0, 16.0, 24.0, 28.0])

    def test_dilated_gradients(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        x.conv2d(k, 3, 3, 2, dilation=2).backward([1.0])
        self.assertEqual(
            x.grad,
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0],
        )
        self.assertEqual(k.grad, [1.0, 3.0, 7.0, 9.0])

    def test_dilated_strided_padded_gradients_all_ones(self):
        # 5x5, K=2, D=2, S=2, P=1 with an all-ones upstream grad: only
        # odd/odd image positions receive grads (10.0 each), and every
        # kernel entry sees the same four image values (sum 52).
        x = Tensor(X5, True)
        k = Tensor(K2, True)
        out = x.conv2d(k, 5, 5, 2, 2, 1, 2)
        self.assertEqual(len(out.data), 9)
        out.backward([1.0] * 9)
        expected_dx = [0.0] * 25
        for j in (6, 8, 16, 18):
            expected_dx[j] = 10.0
        self.assertEqual(x.grad, expected_dx)
        self.assertEqual(k.grad, [52.0, 52.0, 52.0, 52.0])

    def test_gradients_use_distinct_upstream_values(self):
        # single dilated output on 3x3 but pass g=3.0
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        x.conv2d(k, 3, 3, 2, dilation=2).backward([3.0])
        self.assertEqual(
            x.grad,
            [3.0, 0.0, 6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 12.0],
        )
        self.assertEqual(k.grad, [3.0, 9.0, 21.0, 27.0])

    def test_only_parents_requiring_grad_receive_contributions(self):
        # kernel frozen
        x = Tensor(X3, True)
        k = Tensor(K2, False)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        out.backward([1.0])
        self.assertEqual(
            x.grad,
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0],
        )
        self.assertIsNone(k.grad)
        # image frozen
        x = Tensor(X3, False)
        k = Tensor(K2, True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        out.backward([1.0])
        self.assertIsNone(x.grad)
        self.assertEqual(k.grad, [1.0, 3.0, 7.0, 9.0])

    def test_aliased_parents_merge_their_roles(self):
        z = Tensor([float(v) for v in range(1, 17)], True)
        out = z.conv2d(z, 4, 4, 4)
        self.assertEqual(out.data, [float(sum(v * v for v in range(1, 17)))])
        out.backward([1.0])
        # dx[q] = k[q] = q+1 and dk[q] = x[q] = q+1, merged elementwise
        self.assertEqual(z.grad, [2.0 * v for v in range(1, 17)])

    def test_repeated_backward_accumulates(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        out.backward([1.0])
        out.backward([2.0])
        self.assertEqual(
            x.grad,
            [3.0, 0.0, 6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 12.0],
        )
        self.assertEqual(k.grad, [3.0, 9.0, 21.0, 27.0])

    def test_dk_accumulates_in_ascending_order(self):
        # plain D=1 3x3 conv: each kernel entry collects four x values in
        # ascending output order. Make the first contributor 1e16 and the
        # rest 1.0; the later additions are swallowed.
        x = [1e16] + [1.0] * 8
        out = Tensor(x, True).conv2d(Tensor([1.0] * 4, True), 3, 3, 2)
        out.backward([1.0] * 4)
        # dk0 sees x00 (1e16) first, then three 1.0s
        self.assertEqual(out._parents[1].grad[0], 1e16)

    def test_nonfinite_backward_intermediate_is_atomic_failure(self):
        huge = sys_float_max()
        x = Tensor(X3, True)
        k = Tensor([huge, 0.0, 0.0, 0.0], True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        # g*k overflows on the first contribution
        expect(ValueError, lambda: out.backward([huge]))
        # no grad was written anywhere
        self.assertIsNone(x.grad)
        self.assertIsNone(k.grad)

    def test_nonfinite_merge_with_existing_grad_is_value_error(self):
        x = Tensor(X3, True)
        k = Tensor(K2, True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        x.grad = [float("inf")] * 9
        k.grad = [0.0] * 4
        expect(ValueError, lambda: out.backward([1.0]))
        # the failed merge leaves both existing grads exactly in place
        self.assertEqual(x.grad, [float("inf")] * 9)
        self.assertEqual(k.grad, [0.0] * 4)

    def test_gradcheck_dilated_convolution(self):
        k = Tensor(K2, False)
        passed, error = gradcheck(
            lambda d: d.conv2d(k, 3, 3, 2, dilation=2).sum(),
            X3, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_gradcheck_dilated_strided_padded(self):
        k = Tensor(K2, False)
        passed, error = gradcheck(
            lambda d: d.conv2d(k, 5, 5, 2, 2, 1, 2).sum(),
            X5, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


class Conv2dCliRegressionTests(unittest.TestCase):
    """The README CLI surface must keep behaving byte-for-byte."""

    def run_cli(self, args, stdin=""):
        return subprocess.run(
            [sys.executable, AUTOGRAD_PATH] + args,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_tensor_roundtrip_is_byte_stable(self):
        result = self.run_cli(
            ["tensor"], '{"data":[1.0,-2.5,0.0],"requires_grad":true}\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            '{"data":[1.000000,-2.500000,0.000000],'
            '"grad":null,"requires_grad":true}\n',
        )
        self.assertEqual(result.stderr, "")

    def test_tensor_invalid_json_exit_code(self):
        result = self.run_cli(["tensor"], "not json\n")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "invalid JSON input\n")
        self.assertEqual(result.stdout, "")

    def test_unknown_subcommand_prints_usage_and_fails(self):
        result = self.run_cli(["bogus"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("usage:", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
