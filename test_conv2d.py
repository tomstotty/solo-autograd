"""Contract tests for Tensor.conv2d, including dilated cross-correlation.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

from autograd import Tensor, gradcheck


REPO_ROOT = Path(__file__).resolve().parent
AUTOGRAD = REPO_ROOT / "autograd.py"


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


class Conv2dForwardTests(unittest.TestCase):
    def test_basic_3x3_kernel_identity_window(self):
        # 4x4 input, 3x3 all-ones kernel: sums of each 3x3 window.
        x = Tensor([float(v) for v in range(16)])
        k = Tensor([1.0] * 9)
        out = x.conv2d(k, 4, 4, 3)
        # window at (0,0): 0+1+2+4+5+6+8+9+10 = 45
        self.assertEqual(out.data[0], 45.0)
        self.assertEqual(out.data, [45.0, 54.0, 81.0, 90.0])
        self.assertEqual(len(out.data), 4)

    def test_stride_and_padding(self):
        # 3x3 input, 2x2 kernel, stride 2, padding 1
        x = Tensor([1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0])
        k = Tensor([1.0, 0.0, 0.0, 1.0])
        # OH = (3+2-2)//2+1 = 2. In-range taps, in ascending (ir, ic):
        # (0,0): x[0]*1 = 1
        # (0,1): taps (1,0)->x[1]*0, (1,1)->x[2]*1 = 3
        # (1,0): (0,1)->x[3]*0, (1,1)->x[6]*1 = 7
        # (1,1): all four taps in range -> x[4]*1 + x[8]*1 = 14
        out = x.conv2d(k, 3, 3, 2, stride=2, padding=1)
        self.assertEqual(out.data, [1.0, 3.0, 7.0, 14.0])

    def test_default_stride_padding_dilation_is_dilation_one(self):
        x = Tensor([float(v) for v in range(16)])
        k = Tensor([float(v) for v in range(1, 10)])
        explicit = x.conv2d(k, 4, 4, 3, 1, 0, 1)
        defaults = x.conv2d(k, 4, 4, 3)
        self.assertEqual(explicit.data, defaults.data)

    def test_dilation_one_matches_undilated_formula(self):
        x = Tensor([0.5, -1.0, 2.0, 3.0,
                    1.5, -2.0, 0.0, 4.0,
                    -0.5, 1.0, 2.5, -3.0,
                    2.0, 0.0, -1.0, 1.0])
        k = Tensor([1.0, -1.0, 2.0,
                    0.5, 1.0, -2.0,
                    -1.0, 1.5, 1.0])
        out = x.conv2d(k, 4, 4, 3, dilation=1)
        self.assertEqual(len(out.data), 4)
        # hand-check the first output
        expected0 = (
            0.5 * 1.0 + -1.0 * -1.0 + 2.0 * 2.0
            + 1.5 * 0.5 + -2.0 * 1.0 + 0.0 * -2.0
            + -0.5 * -1.0 + 1.0 * 1.5 + 2.5 * 1.0
        )
        self.assertAlmostEqual(out.data[0], expected0)

    def test_dilated_sparse_kernel(self):
        # 5x5 input, a 2x2 kernel with dilation 2 has effective span 3 and
        # touches the four corners of each 3x3 region.
        x = Tensor([float(r * 5 + c) for r in range(5) for c in range(5)])
        k = Tensor([1.0, 10.0, 100.0, 1000.0])
        out = x.conv2d(k, 5, 5, 2, dilation=2)
        # OH = OW = (5-3)//1+1 = 3
        self.assertEqual(len(out.data), 9)
        # output (0,0): x[0]*1 + x[2]*10 + x[10]*100 + x[12]*1000
        self.assertEqual(out.data[0], 0.0 + 20.0 + 1000.0 + 12000.0)
        # output (1,1): x[6]*1 + x[8]*10 + x[16]*100 + x[18]*1000
        self.assertEqual(out.data[4], 6.0 + 80.0 + 1600.0 + 18000.0)
        # output (2,2): x[12]*1 + x[14]*10 + x[22]*100 + x[24]*1000
        self.assertEqual(out.data[8], 12.0 + 140.0 + 2200.0 + 24000.0)

    def test_dilation_with_stride_and_padding(self):
        # 4x4 input, 2x2 kernel, dilation 2 -> span 3; stride 2, padding 1
        # OH = (4+2-3)//2+1 = 2
        x = Tensor([float(v) for v in range(16)])
        k = Tensor([1.0, 1.0, 1.0, 1.0])
        out = x.conv2d(k, 4, 4, 2, stride=2, padding=1, dilation=2)
        self.assertEqual(len(out.data), 4)
        # (0,0): sampled rows (-1,1) cols (-1,1): only (1,1)=5 in range
        # (0,1): rows (-1,1) cols (1,3): (1,1)=5,(1,3)=7 -> 12
        # (1,0): rows (1,3) cols (-1,1): (1,1)=5,(3,1)=13 -> 18
        # (1,1): rows (1,3) cols (1,3): 5+7+13+15 = 40
        self.assertEqual(out.data, [5.0, 12.0, 18.0, 40.0])

    def test_dilation_one_output_shape_unchanged(self):
        x = Tensor([1.0] * 16)
        k = Tensor([1.0] * 4)
        out = x.conv2d(k, 4, 4, 2, dilation=1)
        self.assertEqual(len(out.data), 9)

    def test_accumulation_starts_from_zero_in_ascending_order(self):
        # 2x2 input and image, 2x2 kernel, single output. Place 1e16 at the
        # first visited position so the remaining 1.0 taps are swallowed in
        # the prescribed (or, oc, ir, ic) accumulation order.
        huge = 1e16
        x = _raw([huge, 1.0, 1.0, 1.0], False)
        k = _raw([1.0, 0.0, 0.0, 0.0], False)
        out = x.conv2d(k, 2, 2, 2)
        self.assertEqual(out.data, [huge])

    def test_no_graph_without_requires_grad(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        k = Tensor([1.0, 0.0, 0.0, 1.0])
        out = x.conv2d(k, 2, 2, 2)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)

    def test_graph_built_when_either_parent_requires_grad(self):
        x = Tensor([1.0] * 9)
        k = Tensor([1.0] * 4, True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        self.assertEqual(len(out.data), 1)
        self.assertIs(out.requires_grad, True)
        self.assertEqual(out._parents, (x, k))
        self.assertIsNotNone(out._backward_fn)


class Conv2dValidationTests(unittest.TestCase):
    def test_kernel_must_be_tensor(self):
        x = Tensor([1.0] * 4)
        expect(TypeError, lambda: x.conv2d([1.0, 0.0, 0.0, 1.0], 2, 2, 2))
        expect(TypeError, lambda: x.conv2d("k", 2, 2, 2))
        expect(TypeError, lambda: x.conv2d(None, 2, 2, 2))

    def test_data_must_be_nonempty_float_vectors(self):
        good_k = Tensor([1.0] * 4)
        expect(ValueError, lambda: _raw(1.0, False).conv2d(good_k, 2, 2, 2))
        expect(ValueError, lambda: _raw([], False).conv2d(good_k, 2, 2, 2))
        expect(TypeError, lambda: _raw([1.0, 2], False).conv2d(good_k, 2, 2, 2))
        expect(TypeError,
               lambda: _raw([1.0, True, 0.0, 0.0], False).conv2d(
                   good_k, 2, 2, 2))
        expect(ValueError,
               lambda: _raw([1.0, float("nan"), 0.0, 0.0], False).conv2d(
                   good_k, 2, 2, 2))
        good_x = Tensor([1.0] * 4)
        expect(ValueError, lambda: good_x.conv2d(_raw([], False), 2, 2, 2))
        expect(TypeError, lambda: good_x.conv2d(_raw([1.0, 2, 3, 4], False),
                                                2, 2, 2))
        expect(ValueError,
               lambda: good_x.conv2d(
                   _raw([float("inf"), 0.0, 0.0, 0.0], False), 2, 2, 2))

    def test_dimensions_must_be_positive_ints(self):
        x = Tensor([1.0] * 4)
        k = Tensor([1.0] * 4)
        for kwargs in (
            dict(height=True, width=2),
            dict(height=2.0, width=2),
            dict(height="2", width=2),
            dict(height=2, width=False),
            dict(width=2.0),
            dict(kernel_size=True),
            dict(kernel_size=2.0),
            dict(stride=True),
            dict(stride=1.0),
        ):
            merged = dict(height=2, width=2, kernel_size=2)
            merged.update(kwargs)
            expect(TypeError, lambda m=merged: x.conv2d(k, **m))
        for kwargs in (
            dict(height=0, width=2),
            dict(height=-1, width=2),
            dict(height=2, width=0),
            dict(width=-2),
            dict(kernel_size=0),
            dict(kernel_size=-3),
            dict(stride=0),
            dict(stride=-2),
        ):
            merged = dict(height=2, width=2, kernel_size=2)
            merged.update(kwargs)
            expect(ValueError, lambda m=merged: x.conv2d(k, **m))

    def test_padding_must_be_non_negative_int(self):
        x = Tensor([1.0] * 4)
        k = Tensor([1.0] * 4)
        for bad in (True, False, 1.0, "0", None):
            expect(TypeError,
                   lambda bad=bad: x.conv2d(k, 2, 2, 2, padding=bad))
        expect(ValueError, lambda: x.conv2d(k, 2, 2, 2, padding=-1))

    def test_dilation_must_be_positive_non_bool_int(self):
        x = Tensor([1.0] * 9)
        k = Tensor([1.0])
        for bad in (True, False, 1.0, 2.0, "1", None, [1]):
            expect(TypeError,
                   lambda bad=bad: x.conv2d(k, 3, 3, 1, dilation=bad))
        for bad in (0, -1, -2):
            expect(ValueError,
                   lambda bad=bad: x.conv2d(k, 3, 3, 1, dilation=bad))

    def test_length_mismatches(self):
        k = Tensor([1.0] * 4)
        expect(ValueError,
               lambda: Tensor([1.0] * 9).conv2d(k, 2, 2, 2))
        x = Tensor([1.0] * 4)
        expect(ValueError,
               lambda: x.conv2d(Tensor([1.0] * 9), 2, 2, 2))

    def test_non_positive_output_dimensions_are_value_error(self):
        x = Tensor([1.0] * 4)
        k = Tensor([1.0] * 9)
        # 2x2 image with a 3x3 kernel: OH = 0
        expect(ValueError, lambda: x.conv2d(k, 2, 2, 3))
        # dilation pushes the effective span past the image: span 5 > 4
        expect(ValueError, lambda: x.conv2d(Tensor([1.0] * 4), 4, 4, 2,
                                            dilation=4))
        # kernel wider than the image with no padding
        expect(ValueError, lambda: x.conv2d(Tensor([1.0] * 4), 2, 2, 3))
        # one valid position works (a single-tap kernel always covers (0,0))
        x3 = Tensor([1.0] * 9)
        out = x3.conv2d(Tensor([1.0]), 3, 3, 1, stride=3)
        self.assertEqual(out.data, [1.0])
        # dilation 2 with a 2x2 kernel has span 3, leaving exactly one
        # output position on a 3x3 image
        out = x3.conv2d(Tensor([1.0] * 4), 3, 3, 2, dilation=2)
        self.assertEqual(out.data, [4.0])
        # the width dimension alone can also drive OW to zero
        expect(ValueError,
               lambda: Tensor([1.0] * 6).conv2d(k, 3, 2, 3))

    def test_forward_nonfinite_intermediate_is_value_error(self):
        huge = sys_float_max()
        # four huge terms at the four taps overflow when summed
        x = _raw([huge, 1.0, 1.0, huge], True)
        k = _raw([1.0, 0.0, 0.0, 1.0], True)
        expect(ValueError, lambda: x.conv2d(k, 2, 2, 2))

    def test_forward_failure_is_atomic_no_output(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        k = Tensor([1.0, 0.0, 0.0, 1.0], True)
        # kernel length mismatch raises before any graph exists; inputs are
        # untouched
        expect(ValueError,
               lambda: x.conv2d(Tensor([1.0] * 9), 2, 2, 2, dilation=2))
        self.assertEqual(x.data, [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(x.grad)
        self.assertIsNone(k.grad)
        # non-finite abort likewise leaves inputs untouched
        huge = sys_float_max()
        bx = _raw([huge, 1.0, 1.0, huge], True)
        expect(ValueError, lambda: bx.conv2d(k, 2, 2, 2))
        self.assertIsNone(bx.grad)


class Conv2dBackwardTests(unittest.TestCase):
    def test_basic_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        k = Tensor([1.0, 0.0, 0.0, 1.0], True)
        out = x.conv2d(k, 2, 2, 2)
        self.assertEqual(out.data, [5.0])
        out.backward([1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(k.grad, [1.0, 2.0, 3.0, 4.0])

    def test_backward_with_padding_and_stride(self):
        x = Tensor([1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0], True)
        k = Tensor([1.0, 0.0, 0.0, 1.0], True)
        out = x.conv2d(k, 3, 3, 2, stride=2, padding=1)
        self.assertEqual(out.data, [1.0, 3.0, 7.0, 14.0])
        out.backward([1.0, 2.0, 3.0, 4.0])
        # dx picks up only the (ir=1,ic=1) kernel tap (weight 1)
        self.assertEqual(
            x.grad,
            [1.0, 0.0, 2.0,
             0.0, 4.0, 0.0,
             3.0, 0.0, 4.0],
        )
        # dk[q] sums g[o]*x at each tap; only output (0,0) uses tap (1,1),
        # so contributions per tap follow the enumerated in-range positions
        self.assertEqual(k.grad, [20.0, 36.0, 36.0, 64.0])

    def test_dilated_backward(self):
        x = Tensor([float(v) for v in range(25)], True)
        k = Tensor([1.0, 10.0, 100.0, 1000.0], True)
        out = x.conv2d(k, 5, 5, 2, dilation=2)
        self.assertEqual(len(out.data), 9)
        g = [float(v) for v in range(1, 10)]
        out.backward(g)
        # Reference accumulation in the same ascending (or, oc, ir, ic) order.
        dx = [0.0] * 25
        dk = [0.0] * 4
        H = W = 5
        K = 2
        S = 1
        P = 0
        D = 2
        OW = OH = 3
        for orow in range(OH):
            for ocol in range(OW):
                go = g[orow * OW + ocol]
                for ir in range(K):
                    row = orow * S + ir * D - P
                    if not 0 <= row < H:
                        continue
                    for ic in range(K):
                        col = ocol * S + ic * D - P
                        if not 0 <= col < W:
                            continue
                        j = row * W + col
                        q = ir * K + ic
                        dx[j] += go * k.data[q]
                        dk[q] += go * x.data[j]
        self.assertEqual(x.grad, dx)
        self.assertEqual(k.grad, dk)

    def test_only_parents_requiring_grad_receive_contributions(self):
        x = Tensor([float(v) for v in range(9)])
        k = Tensor([1.0, 10.0, 100.0, 1000.0], True)
        out = x.conv2d(k, 3, 3, 2, dilation=2)
        self.assertEqual(len(out.data), 1)
        out.backward([2.0])
        self.assertIsNone(x.grad)
        # the only output samples the four corners with dilation 2
        self.assertEqual(k.grad, [0.0, 4.0, 12.0, 16.0])

    def test_aliased_parents_merge(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        # same tensor as both input and kernel; roles must merge into one
        # grad contribution.
        out = x.conv2d(x, 2, 2, 2)
        # forward: sum of x[j]**2 over the four taps = 1+4+9+16 = 30
        self.assertEqual(out.data, [30.0])
        out.backward([1.0])
        # input-role grad and kernel-role grad are both [1,2,3,4]; merged
        # they give 2*x = [2,4,6,8]
        self.assertEqual(x.grad, [2.0, 4.0, 6.0, 8.0])

    def test_data_mutation_after_forward_does_not_affect_backward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        k = Tensor([1.0, 0.0, 0.0, 1.0], True)
        out = x.conv2d(k, 2, 2, 2, dilation=1)
        x.data = [100.0, 100.0, 100.0, 100.0]
        k.data = [0.0, 0.0, 0.0, 0.0]
        out.backward([1.0])
        self.assertEqual(x.grad, [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(k.grad, [1.0, 2.0, 3.0, 4.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        k = Tensor([1.0, 0.0, 0.0, 1.0], True)
        out = x.conv2d(k, 2, 2, 2)
        out.backward([1.0])
        out.backward([2.0])
        self.assertEqual(x.grad, [3.0, 0.0, 0.0, 3.0])
        self.assertEqual(k.grad, [3.0, 6.0, 9.0, 12.0])

    def test_backward_nonfinite_is_value_error_and_atomic(self):
        huge = sys_float_max()
        # 3x3 image, 2x2 all-ones kernel: the forward is finite, but the
        # center input receives contributions from four output positions.
        x = Tensor([1.0] * 9, True)
        k = Tensor([1.0] * 4, True)
        out = x.conv2d(k, 3, 3, 2)
        expect(ValueError,
               lambda: out.backward([huge, huge, huge, huge]))
        # the failed pass must not publish any grads
        self.assertIsNone(x.grad)
        self.assertIsNone(k.grad)

    def test_backward_nonfinite_accumulation_leaves_existing_grad(self):
        huge = sys_float_max()
        # 3x3 image with a 2x2 stride-1 kernel gives a 2x2 output; the center
        # input position receives all four output contributions.
        x = Tensor([1.0] * 9, True)
        k = Tensor([1.0] * 4, True)
        out = x.conv2d(k, 3, 3, 2)
        self.assertEqual(out.data, [4.0, 4.0, 4.0, 4.0])
        out.backward([1.0, 1.0, 1.0, 1.0])
        # second pass with huge grads overflows when the center accumulates
        # its second contribution; the existing grad must survive untouched
        expect(ValueError,
               lambda: out.backward([huge, huge, huge, huge]))
        self.assertEqual(
            x.grad,
            [1.0, 2.0, 1.0,
             2.0, 4.0, 2.0,
             1.0, 2.0, 1.0],
        )
        self.assertEqual(k.grad, [4.0, 4.0, 4.0, 4.0])

    def test_gradcheck_dilation_one(self):
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 0.2, -0.8, 1.1]
        kernel = [0.4, -0.3, 0.9, -0.6]

        def fn(d):
            k = Tensor(kernel)
            return d.conv2d(k, 3, 3, 2).sum()

        passed, error = gradcheck(fn, data, eps=1e-6, atol=1e-4)
        self.assertTrue(passed, f"error={error}")

    def test_gradcheck_dilated(self):
        data = [0.3, -1.2, 2.4, 0.7, -0.5, 3.1, 0.2, -0.8,
                1.1, 0.6, -2.0, 0.4, 1.3, -0.9, 0.5, 2.2]
        kernel = [0.4, -0.3, 0.9, -0.6]

        def fn(d):
            k = Tensor(kernel)
            return d.conv2d(k, 4, 4, 2, dilation=2).sum()

        passed, error = gradcheck(fn, data, eps=1e-6, atol=1e-4)
        self.assertTrue(passed, f"error={error}")


class Conv2dCliRegressionTests(unittest.TestCase):
    """The conv2d change must not alter the documented CLI behavior."""

    def _run(self, payload):
        proc = subprocess.run(
            [sys.executable, str(AUTOGRAD), "tensor"],
            input=json.dumps(payload) + "\n",
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return proc

    def test_tensor_roundtrip(self):
        proc = self._run({"data": [1.0, 2.5, -3.0]})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["data"], [1.0, 2.5, -3.0])

    def test_tensor_requires_grad_roundtrip(self):
        proc = self._run({"data": [1.0, 2.0], "requires_grad": True})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertIs(result["requires_grad"], True)

    def test_invalid_payload_exits_two(self):
        proc = self._run({"data": [1.0, float("nan")]})
        self.assertEqual(proc.returncode, 2)
        self.assertNotEqual(proc.stderr, "")

    def test_usage_error_on_unknown_command(self):
        proc = subprocess.run(
            [sys.executable, str(AUTOGRAD), "conv2d"],
            input="",
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("usage:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
