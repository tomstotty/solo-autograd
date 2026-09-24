"""Contract tests for Tensor.conv1d_batch batched cross-correlation.

Only the standard library is used. Discovered via
``python -m unittest discover``.
"""

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


def sys_float_max():
    return 1.7976931348623157e308


# B=2 batches of C=2 channels of N=4 samples, O=2 filters of K=3 taps.
X = [float(v) for v in range(1, 17)]          # length B*C*N = 16
W = [float(v) / 10.0 for v in range(1, 13)]   # length O*C*K = 12


def _reference(x, w, B, C, O, N, K, S, P, D, grad=None):
    L = (N + 2 * P - D * (K - 1) - 1) // S + 1
    y = [0.0] * (B * O * L)
    dx = [0.0] * (B * C * N) if grad is not None else None
    dw = [0.0] * (O * C * K) if grad is not None else None
    for b in range(B):
        for o in range(O):
            for t in range(L):
                oi = (b * O + o) * L + t
                acc = 0.0
                for c in range(C):
                    for r in range(K):
                        j = t * S + r * D - P
                        if not 0 <= j < N:
                            continue
                        xi = (b * C + c) * N + j
                        wi = (o * C + c) * K + r
                        acc += x[xi] * w[wi]
                        if grad is not None:
                            dx[xi] += grad[oi] * w[wi]
                            dw[wi] += grad[oi] * x[xi]
                y[oi] = acc
    if grad is None:
        return y
    return y, dx, dw, L


class Conv1dBatchForwardTests(unittest.TestCase):
    def test_matches_reference(self):
        out = Tensor(X).conv1d_batch(Tensor(W), 2, 2, 2, 4, 3, padding=1)
        ref = _reference(X, W, 2, 2, 2, 4, 3, 1, 1, 1)
        self.assertEqual(out.data, ref)
        self.assertEqual(len(out.data), 2 * 2 * 4)

    def test_output_length_formula(self):
        # N=5, K=2, S=2, P=1, D=2 -> (5+2-2-1)//2+1 = 3
        out = Tensor([0.0] * 5).conv1d_batch(
            Tensor([0.0, 0.0]), 1, 1, 1, 5, 2,
            stride=2, padding=1, dilation=2,
        )
        self.assertEqual(len(out.data), 3)

    def test_stride_dilation_padding_values(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        w = Tensor([2.0, 3.0])
        out = x.conv1d_batch(
            w, 1, 1, 1, 5, 2, stride=2, padding=1, dilation=2
        )
        # t=0: j=1 -> 2*3=6; t=1: j=1 (4) + j=3 (12); t=2: j=3 -> 8
        self.assertEqual(out.data, [6.0, 16.0, 8.0])

    def test_accumulation_runs_from_zero_in_ascending_order(self):
        # one output, K taps; the first (ascending) product is 1e16 and
        # swallows all later 1.0 additions, pinning the mandated order.
        n = 4
        x = [1e16] + [1.0] * (n - 1)
        out = Tensor(x).conv1d_batch(Tensor([1.0] * n), 1, 1, 1, n, n)
        self.assertEqual(out.data, [1e16])

    def test_graph_presence(self):
        out = Tensor(X).conv1d_batch(Tensor(W), 2, 2, 2, 4, 3, padding=1)
        self.assertIs(out.requires_grad, False)
        self.assertEqual(out._parents, ())
        self.assertIsNone(out._backward_fn)
        expect(ValueError, out.backward)
        out = Tensor(X, True).conv1d_batch(
            Tensor(W), 2, 2, 2, 4, 3, padding=1
        )
        self.assertIs(out.requires_grad, True)
        self.assertEqual(len(out._parents), 2)
        self.assertIsNotNone(out._backward_fn)

    def test_forward_snapshots_both_operands(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        w = Tensor([1.0, 2.0], True)
        out = x.conv1d_batch(w, 1, 1, 1, 4, 2)
        saved = list(out.data)
        x.data = [100.0] * 4
        w.data = [0.0, 0.0]
        self.assertEqual(out.data, saved)
        out.backward([1.0] * 3)
        self.assertEqual(x.grad, [1.0, 3.0, 3.0, 2.0])
        self.assertEqual(w.grad, [6.0, 9.0])


class Conv1dBatchValidationTests(unittest.TestCase):
    def test_kernel_must_be_tensor(self):
        expect(
            TypeError,
            lambda: Tensor([1.0] * 4).conv1d_batch([1.0, 1.0], 1, 1, 1, 4, 2),
        )

    def test_data_must_be_nonempty_finite_float_vectors(self):
        good_w = Tensor([1.0, 1.0])
        expect(ValueError,
               lambda: _raw(1.0, False).conv1d_batch(good_w, 1, 1, 1, 4, 2))
        expect(ValueError,
               lambda: _raw([], False).conv1d_batch(good_w, 1, 1, 1, 4, 2))
        expect(TypeError,
               lambda: _raw([1] * 4, False).conv1d_batch(
                   good_w, 1, 1, 1, 4, 2))
        expect(TypeError,
               lambda: _raw([1.0, True, 1.0, 1.0], False).conv1d_batch(
                   good_w, 1, 1, 1, 4, 2))
        good_x = Tensor([1.0] * 4)
        expect(ValueError,
               lambda: good_x.conv1d_batch(_raw([], False), 1, 1, 1, 4, 2))
        expect(ValueError,
               lambda: good_x.conv1d_batch(
                   _raw([float("nan"), 1.0], False), 1, 1, 1, 4, 2))

    def test_requires_grad_must_be_bool(self):
        bad = _raw([1.0] * 4, False)
        bad.requires_grad = "yes"
        expect(TypeError,
               lambda: bad.conv1d_batch(Tensor([1.0, 1.0]), 1, 1, 1, 4, 2))

    def test_sizes_must_be_non_bool_positive_ints(self):
        x, w = Tensor([1.0] * 4), Tensor([1.0, 1.0])
        names = ("batch", "channels", "filters", "length", "size",
                 "stride", "padding", "dilation")
        # padding sits in the positional slot too; it is tested separately
        good = (1, 1, 1, 4, 2, 1, 0, 1)
        for i, name in enumerate(names):
            for bad in (True, 1.0, "1", None, [1]):
                args = list(good)
                args[i] = bad
                with self.subTest(name, bad=bad):
                    expect(TypeError,
                           lambda args=args: x.conv1d_batch(w, *args))
            if name == "padding":
                continue  # zero padding is valid
            args = list(good)
            args[i] = 0
            with self.subTest(name, bad="zero"):
                expect(ValueError,
                       lambda args=args: x.conv1d_batch(w, *args))

    def test_padding_must_be_non_bool_non_negative_int(self):
        x, w = Tensor([1.0] * 4), Tensor([1.0, 1.0])
        for bad in (True, False, 1.0, "0", None):
            expect(TypeError,
                   lambda bad=bad: x.conv1d_batch(
                       w, 1, 1, 1, 4, 2, padding=bad))
        expect(ValueError,
               lambda: x.conv1d_batch(w, 1, 1, 1, 4, 2, padding=-1))

    def test_data_lengths_must_match_shapes(self):
        x, w = Tensor([1.0] * 4), Tensor([1.0, 1.0])
        expect(ValueError,
               lambda: x.conv1d_batch(w, 2, 1, 1, 4, 2))
        expect(ValueError,
               lambda: x.conv1d_batch(Tensor([1.0] * 3), 1, 1, 1, 4, 2))

    def test_output_length_must_be_positive(self):
        x, w = Tensor([1.0] * 4), Tensor([1.0, 1.0])
        # dilation 5 spans 6 positions, wider than N=4: L = -1
        expect(ValueError,
               lambda: x.conv1d_batch(w, 1, 1, 1, 4, 2, dilation=5))

    def test_nonfinite_forward_is_value_error(self):
        x = _raw([float("inf"), 1.0, 1.0, 1.0], False)
        expect(ValueError,
               lambda: x.conv1d_batch(Tensor([1.0, 1.0]), 1, 1, 1, 4, 2))
        huge = sys_float_max()
        x = _raw([huge, huge, 0.0, 0.0], False)
        w = _raw([1.0, 1.0], False)
        expect(ValueError,
               lambda: x.conv1d_batch(w, 1, 1, 1, 4, 2))


class Conv1dBatchBackwardTests(unittest.TestCase):
    def test_gradients_match_reference(self):
        B, C, O, N, K, S, P, D = 2, 2, 2, 4, 3, 1, 1, 1
        x = Tensor(X, True)
        w = Tensor(W, True)
        out = x.conv1d_batch(w, B, C, O, N, K, stride=S, padding=P,
                             dilation=D)
        g = [((i % 5) + 1) * 0.5 for i in range(B * O * 4)]
        out.backward(list(g))
        _, ref_dx, ref_dw, _ = _reference(
            X, W, B, C, O, N, K, S, P, D, g
        )
        self.assertEqual(x.grad, ref_dx)
        self.assertEqual(w.grad, ref_dw)

    def test_dilated_strided_padded_gradients(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], True)
        w = Tensor([2.0, 3.0], True)
        out = x.conv1d_batch(
            w, 1, 1, 1, 5, 2, stride=2, padding=1, dilation=2
        )
        out.backward([1.0, 1.0, 1.0])
        self.assertEqual(x.grad, [0.0, 5.0, 0.0, 5.0, 0.0])
        self.assertEqual(w.grad, [6.0, 6.0])

    def test_dw_reduces_across_batch(self):
        x = Tensor([1.0, 2.0, 1.0, 2.0], True)
        w = Tensor([1.0, 1.0], True)
        out = x.conv1d_batch(w, 2, 1, 1, 2, 2)
        self.assertEqual(out.data, [3.0, 3.0])
        out.backward([1.0, 1.0])
        self.assertEqual(x.grad, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(w.grad, [2.0, 4.0])

    def test_only_parents_requiring_grad_receive_contributions(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        w = Tensor([1.0, 2.0], False)
        x.conv1d_batch(w, 1, 1, 1, 4, 2).backward([1.0] * 3)
        self.assertEqual(x.grad, [1.0, 3.0, 3.0, 2.0])
        self.assertIsNone(w.grad)
        x = Tensor([1.0, 2.0, 3.0, 4.0], False)
        w = Tensor([1.0, 2.0], True)
        x.conv1d_batch(w, 1, 1, 1, 4, 2).backward([1.0] * 3)
        self.assertIsNone(x.grad)
        self.assertEqual(w.grad, [6.0, 9.0])

    def test_aliased_parents_merge_roles(self):
        z = Tensor([1.0, 2.0, 3.0, 4.0], True)
        out = z.conv1d_batch(z, 1, 1, 1, 4, 4)
        self.assertEqual(out.data, [30.0])
        out.backward([1.0])
        self.assertEqual(z.grad, [2.0, 4.0, 6.0, 8.0])

    def test_repeated_backward_accumulates(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        w = Tensor([1.0, 2.0], True)
        out = x.conv1d_batch(w, 1, 1, 1, 4, 2)
        out.backward([1.0] * 3)
        out.backward([1.0] * 3)
        self.assertEqual(x.grad, [2.0, 6.0, 6.0, 4.0])
        self.assertEqual(w.grad, [12.0, 18.0])

    def test_grad_validation(self):
        out = Tensor([1.0] * 4, True).conv1d_batch(
            Tensor([1.0, 1.0], True), 1, 1, 1, 4, 2
        )
        # output length 3: omitted and scalar/bool/int/float -> ValueError,
        # wrong length or non-finite element -> ValueError
        for bad in ([], True, False, 1, 1.0, [1.0, 1.0],
                    [1.0, 1.0, float("nan")]):
            expect(ValueError, lambda bad=bad: out.backward(bad))
        # None and other non-lists, or non-float elements -> TypeError
        expect(TypeError, lambda: out.backward(None))
        expect(TypeError, lambda: out.backward({}))
        expect(TypeError, lambda: out.backward([1, 1, 1]))
        expect(ValueError, lambda: out.backward())

    def test_nonfinite_backward_is_atomic_failure(self):
        huge = sys_float_max()
        # ones keep every forward partial finite (a single huge addend);
        # g*w0 = huge*huge overflows only in the backward pass.
        x = Tensor([1.0] * 4, True)
        w = Tensor([huge, 0.0], True)
        out = x.conv1d_batch(w, 1, 1, 1, 4, 2)
        expect(ValueError, lambda: out.backward([huge, 0.0, 0.0]))
        self.assertIsNone(x.grad)
        self.assertIsNone(w.grad)

    def test_failed_merge_leaves_existing_grads(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], True)
        w = Tensor([1.0, 2.0], True)
        out = x.conv1d_batch(w, 1, 1, 1, 4, 2)
        x.grad = [float("inf")] * 4
        expect(ValueError, lambda: out.backward([1.0] * 3))
        self.assertEqual(x.grad, [float("inf")] * 4)
        self.assertIsNone(w.grad)

    def test_gradcheck_input(self):
        w = Tensor(W, False)
        passed, error = gradcheck(
            lambda d: d.conv1d_batch(
                w, 2, 2, 2, 4, 3, padding=1).sum(),
            X, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")

    def test_gradcheck_weight(self):
        x = Tensor(X, False)
        passed, error = gradcheck(
            lambda d: x.conv1d_batch(d, 2, 2, 2, 4, 3).sum(),
            W, eps=1e-6, atol=1e-5,
        )
        self.assertTrue(passed, f"error={error}")


class Conv1dBatchCliRegressionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
