"""Verifiable contract tests for Tensor.binary_cross_entropy_with_logits.

Standard library only; discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def ref_bce(xs, ys):
    """Independent hand computation of the stable BCE-with-logits mean."""
    total = 0.0
    for x, y in zip(xs, ys):
        total += max(x, 0.0) - x * y + math.log1p(math.exp(-abs(x)))
    return total / len(xs)


def ref_sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ref_dx(x, y, grad, n):
    return grad * (ref_sigmoid(x) - y) / n


def ref_dy(x, grad, n):
    return -grad * x / n


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


class ForwardValueTest(unittest.TestCase):

    def test_single_element_exact(self):
        # max(0,0) - 0*0.5 + log1p(exp(0)) == log(2)
        out = Tensor([0.0]).binary_cross_entropy_with_logits(Tensor([0.5]))
        self.assertEqual(out.data, math.log(2.0))

    def test_hand_computed_vectors(self):
        cases = [
            ([0.3, -1.5, 2.0], [0.2, 0.7, 1.0]),
            ([-0.0, 0.0], [0.0, 1.0]),
            ([5.5, -7.25, 0.125, -0.375], [0.9, 0.1, 0.5, 0.25]),
            ([1e-12, -1e-12], [1.0, 0.0]),
        ]
        for xs, ys in cases:
            with self.subTest(xs=xs, ys=ys):
                out = Tensor(xs).binary_cross_entropy_with_logits(Tensor(ys))
                self.assertEqual(out.data, ref_bce(xs, ys))

    def test_target_boundaries_allowed(self):
        out = Tensor([0.4, -0.4]).binary_cross_entropy_with_logits(
            Tensor([0.0, 1.0])
        )
        self.assertEqual(out.data, ref_bce([0.4, -0.4], [0.0, 1.0]))

    def test_large_logits_stable(self):
        # exp() of a large-magnitude logit must never be evaluated.
        out = Tensor([1000.0, -1000.0]).binary_cross_entropy_with_logits(
            Tensor([1.0, 0.0])
        )
        self.assertTrue(math.isfinite(out.data))
        self.assertEqual(out.data, 0.0)
        out = Tensor([1000.0]).binary_cross_entropy_with_logits(Tensor([0.0]))
        self.assertEqual(out.data, 1000.0)
        out = Tensor([-1000.0]).binary_cross_entropy_with_logits(
            Tensor([1.0])
        )
        self.assertEqual(out.data, 1000.0)

    def test_gradcheck_self(self):
        target = Tensor([0.2, 0.7, 1.0])
        passed, err = gradcheck(
            lambda t: t.binary_cross_entropy_with_logits(target),
            [0.3, -1.5, 2.0],
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_target(self):
        logits = Tensor([0.3, -1.5, 2.0])
        passed, err = gradcheck(
            lambda t: logits.binary_cross_entropy_with_logits(t),
            [0.2, 0.7, 0.9],
        )
        self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_large_logits(self):
        target = Tensor([1.0, 0.0])
        passed, err = gradcheck(
            lambda t: t.binary_cross_entropy_with_logits(target),
            [1000.0, -1000.0],
        )
        self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_self_only(self):
        x = Tensor([0.3, -1.5], True)
        t = Tensor([0.2, 0.7])
        out = x.binary_cross_entropy_with_logits(t)
        self.assertTrue(out.requires_grad)
        out.backward()
        self.assertEqual(
            x.grad, [ref_dx(0.3, 0.2, 1.0, 2), ref_dx(-1.5, 0.7, 1.0, 2)]
        )
        self.assertIsNone(t.grad)

    def test_backward_target_only(self):
        x = Tensor([0.3, -1.5])
        t = Tensor([0.2, 0.7], True)
        x.binary_cross_entropy_with_logits(t).backward()
        self.assertIsNone(x.grad)
        self.assertEqual(t.grad, [ref_dy(0.3, 1.0, 2), ref_dy(-1.5, 1.0, 2)])

    def test_backward_both(self):
        x = Tensor([0.3, -1.5, 2.0], True)
        t = Tensor([0.2, 0.7, 1.0], True)
        x.binary_cross_entropy_with_logits(t).backward()
        self.assertEqual(
            x.grad,
            [ref_dx(a, b, 1.0, 3) for a, b in zip([0.3, -1.5, 2.0],
                                                  [0.2, 0.7, 1.0])],
        )
        self.assertEqual(
            t.grad, [ref_dy(a, 1.0, 3) for a in [0.3, -1.5, 2.0]]
        )

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5]).binary_cross_entropy_with_logits(
            Tensor([0.2, 0.7])
        )
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        self.assertEqual(out.data, ref_bce([0.3, -1.5], [0.2, 0.7]))
        with self.assertRaises(ValueError):
            out.backward()

    def test_finite_float_upstream(self):
        x = Tensor([0.3, -1.5], True)
        t = Tensor([0.2, 0.7], True)
        x.binary_cross_entropy_with_logits(t).backward(2.5)
        self.assertEqual(
            x.grad, [ref_dx(0.3, 0.2, 2.5, 2), ref_dx(-1.5, 0.7, 2.5, 2)]
        )
        self.assertEqual(
            t.grad, [ref_dy(0.3, 2.5, 2), ref_dy(-1.5, 2.5, 2)]
        )

    def test_same_tensor_both_sides(self):
        x = Tensor([0.25, 0.75], True)
        out = x.binary_cross_entropy_with_logits(x)
        self.assertEqual(out.data, ref_bce([0.25, 0.75], [0.25, 0.75]))
        out.backward()
        self.assertEqual(
            x.grad,
            [
                ref_dx(0.25, 0.25, 1.0, 2) + ref_dy(0.25, 1.0, 2),
                ref_dx(0.75, 0.75, 1.0, 2) + ref_dy(0.75, 1.0, 2),
            ],
        )

    def test_shared_path_accumulates(self):
        x = Tensor([0.3, -1.5], True)
        t = Tensor([0.2, 0.7], True)
        out = x.binary_cross_entropy_with_logits(t)
        out.add(out).backward()
        self.assertEqual(
            x.grad, [ref_dx(0.3, 0.2, 2.0, 2), ref_dx(-1.5, 0.7, 2.0, 2)]
        )
        self.assertEqual(
            t.grad, [ref_dy(0.3, 2.0, 2), ref_dy(-1.5, 2.0, 2)]
        )

    def test_repeated_backward_accumulates(self):
        x = Tensor([0.3, -1.5], True)
        t = Tensor([0.2, 0.7], True)
        out = x.binary_cross_entropy_with_logits(t)
        out.backward()
        out.backward()
        self.assertEqual(
            x.grad, [ref_dx(0.3, 0.2, 2.0, 2), ref_dx(-1.5, 0.7, 2.0, 2)]
        )
        self.assertEqual(
            t.grad, [ref_dy(0.3, 2.0, 2), ref_dy(-1.5, 2.0, 2)]
        )

    def test_snapshot_survives_mutation_and_replacement(self):
        x = Tensor([0.3, -1.5], True)
        t = Tensor([0.2, 0.7], True)
        out = x.binary_cross_entropy_with_logits(t)
        x.data[0] = 999.0            # in-place mutation after forward
        t.data = [0.0, 0.0]          # wholesale replacement after forward
        out.backward()
        self.assertEqual(
            x.grad, [ref_dx(0.3, 0.2, 1.0, 2), ref_dx(-1.5, 0.7, 1.0, 2)]
        )
        self.assertEqual(
            t.grad, [ref_dy(0.3, 1.0, 2), ref_dy(-1.5, 1.0, 2)]
        )

    def test_large_logits_backward(self):
        x = Tensor([1000.0, -1000.0], True)
        t = Tensor([1.0, 0.0], True)
        x.binary_cross_entropy_with_logits(t).backward()
        self.assertEqual(x.grad, [0.0, 0.0])
        self.assertEqual(t.grad, [-500.0, 500.0])


class ErrorContractTest(unittest.TestCase):

    def test_target_not_tensor(self):
        x = Tensor([0.3])
        for bad in (None, True, 1, 0.5, "0.5", [0.5], (0.5,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.binary_cross_entropy_with_logits(bad)

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).binary_cross_entropy_with_logits(Tensor([0.5]))
        with self.assertRaises(ValueError):
            Tensor([0.3]).binary_cross_entropy_with_logits(
                make_bad(data=0.5)
            )

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).binary_cross_entropy_with_logits(Tensor([0.5]))
        with self.assertRaises(ValueError):
            Tensor([0.3]).binary_cross_entropy_with_logits(make_bad(data=[]))

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).binary_cross_entropy_with_logits(
                        Tensor([0.5, 0.5])
                    )
                with self.assertRaises(TypeError):
                    Tensor([0.3, 0.4]).binary_cross_entropy_with_logits(
                        make_bad(data=bad)
                    )

    def test_requires_grad_type(self):
        with self.assertRaises(TypeError):
            make_bad(requires_grad=1).binary_cross_entropy_with_logits(
                Tensor([0.5])
            )
        with self.assertRaises(TypeError):
            Tensor([0.3]).binary_cross_entropy_with_logits(
                make_bad(requires_grad="yes")
            )

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan")], [-float("inf")]):
            with self.subTest(bad=bad):
                n = len(bad)
                with self.assertRaises(ValueError):
                    make_bad(data=bad).binary_cross_entropy_with_logits(
                        Tensor([0.5] * n)
                    )
                with self.assertRaises(ValueError):
                    Tensor([0.3] * n).binary_cross_entropy_with_logits(
                        make_bad(data=bad)
                    )

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([0.3, 0.4]).binary_cross_entropy_with_logits(
                Tensor([0.5])
            )
        with self.assertRaises(ValueError):
            Tensor([0.3]).binary_cross_entropy_with_logits(
                Tensor([0.5, 0.6])
            )

    def test_target_out_of_range(self):
        for bad_y in (-1e-9, -0.5, 1.0000000001, 2.0):
            with self.subTest(bad_y=bad_y):
                with self.assertRaises(ValueError):
                    Tensor([0.3]).binary_cross_entropy_with_logits(
                        Tensor([bad_y])
                    )


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5], True)
        self.t = Tensor([0.2, 0.7], True)
        self.out = self.x.binary_cross_entropy_with_logits(self.t)

    def test_upstream_type_errors(self):
        for bad in (None, True, 1, "x", (1.0,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.out.backward(bad)

    def test_upstream_list_for_scalar_out(self):
        with self.assertRaises(ValueError):
            self.out.backward([1.0])

    def test_upstream_non_finite(self):
        with self.assertRaises(ValueError):
            self.out.backward(float("inf"))
        with self.assertRaises(ValueError):
            self.out.backward(float("nan"))

    def test_failed_upstream_leaves_no_grad(self):
        for bad in (None, True, 1, [1.0], float("inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    self.out.backward(bad)
                self.assertIsNone(self.x.grad)
                self.assertIsNone(self.t.grad)


class AtomicityTest(unittest.TestCase):

    def test_forward_intermediate_overflow(self):
        # 1e308 + 1e308 overflows the running total.
        x = Tensor([1e308, 1e308], True)
        t = Tensor([0.0, 0.0], True)
        with self.assertRaises(ValueError):
            x.binary_cross_entropy_with_logits(t)
        self.assertIsNone(x.grad)
        self.assertIsNone(t.grad)
        self.assertEqual(x.data, [1e308, 1e308])
        self.assertEqual(t.data, [0.0, 0.0])

    def test_backward_intermediate_overflow_is_atomic(self):
        # dy = -grad * x / n overflows for grad = x = 1e308.
        x = Tensor([1e308], True)
        t = Tensor([0.5], True)
        out = x.binary_cross_entropy_with_logits(t)
        out.backward(1.0)
        before = (list(x.grad), list(t.grad), out.grad)
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertEqual(x.grad, before[0])
        self.assertEqual(t.grad, before[1])
        self.assertEqual(out.grad, before[2])

    def test_existing_grad_merge_overflow_is_atomic(self):
        # sigmoid(1000) - 0.0 == 1.0, so dx = 1e308; merging with the
        # pre-existing 1e308 grad overflows.
        x = Tensor([1000.0], True)
        t = Tensor([0.0])
        out = x.binary_cross_entropy_with_logits(t)
        x.grad = [1e308]
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertEqual(x.grad, [1e308])
        self.assertIsNone(t.grad)


if __name__ == "__main__":
    unittest.main()
