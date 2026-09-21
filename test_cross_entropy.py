"""Verifiable contract tests for Tensor.cross_entropy with label smoothing.

Standard library only; discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def ref_cross_entropy(xs, target, alpha):
    """Independent hand computation following the prescribed operation order.

    m = max(data), z_i = exp(x_i - m), s = sum(z),
    l_i = x_i - m - log(s); q_i = alpha/n plus (1 - alpha) at the target;
    L is accumulated from 0.0 in ascending index order as L -= q_i * l_i.
    """
    m = max(xs)
    z = [math.exp(x_i - m) for x_i in xs]
    s = sum(z)
    log_s = math.log(s)
    l = [x_i - m - log_s for x_i in xs]
    n = len(xs)
    q = [alpha / n] * n
    q[target] = q[target] + (1.0 - alpha)
    total = 0.0
    for i in range(n):
        total -= q[i] * l[i]
    return total, l, q


def ref_grad(l, q, g):
    return [g * (math.exp(l_i) - q_i) for l_i, q_i in zip(l, q)]


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


class ForwardValueTest(unittest.TestCase):

    def test_single_element_is_zero(self):
        # Softmax of one logit is 1, so the loss is 0 for every alpha.
        for alpha in (0.0, 0.1, 0.5, 0.999):
            with self.subTest(alpha=alpha):
                out = Tensor([3.7]).cross_entropy(0, alpha)
                self.assertEqual(out.data, 0.0)

    def test_uniform_logits_hand_values(self):
        self.assertEqual(Tensor([0.0, 0.0]).cross_entropy(0).data, math.log(2.0))
        total, _, _ = ref_cross_entropy([0.0, 0.0], 1, 0.5)
        self.assertEqual(Tensor([0.0, 0.0]).cross_entropy(1, 0.5).data, total)
        self.assertEqual(total, math.log(2.0))

    def test_hand_computed_vectors(self):
        cases = [
            ([2.0, 0.0, 1.0], 0, 0.0),
            ([2.0, 0.0, 1.0], 0, 0.1),
            ([0.3, -1.5, 2.0], 1, 0.5),
            ([-0.0, 0.0], 0, 0.999),
            ([5.5, -7.25, 0.125, -0.375], 3, 0.25),
            ([1e-12, -1e-12], 0, 0.0),
        ]
        for xs, target, alpha in cases:
            with self.subTest(xs=xs, target=target, alpha=alpha):
                total, _, _ = ref_cross_entropy(xs, target, alpha)
                out = Tensor(xs).cross_entropy(target, alpha)
                self.assertEqual(out.data, total)

    def test_default_label_smoothing_matches_hard_target(self):
        # label_smoothing defaults to 0.0 and preserves the prior behavior.
        xs = [0.3, -1.5, 2.0]
        explicit = Tensor(xs).cross_entropy(2, 0.0).data
        default = Tensor(xs).cross_entropy(2).data
        total, l, _ = ref_cross_entropy(xs, 2, 0.0)
        self.assertEqual(default, explicit)
        self.assertEqual(default, total)
        self.assertEqual(default, -l[2])

    def test_large_logits_stable(self):
        # exp() of a large positive raw logit must never be evaluated.
        out = Tensor([1000.0, -1000.0]).cross_entropy(0)
        self.assertTrue(math.isfinite(out.data))
        self.assertEqual(out.data, 0.0)
        out = Tensor([1000.0, -1000.0]).cross_entropy(1)
        self.assertEqual(out.data, 2000.0)
        total, _, _ = ref_cross_entropy([1000.0, -1000.0], 0, 0.2)
        out = Tensor([1000.0, -1000.0]).cross_entropy(0, 0.2)
        self.assertTrue(math.isfinite(out.data))
        self.assertEqual(out.data, total)
        self.assertEqual(out.data, 200.0)

    def test_gradcheck(self):
        for alpha in (0.0, 0.1, 0.5):
            with self.subTest(alpha=alpha):
                passed, err = gradcheck(
                    lambda t, a=alpha: t.cross_entropy(1, a),
                    [0.3, -1.5, 2.0],
                )
                self.assertTrue(passed, f"alpha={alpha} max abs error {err}")

    def test_gradcheck_large_logits(self):
        passed, err = gradcheck(
            lambda t: t.cross_entropy(0, 0.2),
            [1000.0, -1000.0],
        )
        self.assertTrue(passed, f"max abs error {err}")


class BackwardTest(unittest.TestCase):

    def test_backward_hard_target(self):
        x = Tensor([0.0, 0.0], True)
        x.cross_entropy(0).backward()
        self.assertEqual(x.grad, [-0.5, 0.5])

    def test_backward_smoothed(self):
        x = Tensor([0.0, 0.0], True)
        x.cross_entropy(0, 0.5).backward()
        self.assertEqual(x.grad, [-0.25, 0.25])

    def test_backward_matches_reference(self):
        xs, target, alpha = [0.3, -1.5, 2.0], 1, 0.1
        x = Tensor(xs, True)
        x.cross_entropy(target, alpha).backward()
        _, l, q = ref_cross_entropy(xs, target, alpha)
        self.assertEqual(x.grad, ref_grad(l, q, 1.0))

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5]).cross_entropy(0, 0.3)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward()

    def test_finite_float_upstream(self):
        xs, target, alpha = [0.3, -1.5, 2.0], 2, 0.4
        x = Tensor(xs, True)
        x.cross_entropy(target, alpha).backward(2.5)
        _, l, q = ref_cross_entropy(xs, target, alpha)
        self.assertEqual(x.grad, ref_grad(l, q, 2.5))

    def test_large_logits_backward(self):
        x = Tensor([1000.0, -1000.0], True)
        x.cross_entropy(0, 0.2).backward()
        _, l, q = ref_cross_entropy([1000.0, -1000.0], 0, 0.2)
        self.assertEqual(x.grad, ref_grad(l, q, 1.0))
        x.zero_grad()
        x.cross_entropy(1).backward()
        self.assertEqual(x.grad, [1.0, -1.0])

    def test_snapshot_survives_mutation_and_replacement(self):
        xs, target, alpha = [0.3, -1.5, 2.0], 1, 0.2
        x = Tensor(xs, True)
        out = x.cross_entropy(target, alpha)
        x.data[0] = 999.0            # in-place mutation after forward
        x.data = [9.0, 9.0, 9.0]     # wholesale replacement after forward
        out.backward()
        total, l, q = ref_cross_entropy(xs, target, alpha)
        self.assertEqual(out.data, total)
        self.assertEqual(x.grad, ref_grad(l, q, 1.0))

    def test_shared_path_accumulates(self):
        xs, target, alpha = [0.3, -1.5], 0, 0.3
        x = Tensor(xs, True)
        out = x.cross_entropy(target, alpha)
        out.add(out).backward()
        _, l, q = ref_cross_entropy(xs, target, alpha)
        self.assertEqual(x.grad, ref_grad(l, q, 2.0))

    def test_repeated_backward_accumulates(self):
        xs, target, alpha = [0.3, -1.5], 0, 0.3
        x = Tensor(xs, True)
        out = x.cross_entropy(target, alpha)
        out.backward()
        out.backward()
        _, l, q = ref_cross_entropy(xs, target, alpha)
        self.assertEqual(x.grad, ref_grad(l, q, 2.0))


class ErrorContractTest(unittest.TestCase):

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy(0)

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).cross_entropy(0)

    def test_data_not_list_passes_target_type_first(self):
        # The data precondition is checked before the target type.
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy(True)

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).cross_entropy(0)

    def test_requires_grad_type(self):
        with self.assertRaises(TypeError):
            make_bad(requires_grad=1).cross_entropy(0)
        with self.assertRaises(TypeError):
            make_bad(requires_grad="yes").cross_entropy(0)

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan")], [-float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_bad(data=bad).cross_entropy(0)

    def test_target_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (None, True, False, 1.0, 0.5, "0", [0], (0,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy(bad)

    def test_target_out_of_range(self):
        x = Tensor([0.3, 0.4, 0.5])
        for bad in (-1, 3, 100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy(bad)

    def test_label_smoothing_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (0, 1, True, False, "0.0", None, [0.0]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy(0, bad)

    def test_label_smoothing_range(self):
        x = Tensor([0.3, 0.4])
        for bad in (
            1.0,
            1.5,
            -0.1,
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy(0, bad)

    def test_target_range_checked_before_smoothing(self):
        with self.assertRaises(ValueError):
            Tensor([0.3]).cross_entropy(5, 1.0)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5], True)
        self.out = self.x.cross_entropy(0, 0.2)

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


class AtomicityTest(unittest.TestCase):

    def test_forward_failure_leaves_input_untouched(self):
        x = make_bad(data=[0.3, float("inf")], requires_grad=True)
        original = list(x.data)
        with self.assertRaises(ValueError):
            x.cross_entropy(0)
        self.assertEqual(x.data, original)
        self.assertIsNone(x.grad)

    def test_non_finite_upstream_is_atomic(self):
        # A non-finite upstream is rejected before any grad is written;
        # the graph stays usable afterwards.
        x = Tensor([1000.0, -1000.0], True)
        out = x.cross_entropy(1)
        with self.assertRaises(ValueError):
            out.backward(float("inf"))
        self.assertIsNone(x.grad)
        out.backward()
        self.assertEqual(x.grad, [1.0, -1.0])

    def test_existing_grad_merge_overflow_is_atomic(self):
        # The finite contribution dx = [1e308, -1e308] merged with a
        # pre-existing 1e308 grad overflows and must leave every grad
        # untouched; the graph stays usable afterwards.
        x = Tensor([1000.0, -1000.0], True)
        out = x.cross_entropy(1)
        x.grad = [1e308, -1e308]
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertEqual(x.grad, [1e308, -1e308])
        x.zero_grad()
        out.backward()
        self.assertEqual(x.grad, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
