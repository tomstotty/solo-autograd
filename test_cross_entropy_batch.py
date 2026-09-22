"""Verifiable contract tests for Tensor.cross_entropy_batch.

Standard library only; discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck, jacobiancheck


def ref_batch(xs, targets, classes):
    """Independent hand computation following the prescribed order."""
    batch = len(xs) // classes
    rows_l = []
    total = 0.0
    for b in range(batch):
        row = xs[b * classes:(b + 1) * classes]
        m = max(row)
        z = [math.exp(v - m) for v in row]
        s = sum(z)
        l = [v - m - math.log(s) for v in row]
        rows_l.append(l)
        total += -l[targets[b]]
    return total / batch, rows_l


def ref_grad(xs, targets, classes, g):
    _, rows_l = ref_batch(xs, targets, classes)
    batch = len(rows_l)
    dx = []
    for b, l in enumerate(rows_l):
        for i, l_i in enumerate(l):
            indicator = 1.0 if i == targets[b] else 0.0
            dx.append(g * (math.exp(l_i) - indicator) / batch)
    return dx


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


class ForwardValueTest(unittest.TestCase):

    def test_single_class_is_zero(self):
        out = Tensor([3.7, -2.0]).cross_entropy_batch([0, 0], 1)
        self.assertEqual(out.data, 0.0)

    def test_uniform_logits_hand_values(self):
        out = Tensor([0.0, 0.0, 0.0, 0.0]).cross_entropy_batch([0, 1], 2)
        self.assertEqual(out.data, math.log(2.0))

    def test_hand_computed_batches(self):
        cases = [
            ([2.0, 0.0, 1.0, 0.3, -1.5, 2.0], [0, 1], 3),
            ([0.3, -1.5, 2.0, 5.5, -7.25, 0.125], [2, 0], 3),
            ([5.5, -7.25, 0.125, -0.375, 1.0, 2.0, 0.0, -1.0], [3, 0], 4),
            ([1e-12, -1e-12, 0.0, -0.0], [0, 1], 2),
        ]
        for xs, targets, classes in cases:
            with self.subTest(xs=xs, targets=targets, classes=classes):
                expected, _ = ref_batch(xs, targets, classes)
                out = Tensor(xs).cross_entropy_batch(targets, classes)
                self.assertEqual(out.data, expected)
                self.assertNotIsInstance(out.data, list)

    def test_matches_single_cross_entropy_for_one_row(self):
        xs = [0.3, -1.5, 2.0]
        batched = Tensor(xs).cross_entropy_batch([1], 3).data
        single = Tensor(xs).cross_entropy(1).data
        self.assertEqual(batched, single)

    def test_large_logits_stable(self):
        out = Tensor(
            [1000.0, -1000.0, -1000.0, 1000.0]
        ).cross_entropy_batch([0, 1], 2)
        self.assertTrue(math.isfinite(out.data))
        self.assertEqual(out.data, 0.0)
        out = Tensor(
            [1000.0, -1000.0, -1000.0, 1000.0]
        ).cross_entropy_batch([1, 0], 2)
        self.assertEqual(out.data, 2000.0)

    def test_gradcheck(self):
        xs = [0.3, -1.5, 2.0, -0.2, 1.1, 0.4]
        for targets, classes in (([1, 2], 3), ([2, 0], 3), ([0, 1], 3)):
            with self.subTest(targets=targets):
                passed, err = gradcheck(
                    lambda t, tg=targets, c=classes:
                        t.cross_entropy_batch(tg, c),
                    xs,
                )
                self.assertTrue(passed, f"max abs error {err}")

    def test_gradcheck_large_logits(self):
        passed, err = gradcheck(
            lambda t: t.cross_entropy_batch([0, 1], 2),
            [1000.0, -1000.0, -1000.0, 1000.0],
        )
        self.assertTrue(passed, f"max abs error {err}")


class ValidationTest(unittest.TestCase):

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy_batch([0], 1)

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).cross_entropy_batch([0], 1)

    def test_data_precedes_classes_and_targets(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy_batch("x", True)

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).cross_entropy_batch([0], 2)

    def test_requires_grad_type(self):
        for flag in (1, "yes"):
            with self.subTest(flag=flag):
                with self.assertRaises(TypeError):
                    make_bad(requires_grad=flag).cross_entropy_batch([0], 1)

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan")], [-float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_bad(data=bad).cross_entropy_batch([0], 1)

    def test_classes_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (True, False, 2.0, 0.5, "2", None, [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch([0], bad)

    def test_classes_positive(self):
        x = Tensor([0.3, 0.4])
        for bad in (0, -1, -10):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy_batch([0], bad)

    def test_length_divisibility(self):
        x = Tensor([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            x.cross_entropy_batch([0], 2)

    def test_targets_type(self):
        x = Tensor([0.3, 0.4, 0.5, 0.6])
        for bad in (None, True, 1, (0, 0), "00", 0):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch(bad, 2)

    def test_target_element_type(self):
        x = Tensor([0.3, 0.4, 0.5, 0.6])
        for bad in ([0, True], [0, 1.0], [None, 0], [0, "1"], [[0], 0]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch(bad, 2)

    def test_targets_length(self):
        x = Tensor([0.3, 0.4, 0.5, 0.6])
        with self.assertRaises(ValueError):
            x.cross_entropy_batch([0], 2)
        with self.assertRaises(ValueError):
            x.cross_entropy_batch([0, 0, 0], 2)

    def test_targets_range(self):
        x = Tensor([0.3, 0.4, 0.5, 0.6])
        for bad in ([-1, 0], [0, 2], [3, 0]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy_batch(bad, 2)


class BackwardTest(unittest.TestCase):

    def test_backward_uniform(self):
        x = Tensor([0.0, 0.0, 0.0, 0.0], True)
        x.cross_entropy_batch([0, 1], 2).backward()
        self.assertEqual(x.grad, [-0.25, 0.25, 0.25, -0.25])

    def test_backward_matches_reference(self):
        xs, targets, classes = [0.3, -1.5, 2.0, -0.2, 1.1, 0.4], [1, 2], 3
        x = Tensor(xs, True)
        x.cross_entropy_batch(targets, classes).backward()
        self.assertEqual(x.grad, ref_grad(xs, targets, classes, 1.0))

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5]).cross_entropy_batch([0], 2)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward()

    def test_finite_float_upstream(self):
        xs, targets, classes = [0.3, -1.5, 2.0, -0.2, 1.1, 0.4], [2, 0], 3
        x = Tensor(xs, True)
        x.cross_entropy_batch(targets, classes).backward(2.5)
        self.assertEqual(x.grad, ref_grad(xs, targets, classes, 2.5))

    def test_large_logits_backward(self):
        x = Tensor(
            [1000.0, -1000.0, -1000.0, 1000.0], True
        )
        x.cross_entropy_batch([1, 0], 2).backward()
        self.assertEqual(x.grad, [0.5, -0.5, -0.5, 0.5])

    def test_snapshot_survives_mutation_and_replacement(self):
        xs, targets, classes = [0.3, -1.5, 2.0, -0.2, 1.1, 0.4], [1, 2], 3
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        x.data[0] = 999.0
        x.data = [9.0] * 6
        out.backward()
        expected, _ = ref_batch(xs, targets, classes)
        self.assertEqual(out.data, expected)
        self.assertEqual(x.grad, ref_grad(xs, targets, classes, 1.0))

    def test_targets_snapshot_survives_mutation(self):
        xs, targets, classes = [0.3, -1.5, 2.0, -0.2, 1.1, 0.4], [1, 2], 3
        x = Tensor(xs, True)
        target_copy = list(targets)
        out = x.cross_entropy_batch(targets, classes)
        targets[0] = 2
        targets.append(0)
        out.backward()
        self.assertEqual(x.grad, ref_grad(xs, target_copy, classes, 1.0))

    def test_shared_path_accumulates(self):
        xs, targets, classes = [0.3, -1.5, 0.2, 1.0], [0, 1], 2
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        out.add(out).backward()
        self.assertEqual(x.grad, ref_grad(xs, targets, classes, 2.0))

    def test_repeated_backward_accumulates(self):
        xs, targets, classes = [0.3, -1.5, 0.2, 1.0], [0, 1], 2
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        out.backward()
        out.backward()
        self.assertEqual(x.grad, ref_grad(xs, targets, classes, 2.0))


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5, 0.2, 1.0], True)
        self.out = self.x.cross_entropy_batch([0, 1], 2)

    def test_upstream_type_errors(self):
        for bad in (None, True, False, 1, "x", (1.0,), object()):
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
            x.cross_entropy_batch([0], 2)
        self.assertEqual(x.data, original)
        self.assertIsNone(x.grad)

    def test_non_finite_upstream_is_atomic(self):
        x = Tensor(
            [1000.0, -1000.0, -1000.0, 1000.0], True
        )
        out = x.cross_entropy_batch([1, 0], 2)
        with self.assertRaises(ValueError):
            out.backward(float("inf"))
        self.assertIsNone(x.grad)
        out.backward()
        self.assertEqual(x.grad, [0.5, -0.5, -0.5, 0.5])

    def test_existing_grad_merge_overflow_is_atomic(self):
        x = Tensor(
            [1000.0, -1000.0, -1000.0, 1000.0], True
        )
        out = x.cross_entropy_batch([1, 0], 2)
        x.grad = [1.5e308, -1.5e308, 1.5e308, -1.5e308]
        with self.assertRaises(ValueError):
            # Each contribution is +/-7.5e307; merging with the
            # pre-existing 1.5e308 grad overflows and must leave every
            # grad untouched.
            out.backward(1.5e308)
        self.assertEqual(x.grad, [1.5e308, -1.5e308, 1.5e308, -1.5e308])
        x.zero_grad()
        out.backward()
        self.assertEqual(x.grad, [0.5, -0.5, -0.5, 0.5])


class JacobianCheckNoneGradTest(unittest.TestCase):

    def test_none_analytic_grad_is_type_error(self):
        # The fn's graph never involves the input tensor, so its analytic
        # grad is None; that must surface as TypeError, not ValueError.
        with self.assertRaises(TypeError):
            jacobiancheck(
                lambda t: Tensor([1.0]).add(Tensor([2.0], True)),
                [0.5],
            )


if __name__ == "__main__":
    unittest.main()
