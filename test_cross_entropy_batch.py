"""Verifiable contract tests for Tensor.cross_entropy_batch.

Batched mean hard-target cross entropy over a flattened B x classes
float list. Standard library only; discovered by
``python -m unittest discover``.
"""

import math
import unittest

from autograd import Tensor, gradcheck


def ref_batch(xs, targets, classes):
    """Independent hand computation in the prescribed operation order.

    For each row: m = max(row), z_i = exp(x_i - m), s = sum(z),
    l_i = x_i - m - log(s). L starts at 0.0 and subtracts l_target once
    per row in ascending row order, then the result is L / B.
    """
    B = len(xs) // classes
    l = []
    total = 0.0
    for q in range(B):
        row = xs[q * classes:(q + 1) * classes]
        m = max(row)
        z = [math.exp(x_i - m) for x_i in row]
        s = sum(z)
        row_l = [x_i - m - math.log(s) for x_i in row]
        l.extend(row_l)
        total -= row_l[targets[q]]
    return total / B, l


def ref_grad_batch(l, targets, classes, g):
    """g * (exp(l_i) - indicator(i == target)) / B for every element."""
    B = len(l) // classes
    grad = []
    for q in range(B):
        for i in range(classes):
            indicator = 1.0 if i == targets[q] else 0.0
            grad.append(
                g * (math.exp(l[q * classes + i]) - indicator) / B
            )
    return grad


def make_bad(data=None, requires_grad=None):
    """A valid Tensor mutated into an invalid state after construction."""
    t = Tensor([0.5])
    if data is not None:
        t.data = data
    if requires_grad is not None:
        t.requires_grad = requires_grad
    return t


class ForwardValueTest(unittest.TestCase):

    def test_single_class_rows_are_zero(self):
        out = Tensor([3.7, 9.1, -4.2]).cross_entropy_batch([0, 0, 0], 1)
        self.assertEqual(out.data, 0.0)

    def test_uniform_logits_hand_values(self):
        # Both rows are uniform 2-class rows, regardless of target.
        self.assertEqual(
            Tensor([0.0, 0.0, 0.0, 0.0]).cross_entropy_batch([0, 1], 2).data,
            math.log(2.0),
        )

    def test_hand_computed_batches(self):
        cases = [
            ([2.0, 0.0, 1.0, 0.3, -1.5, 2.0], [0, 2], 3),
            ([0.3, -1.5, 2.0], [1], 3),
            ([-0.0, 0.0, 5.5, -7.25], [0, 1], 2),
            ([5.5, -7.25, 0.125, -0.375, 1.0, 2.0, 3.0, 4.0], [3, 0], 4),
            ([1e-12, -1e-12], [0], 2),
        ]
        for xs, targets, classes in cases:
            with self.subTest(xs=xs, targets=targets, classes=classes):
                total, _ = ref_batch(xs, targets, classes)
                out = Tensor(xs).cross_entropy_batch(targets, classes)
                self.assertEqual(out.data, total)

    def test_mean_of_rows(self):
        # The batched mean equals the mean of single-row losses.
        rows = [[2.0, 0.0, 1.0], [0.3, -1.5, 2.0]]
        targets = [0, 2]
        xs = rows[0] + rows[1]
        batched = Tensor(xs).cross_entropy_batch(targets, 3).data
        single0 = Tensor(rows[0]).cross_entropy(targets[0]).data
        single1 = Tensor(rows[1]).cross_entropy(targets[1]).data
        self.assertEqual(batched, (single0 + single1) / 2.0)

    def test_large_logits_stable(self):
        out = Tensor(
            [1000.0, -1000.0, 1000.0, -1000.0]
        ).cross_entropy_batch([0, 1], 2)
        self.assertTrue(math.isfinite(out.data))
        self.assertEqual(out.data, 1000.0)
        out = Tensor([1000.0, -1000.0]).cross_entropy_batch([0], 2)
        self.assertEqual(out.data, 0.0)

    def test_output_is_scalar(self):
        out = Tensor([0.3, -1.5, 1.0, 0.2]).cross_entropy_batch([1, 0], 2)
        self.assertNotIsInstance(out.data, list)

    def test_gradcheck(self):
        cases = [
            ([0.3, -1.5, 2.0, -0.2, 0.7, 1.1], [1, 0], 3),
            ([0.3, -1.5, 1.0, 0.2], [1, 0], 2),
        ]
        for xs, targets, classes in cases:
            with self.subTest(xs=xs):
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


class BackwardTest(unittest.TestCase):

    def test_backward_uniform_rows(self):
        x = Tensor([0.0, 0.0, 0.0, 0.0], True)
        x.cross_entropy_batch([0, 1], 2).backward()
        self.assertEqual(x.grad, [-0.25, 0.25, 0.25, -0.25])

    def test_backward_matches_reference(self):
        xs, targets, classes = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4], [1, 2], 3
        x = Tensor(xs, True)
        x.cross_entropy_batch(targets, classes).backward()
        _, l = ref_batch(xs, targets, classes)
        self.assertEqual(x.grad, ref_grad_batch(l, targets, classes, 1.0))

    def test_no_grad_no_graph(self):
        out = Tensor([0.3, -1.5, 0.1, 0.2]).cross_entropy_batch([0, 1], 2)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        with self.assertRaises(ValueError):
            out.backward()

    def test_finite_float_upstream(self):
        xs, targets, classes = [0.3, -1.5, 2.0], [2], 3
        x = Tensor(xs, True)
        x.cross_entropy_batch(targets, classes).backward(2.5)
        _, l = ref_batch(xs, targets, classes)
        self.assertEqual(x.grad, ref_grad_batch(l, targets, classes, 2.5))

    def test_large_logits_backward(self):
        x = Tensor([1000.0, -1000.0, 1000.0, -1000.0], True)
        x.cross_entropy_batch([0, 1], 2).backward()
        # Softmax of each extreme row is [1.0, 0.0]; subtracting the
        # one-hot target leaves zero on the confident logit.
        self.assertEqual(x.grad, [0.0, 0.0, 0.5, -0.5])

    def test_snapshot_survives_mutation_and_replacement(self):
        xs, targets, classes = [0.3, -1.5, 2.0, 1.0, 0.2, -0.4], [1, 2], 3
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        x.data[0] = 999.0            # in-place mutation after forward
        x.data = [9.0] * 6          # wholesale replacement after forward
        out.backward()
        total, l = ref_batch(xs, targets, classes)
        self.assertEqual(out.data, total)
        self.assertEqual(x.grad, ref_grad_batch(l, targets, classes, 1.0))

    def test_targets_list_snapshot_survives_mutation(self):
        xs, targets, classes = [0.3, -1.5, 2.0, 1.0], [1, 0], 2
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        targets[0] = 0
        targets.append(3)
        out.backward()
        _, l = ref_batch(xs, [1, 0], classes)
        self.assertEqual(x.grad, ref_grad_batch(l, [1, 0], classes, 1.0))

    def test_shared_path_accumulates(self):
        xs, targets, classes = [0.3, -1.5, 1.0, 0.2], [0, 1], 2
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        out.add(out).backward()
        _, l = ref_batch(xs, targets, classes)
        self.assertEqual(x.grad, ref_grad_batch(l, targets, classes, 2.0))

    def test_repeated_backward_accumulates(self):
        xs, targets, classes = [0.3, -1.5, 1.0, 0.2], [0, 1], 2
        x = Tensor(xs, True)
        out = x.cross_entropy_batch(targets, classes)
        out.backward()
        out.backward()
        _, l = ref_batch(xs, targets, classes)
        self.assertEqual(x.grad, ref_grad_batch(l, targets, classes, 2.0))


class ErrorContractTest(unittest.TestCase):

    def test_data_not_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy_batch([0], 1)

    def test_data_empty_list(self):
        with self.assertRaises(ValueError):
            make_bad(data=[]).cross_entropy_batch([], 1)

    def test_data_not_list_checked_first(self):
        # The data precondition is checked before classes and targets.
        with self.assertRaises(ValueError):
            make_bad(data=0.5).cross_entropy_batch(True, "x")

    def test_element_type(self):
        for bad in ([0.3, 1], [0.3, True], [0.3, "x"], [0.3, None]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    make_bad(data=bad).cross_entropy_batch([0], 2)

    def test_requires_grad_type(self):
        with self.assertRaises(TypeError):
            make_bad(requires_grad=1).cross_entropy_batch([0], 1)
        with self.assertRaises(TypeError):
            make_bad(requires_grad="yes").cross_entropy_batch([0], 1)

    def test_non_finite_elements(self):
        for bad in ([0.3, float("inf")], [float("nan"), 0.1], [-float("inf")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_bad(data=bad).cross_entropy_batch(
                        [0] * len(bad), len(bad)
                    )

    def test_classes_type(self):
        x = Tensor([0.3, 0.4])
        for bad in (None, 1.0, 0.5, "2", [2], (2,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch([0], bad)

    def test_classes_bool_is_type_error(self):
        x = Tensor([0.3, 0.4])
        with self.assertRaises(TypeError):
            x.cross_entropy_batch([0], True)
        with self.assertRaises(TypeError):
            x.cross_entropy_batch([], False)

    def test_classes_non_positive(self):
        x = Tensor([0.3, 0.4])
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy_batch([0], bad)

    def test_length_must_be_divisible(self):
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2, 0.3]).cross_entropy_batch([0], 2)

    def test_targets_type(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in (None, True, False, 1, 1.0, "00", (0, 0), {0: 0}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch(bad, 2)

    def test_targets_length(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            x.cross_entropy_batch([0], 2)
        with self.assertRaises(ValueError):
            x.cross_entropy_batch([0, 0, 0], 2)

    def test_target_element_type(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in ([0, True], [False, 0], [0, 1.0], [None, 0], [0, "1"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    x.cross_entropy_batch(bad, 2)

    def test_target_out_of_range(self):
        x = Tensor([0.1, 0.2, 0.3, 0.4])
        for bad in ([-1, 0], [0, 2], [3, 0]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    x.cross_entropy_batch(bad, 2)

    def test_classes_checked_before_targets(self):
        # A bad classes type is a TypeError even with a malformed targets
        # list, and a non-positive classes is a ValueError even when the
        # targets length cannot match.
        with self.assertRaises(TypeError):
            Tensor([0.1, 0.2]).cross_entropy_batch("x", 1.0)
        with self.assertRaises(ValueError):
            Tensor([0.1, 0.2]).cross_entropy_batch([0, 1, 2], 0)


class UpstreamGradContractTest(unittest.TestCase):

    def setUp(self):
        self.x = Tensor([0.3, -1.5, 1.0, 0.2], True)
        self.out = self.x.cross_entropy_batch([0, 1], 2)

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
            x.cross_entropy_batch([0], 2)
        self.assertEqual(x.data, original)
        self.assertIsNone(x.grad)

    def test_non_finite_upstream_is_atomic(self):
        x = Tensor([1000.0, -1000.0], True)
        out = x.cross_entropy_batch([1], 2)
        with self.assertRaises(ValueError):
            out.backward(float("inf"))
        self.assertIsNone(x.grad)
        out.backward()
        self.assertEqual(x.grad, [1.0, -1.0])

    def test_existing_grad_merge_overflow_is_atomic(self):
        x = Tensor([1000.0, -1000.0], True)
        out = x.cross_entropy_batch([1], 2)
        x.grad = [1e308, -1e308]
        with self.assertRaises(ValueError):
            out.backward(1e308)
        self.assertEqual(x.grad, [1e308, -1e308])
        x.zero_grad()
        out.backward()
        self.assertEqual(x.grad, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
