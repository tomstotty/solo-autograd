"""Contract tests for Adagrad int/bool data rejection and atomic load.

Standard library only; discovered by ``python -m unittest discover``.
"""

import unittest

from autograd import Adagrad, Tensor, dump_adagrad, load_adagrad


def mutate(value):
    """A valid parameter whose data is later replaced by ``value``."""
    parameter = Tensor(
        [0.0] * len(value) if isinstance(value, list) else 0.0, True
    )
    parameter.data = value
    return parameter


class IntBoolDataTests(unittest.TestCase):

    def test_construction_rejects_int_and_bool_data(self):
        for bad in (2, True, False, [1, 2], [1.0, True], [True, False]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Adagrad([mutate(bad)], 0.1)

    def _make_vector(self):
        parameter = Tensor([0.1, 0.2], True)
        optimizer = Adagrad([parameter], 0.1)
        parameter.grad = [0.3, 0.4]
        return optimizer, parameter

    def test_step_rejects_int_and_bool_data(self):
        for bad in ([1, 2], [1.0, True], [True, False]):
            with self.subTest(bad=bad):
                optimizer, parameter = self._make_vector()
                slots = [list(optimizer.sum_sq[0])]
                parameter.data = bad
                with self.assertRaises(TypeError):
                    optimizer.step()
                # Failure is atomic: nothing moved.
                self.assertEqual(parameter.data, bad)
                self.assertEqual(optimizer.sum_sq, slots)
                self.assertEqual(parameter.grad, [0.3, 0.4])

    def test_step_rejects_scalar_int_data(self):
        parameter = Tensor(1.5, True)
        optimizer = Adagrad([parameter], 0.1)
        parameter.grad = 1.0
        parameter.data = 2
        with self.assertRaises(TypeError):
            optimizer.step()
        self.assertEqual(parameter.data, 2)
        self.assertEqual(optimizer.sum_sq, [0.0])

    def test_dump_rejects_int_and_bool_data(self):
        for bad in (2, True, [1, 2], [1.0, False]):
            with self.subTest(bad=bad):
                optimizer, parameter = self._make_vector()
                parameter.data = bad
                with self.assertRaises(TypeError):
                    dump_adagrad(optimizer)


class LoadAtomicityTests(unittest.TestCase):

    def setUp(self):
        self.parameter = Tensor([0.1, 0.2], True)
        self.optimizer = Adagrad([self.parameter], 0.1)
        self.parameter.grad = [0.3, 0.4]

    def test_invalid_text_is_value_error(self):
        for bad in (
            "",
            "{",
            "not json",
            "null",
            '{"parameters":[],"sum_sq":[]}',
            '{"parameters":[{"data":[0.1],"requires_grad":true}],'
            '"sum_sq":[]}',
            '{"parameters":[{"data":[0.1,0.2],"requires_grad":true}],'
            '"sum_sq":[[0.0]]}',
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_adagrad(self.optimizer, bad)

    def test_failed_load_changes_nothing(self):
        data_before = list(self.parameter.data)
        slots_before = list(self.optimizer.sum_sq[0])
        grad_before = list(self.parameter.grad)
        with self.assertRaises(ValueError):
            load_adagrad(self.optimizer, "{")
        self.assertEqual(self.parameter.data, data_before)
        self.assertEqual(self.optimizer.sum_sq[0], slots_before)
        self.assertEqual(self.parameter.grad, grad_before)

    def test_valid_round_trip_still_works(self):
        text = dump_adagrad(self.optimizer)
        target_parameter = Tensor([9.0, 9.0], True)
        target = Adagrad([target_parameter], 0.5)
        target.sum_sq = [[7.0, 7.0]]
        load_adagrad(target, text)
        self.assertEqual(target_parameter.data, [0.1, 0.2])
        self.assertEqual(target.sum_sq, [[0.0, 0.0]])
        self.assertIsNone(target_parameter.grad)
        self.assertEqual(target.lr, 0.5)


if __name__ == "__main__":
    unittest.main()
