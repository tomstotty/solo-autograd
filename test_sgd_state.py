"""Tests for dump_sgd / load_sgd: strict SGD parameter serialization."""

import math
import unittest

from autograd import Adam, RMSprop, SGD, Tensor, dump_sgd, load_sgd


def _make_optimizer():
    """Build a fresh two-parameter SGD optimizer (one scalar, one vector)."""
    return SGD(
        [Tensor(0.0, requires_grad=True), Tensor([1.0, -2.0], True)],
        0.25,
    )


# Deterministic grad schedule used by the continued-step tests; every value
# has an exact six-decimal representation so formatted dumps stay exact.
_GRADS = [
    (0.5, [1.0, -2.0]),
    (-0.25, [0.75, 0.5]),
    (1.25, [-1.5, 2.25]),
    (0.125, [0.375, -0.625]),
    (-1.0, [2.5, -0.5]),
]


def _apply_step(optimizer, grads):
    scalar, vector = optimizer.parameters
    scalar.grad, vector.grad = grads
    optimizer.step()


class DumpSgdFormatTests(unittest.TestCase):
    def test_exact_text_scalar_and_vector(self):
        optimizer = SGD(
            [Tensor(1.5), Tensor([1.0, -0.0], requires_grad=True)],
            0.1,
        )
        self.assertEqual(
            dump_sgd(optimizer),
            '{"parameters":['
            '{"data":1.500000,"requires_grad":false},'
            '{"data":[1.000000,0.000000],"requires_grad":true}'
            "]}",
        )

    def test_no_whitespace_or_trailing_newline(self):
        text = dump_sgd(_make_optimizer())
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\t", "\n", "\r"):
            self.assertNotIn(whitespace, text)

    def test_only_top_level_parameters_key(self):
        text = dump_sgd(_make_optimizer())
        self.assertTrue(text.startswith('{"parameters":['))
        self.assertTrue(text.endswith("]}"))
        self.assertNotIn('"lr"', text)
        self.assertNotIn('"grad"', text)

    def test_negative_zero_written_as_zero(self):
        optimizer = SGD([Tensor(-0.0)], 1.0)
        self.assertEqual(
            dump_sgd(optimizer),
            '{"parameters":[{"data":0.000000,"requires_grad":false}]}',
        )

    def test_floats_have_exactly_six_decimals(self):
        optimizer = SGD([Tensor([1.0 / 8.0, 2.0 / 3.0])], 0.5)
        text = dump_sgd(optimizer)
        self.assertIn("0.125000", text)
        self.assertIn("0.666667", text)


class RoundTripTests(unittest.TestCase):
    def test_round_trip_scalar_and_vector_values(self):
        optimizer = SGD(
            [Tensor(-3.5, requires_grad=False), Tensor([1.25, -2.75], True)],
            0.1,
        )
        load_sgd(optimizer, dump_sgd(optimizer))
        scalar, vector = optimizer.parameters
        self.assertEqual(scalar.data, -3.5)
        self.assertFalse(scalar.requires_grad)
        self.assertEqual(vector.data, [1.25, -2.75])
        self.assertTrue(vector.requires_grad)

    def test_load_clears_grads(self):
        optimizer = _make_optimizer()
        for parameter in optimizer.parameters:
            parameter.grad = 0.0 if isinstance(parameter.data, float) else [0.0, 0.0]
        load_sgd(optimizer, dump_sgd(optimizer))
        for parameter in optimizer.parameters:
            self.assertIsNone(parameter.grad)

    def test_load_preserves_list_identities_and_lr(self):
        optimizer = _make_optimizer()
        parameters = optimizer.parameters
        identities = [id(parameter) for parameter in parameters]
        original_lr = optimizer.lr
        load_sgd(optimizer, dump_sgd(optimizer))
        self.assertIs(optimizer.parameters, parameters)
        self.assertEqual([id(p) for p in optimizer.parameters], identities)
        self.assertEqual(optimizer.lr, original_lr)
        self.assertIsInstance(optimizer.lr, float)

    def test_load_returns_none(self):
        optimizer = _make_optimizer()
        self.assertIsNone(load_sgd(optimizer, dump_sgd(optimizer)))


class ParameterOrderTests(unittest.TestCase):
    def test_array_follows_parameters_order(self):
        a = Tensor(1.0)
        b = Tensor([2.0, 3.0])
        c = Tensor(-4.0)
        optimizer = SGD([a, b, c], 0.5)
        self.assertEqual(
            dump_sgd(optimizer),
            '{"parameters":['
            '{"data":1.000000,"requires_grad":false},'
            '{"data":[2.000000,3.000000],"requires_grad":false},'
            '{"data":-4.000000,"requires_grad":false}'
            "]}",
        )

    def test_entries_apply_positionally_not_by_value(self):
        source = SGD([Tensor(9.0), Tensor([8.0, 7.0])], 0.5)
        text = dump_sgd(source)
        target = SGD([Tensor(0.0), Tensor([0.0, 0.0])], 0.1)
        load_sgd(target, text)
        self.assertEqual(target.parameters[0].data, 9.0)
        self.assertEqual(target.parameters[1].data, [8.0, 7.0])

    def test_swapped_shapes_between_positions_is_value_error(self):
        source = SGD([Tensor(9.0), Tensor([8.0, 7.0])], 0.5)
        text = dump_sgd(source)
        target = SGD([Tensor([0.0, 0.0]), Tensor(0.0)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(target, text)


class ContinuedStepTests(unittest.TestCase):
    def test_dump_load_then_continue_is_byte_identical(self):
        uninterrupted = _make_optimizer()
        resumed = _make_optimizer()
        for grads in _GRADS[:3]:
            _apply_step(uninterrupted, grads)
        for grads in _GRADS[:1]:
            _apply_step(resumed, grads)

        snapshot = dump_sgd(uninterrupted)
        load_sgd(resumed, snapshot)
        # The resumed optimizer must reproduce the checkpoint exactly.
        self.assertEqual(dump_sgd(resumed), snapshot)

        for grads in _GRADS[3:]:
            _apply_step(uninterrupted, grads)
            _apply_step(resumed, grads)
        self.assertEqual(dump_sgd(resumed), dump_sgd(uninterrupted))

    def test_repeated_dump_after_load_is_byte_stable(self):
        optimizer = _make_optimizer()
        _apply_step(optimizer, _GRADS[0])
        text = dump_sgd(optimizer)
        for _ in range(3):
            load_sgd(optimizer, text)
            self.assertEqual(dump_sgd(optimizer), text)

    def test_step_after_load_matches_uninterrupted_values(self):
        uninterrupted = _make_optimizer()
        resumed = _make_optimizer()
        _apply_step(uninterrupted, _GRADS[0])
        _apply_step(uninterrupted, _GRADS[1])
        _apply_step(resumed, _GRADS[0])
        load_sgd(resumed, dump_sgd(uninterrupted))
        _apply_step(uninterrupted, _GRADS[2])
        _apply_step(resumed, _GRADS[2])
        for got, expected in zip(resumed.parameters, uninterrupted.parameters):
            self.assertEqual(got.data, expected.data)


class CorruptTextTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = _make_optimizer()
        self.good = dump_sgd(self.optimizer)

    def expect_value_error(self, text):
        with self.assertRaises(ValueError):
            load_sgd(self.optimizer, text)

    def test_parse_failures(self):
        self.expect_value_error("")
        self.expect_value_error("not json")
        self.expect_value_error(self.good[:-1])
        self.expect_value_error('{"parameters":[]}')

    def test_whitespace_is_rejected(self):
        self.expect_value_error(' {"parameters":[]}')
        self.expect_value_error(self.good + " ")
        self.expect_value_error(self.good.replace('":', '" :'))
        self.expect_value_error(self.good.replace(",", ", "))
        self.expect_value_error(self.good.replace(":", ": "))

    def test_trailing_newline_is_rejected(self):
        self.expect_value_error(self.good + "\n")
        self.expect_value_error(self.good + "\r\n")

    def test_exponent_notation_is_rejected(self):
        self.expect_value_error(self.good.replace("0.000000", "0e0", 1))
        self.expect_value_error(self.good.replace("1.000000", "1.000000e0", 1))
        self.expect_value_error(self.good.replace("-2.000000", "-2e0", 1))

    def test_integer_float_notation_is_rejected(self):
        self.expect_value_error(self.good.replace("0.000000", "0", 1))
        self.expect_value_error(self.good.replace("1.000000", "1", 1))

    def test_negative_zero_token_is_rejected(self):
        self.expect_value_error(self.good.replace("0.000000", "-0.000000", 1))

    def test_non_finite_tokens_are_rejected(self):
        self.expect_value_error(self.good.replace("0.000000", "NaN", 1))
        self.expect_value_error(self.good.replace("0.000000", "Infinity", 1))
        self.expect_value_error(self.good.replace("0.000000", "-Infinity", 1))

    def test_duplicate_and_extra_keys_are_rejected(self):
        duplicated_entry = (
            '{"parameters":['
            '{"data":0.000000,"requires_grad":true,'
            '"data":0.000000,"requires_grad":true},'
            '{"data":[1.000000,-2.000000],"requires_grad":true}]}'
        )
        self.expect_value_error(duplicated_entry)
        self.expect_value_error(self.good[:-1] + ',"lr":0.250000}')
        self.expect_value_error(
            '{"lr":0.250000,' + self.good[1:]
        )

    def test_wrong_key_order_is_rejected(self):
        reordered = (
            '{"parameters":['
            '{"requires_grad":true,"data":0.000000}'
            + ',{"data":[1.000000,-2.000000],"requires_grad":true}]}'
        )
        self.expect_value_error(reordered)
        self.expect_value_error(
            self.good.replace('"data":', '"requires_grad":x,"data":', 1)
        )

    def test_wrong_top_level_key_is_rejected(self):
        self.expect_value_error(self.good.replace("parameters", "parameter", 1))

    def test_entry_count_mismatch_is_value_error(self):
        one = (
            '{"parameters":[{"data":0.000000,"requires_grad":true}]}'
        )
        three = (
            '{"parameters":['
            '{"data":0.000000,"requires_grad":true},'
            '{"data":0.000000,"requires_grad":true},'
            '{"data":[1.000000,-2.000000],"requires_grad":true}]}'
        )
        self.expect_value_error(one)
        self.expect_value_error(three)

    def test_shape_mismatch_is_value_error(self):
        scalar_as_vector = (
            '{"parameters":['
            '{"data":[0.000000],"requires_grad":true},'
            '{"data":[1.000000,-2.000000],"requires_grad":true}]}'
        )
        vector_as_scalar = (
            '{"parameters":['
            '{"data":0.000000,"requires_grad":true},'
            '{"data":1.000000,"requires_grad":true}]}'
        )
        wrong_length = (
            '{"parameters":['
            '{"data":0.000000,"requires_grad":true},'
            '{"data":[1.000000,-2.000000,3.000000],"requires_grad":true}]}'
        )
        self.expect_value_error(scalar_as_vector)
        self.expect_value_error(vector_as_scalar)
        self.expect_value_error(wrong_length)


class TypeErrorTests(unittest.TestCase):
    def test_dump_requires_sgd(self):
        for build in (
            lambda: Adam([Tensor(1.0)], 0.1),
            lambda: RMSprop([Tensor(1.0)], 0.1),
        ):
            with self.assertRaises(TypeError):
                dump_sgd(build())
        with self.assertRaises(TypeError):
            dump_sgd(object())

    def test_load_requires_sgd(self):
        text = dump_sgd(_make_optimizer())
        with self.assertRaises(TypeError):
            load_sgd(Adam([Tensor(1.0)], 0.1), text)
        with self.assertRaises(TypeError):
            load_sgd(RMSprop([Tensor(1.0)], 0.1), text)
        with self.assertRaises(TypeError):
            load_sgd(object(), text)

    def test_load_text_must_be_str(self):
        optimizer = _make_optimizer()
        for bad in (None, 1, 1.0, b"", [], {}):
            with self.assertRaises(TypeError):
                load_sgd(optimizer, bad)

    def test_parameters_container_type(self):
        optimizer = _make_optimizer()
        optimizer.parameters = (Tensor(1.0),)
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)
        with self.assertRaises(TypeError):
            load_sgd(optimizer, dump_sgd(_make_optimizer()))

    def test_parameters_element_type(self):
        optimizer = _make_optimizer()
        optimizer.parameters = [1.0]
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)
        optimizer.parameters = [Tensor(1.0), 2.0]
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)

    def test_empty_parameters(self):
        optimizer = _make_optimizer()
        optimizer.parameters = []
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)
        with self.assertRaises(ValueError):
            load_sgd(optimizer, dump_sgd(_make_optimizer()))

    def test_duplicate_parameters(self):
        parameter = Tensor(1.0)
        optimizer = _make_optimizer()
        optimizer.parameters = [parameter, parameter]
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)

    def test_lr_type(self):
        optimizer = _make_optimizer()
        for bad in (1, True, False, "0.1", None):
            optimizer.lr = bad
            with self.assertRaises(TypeError):
                dump_sgd(optimizer)

    def test_lr_value(self):
        optimizer = _make_optimizer()
        for bad in (0.0, -0.1, float("inf"), float("-inf"), float("nan")):
            optimizer.lr = bad
            with self.assertRaises(ValueError):
                dump_sgd(optimizer)

    def test_tensor_data_type(self):
        optimizer = _make_optimizer()
        optimizer.parameters[0].data = 1
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = True
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = "1.0"
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = [1.0, 2]
        with self.assertRaises(TypeError):
            dump_sgd(optimizer)

    def test_tensor_data_value(self):
        optimizer = _make_optimizer()
        optimizer.parameters[0].data = []
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = float("nan")
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = float("inf")
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)
        optimizer.parameters[0].data = [1.0, float("nan")]
        with self.assertRaises(ValueError):
            dump_sgd(optimizer)

    def test_requires_grad_type(self):
        optimizer = _make_optimizer()
        for bad in (1, 0, "true", None):
            optimizer.parameters[0].requires_grad = bad
            with self.assertRaises(TypeError):
                dump_sgd(optimizer)


class FailureAtomicityTests(unittest.TestCase):
    def _snapshot(self, optimizer):
        return [
            (
                parameter.data
                if isinstance(parameter.data, float)
                else list(parameter.data),
                type(parameter.data),
                parameter.requires_grad,
                parameter.grad,
            )
            for parameter in optimizer.parameters
        ]

    def test_failed_load_changes_nothing(self):
        optimizer = _make_optimizer()
        _apply_step(optimizer, _GRADS[0])
        for parameter in optimizer.parameters:
            parameter.grad = (
                0.5
                if isinstance(parameter.data, float)
                else [0.5, 0.5]
            )
        snapshot = self._snapshot(optimizer)
        parameter_list = optimizer.parameters
        lr = optimizer.lr

        bad_texts = [
            "",
            dump_sgd(optimizer) + "\n",
            dump_sgd(optimizer).replace("-0.125000", "-0.125", 1),
            '{"parameters":[{"data":0.000000,"requires_grad":true}]}',
            dump_sgd(optimizer).replace(
                "[0.750000,", "[1.750000,0.750000,", 1
            ),
        ]
        for text in bad_texts:
            with self.assertRaises(ValueError):
                load_sgd(optimizer, text)
            self.assertEqual(self._snapshot(optimizer), snapshot)
            self.assertIs(optimizer.parameters, parameter_list)
            self.assertEqual(optimizer.lr, lr)
            for parameter, (_, _, _, grad) in zip(
                optimizer.parameters, snapshot
            ):
                self.assertIs(parameter.grad, grad)
                self.assertIsNotNone(parameter.grad)

    def test_failed_load_with_non_str_text_changes_nothing(self):
        optimizer = _make_optimizer()
        snapshot = self._snapshot(optimizer)
        for bad in (None, 1, b"x"):
            with self.assertRaises(TypeError):
                load_sgd(optimizer, bad)
        self.assertEqual(self._snapshot(optimizer), snapshot)

    def test_failed_load_of_mutated_state_changes_nothing(self):
        optimizer = _make_optimizer()
        good = dump_sgd(optimizer)
        optimizer.parameters[0].data = float("nan")
        with self.assertRaises(ValueError):
            load_sgd(optimizer, good)
        self.assertTrue(math.isnan(optimizer.parameters[0].data))
        optimizer.parameters[0].data = 5.0


if __name__ == "__main__":
    unittest.main()
