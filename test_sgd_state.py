"""Contract tests for dump_sgd / load_sgd.

Standard library only; discovered by ``python -m unittest discover``.
"""

import math
import unittest

from autograd import Adam, RMSprop, SGD, Tensor, dump_sgd, load_sgd


SCALAR_ENTRY = '{"data":1.500000,"requires_grad":true}'
VECTOR_ENTRY = '{"data":[0.000000,2.000000],"requires_grad":false}'


def make_optimizer():
    return SGD(
        [Tensor(1.5, True), Tensor([-0.0, 2.0], False)],
        0.1,
    )


class DumpFormatTests(unittest.TestCase):

    def test_exact_textual_form(self):
        text = dump_sgd(make_optimizer())
        self.assertEqual(
            text,
            '{"parameters":[' + SCALAR_ENTRY + "," + VECTOR_ENTRY + "]}",
        )

    def test_no_whitespace_or_trailing_newline(self):
        text = dump_sgd(make_optimizer())
        self.assertNotIn(" ", text)
        self.assertNotIn("\n", text)
        self.assertNotIn("\t", text)
        self.assertFalse(text.endswith("\n"))

    def test_top_level_has_only_parameters(self):
        text = dump_sgd(make_optimizer())
        self.assertTrue(text.startswith('{"parameters":['))
        self.assertTrue(text.endswith("]}"))
        self.assertNotIn("lr", text)

    def test_negative_zero_written_positive(self):
        text = dump_sgd(SGD([Tensor(-0.0)], 1.0))
        self.assertEqual(text, '{"parameters":[{"data":0.000000,'
                         '"requires_grad":false}]}')

    def test_returns_str(self):
        self.assertIsInstance(dump_sgd(make_optimizer()), str)


class RoundTripTests(unittest.TestCase):

    def test_scalar_and_vector_round_trip(self):
        optimizer = make_optimizer()
        target = SGD([Tensor(99.0, False), Tensor([9.0, 9.0], True)], 0.5)
        result = load_sgd(target, dump_sgd(optimizer))
        self.assertIsNone(result)
        self.assertEqual(target.parameters[0].data, 1.5)
        self.assertIs(target.parameters[0].requires_grad, True)
        self.assertEqual(target.parameters[1].data, [0.0, 2.0])
        self.assertIs(target.parameters[1].requires_grad, False)

    def test_load_clears_grads(self):
        source = make_optimizer()
        target = make_optimizer()
        target.parameters[0].grad = 3.25
        target.parameters[1].grad = [1.0, -1.0]
        load_sgd(target, dump_sgd(source))
        self.assertIsNone(target.parameters[0].grad)
        self.assertIsNone(target.parameters[1].grad)

    def test_identities_and_lr_preserved(self):
        source = make_optimizer()
        target = make_optimizer()
        target.lr = 0.5
        parameter_objects = list(target.parameters)
        result = load_sgd(target, dump_sgd(source))
        self.assertIsNone(result)
        self.assertIs(target.parameters[0], parameter_objects[0])
        self.assertIs(target.parameters[1], parameter_objects[1])
        self.assertEqual(target.lr, 0.5)

    def test_round_trip_is_stable(self):
        optimizer = make_optimizer()
        text = dump_sgd(optimizer)
        load_sgd(optimizer, text)
        self.assertEqual(dump_sgd(optimizer), text)


class OrderTests(unittest.TestCase):

    def test_arrays_follow_parameters_order(self):
        a = Tensor(1.0, True)
        b = Tensor(2.0, False)
        text = dump_sgd(SGD([a, b], 0.1))
        self.assertEqual(
            text,
            '{"parameters":['
            '{"data":1.000000,"requires_grad":true},'
            '{"data":2.000000,"requires_grad":false}]}',
        )
        text_reversed = dump_sgd(SGD([b, a], 0.1))
        target = SGD([Tensor(0.0), Tensor(0.0)], 0.9)
        load_sgd(target, text_reversed)
        self.assertEqual(target.parameters[0].data, 2.0)
        self.assertIs(target.parameters[0].requires_grad, False)
        self.assertEqual(target.parameters[1].data, 1.0)
        self.assertIs(target.parameters[1].requires_grad, True)

    def test_swapped_shapes_rejected(self):
        scalar = Tensor(1.0, True)
        vector = Tensor([2.0, 3.0], True)
        text = dump_sgd(SGD([scalar, vector], 0.1))
        # Entries swapped: scalar position gets a vector and vice versa.
        swapped = (
            '{"parameters":['
            '{"data":[2.000000,3.000000],"requires_grad":true},'
            '{"data":1.000000,"requires_grad":true}]}'
        )
        self.assertNotEqual(text, swapped)
        target = SGD([Tensor(0.0), Tensor([0.0, 0.0])], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(target, swapped)


class ContinuedStepTests(unittest.TestCase):

    def test_steps_after_load_byte_identical(self):
        # Grads are chosen as multiples of 1/8 and lr is 1/8, so every
        # updated value stays a multiple of 1/64: exactly representable in
        # binary and with at most six decimals. An optimizer reloaded from a
        # dump must therefore reproduce the uninterrupted trajectory byte
        # for byte (no rounding perturbation hiding a discrepancy).
        def run(optimizer, steps, start=1):
            dumps = []
            scalar_tensor, vector_tensor, frozen_tensor = optimizer.parameters
            for step in range(start, start + steps):
                scalar_tensor.grad = 0.125 * step
                vector_tensor.grad = [0.25 * step, 0.125 * step]
                optimizer.step()
                optimizer.zero_grad()
                dumps.append(dump_sgd(optimizer))
            return dumps

        uninterrupted = SGD(
            [Tensor(0.75, True), Tensor([1.5, -2.0], True), Tensor(3.0, False)],
            0.125,
        )
        expected = run(uninterrupted, 8)

        interrupted = SGD(
            [Tensor(0.75, True), Tensor([1.5, -2.0], True), Tensor(3.0, False)],
            0.125,
        )
        run(interrupted, 3)
        snapshot = dump_sgd(interrupted)

        reloaded = SGD(
            [Tensor(-9.0, False), Tensor([8.0, 8.0], False), Tensor(42.0)],
            0.125,
        )
        load_sgd(reloaded, snapshot)
        actual = run(reloaded, 5, start=4)
        self.assertEqual(actual, expected[3:])
        self.assertEqual(dump_sgd(reloaded), expected[-1])


class CorruptTextTests(unittest.TestCase):

    VALID = '{"parameters":[{"data":1.000000,"requires_grad":true}]}'

    def assert_rejected(self, text):
        optimizer = SGD([Tensor(0.0, True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(optimizer, text)

    def test_parse_failures(self):
        cases = [
            "",
            "not json",
            "{}",
            "[]",
            "null",
            '{"parameters":[]}',
            self.VALID.lower().replace("true", "True"),
            '{"parameters":[{"data":null,"requires_grad":true}]}',
            '{"parameters":[{"data":1.000000,"requires_grad":1}]}',
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assert_rejected(text)

    def test_whitespace_rejected(self):
        self.assert_rejected(" " + self.VALID)
        self.assert_rejected(self.VALID + " ")
        self.assert_rejected('\n' + self.VALID)
        self.assert_rejected(self.VALID.replace(":", " :"))
        self.assert_rejected(self.VALID.replace(",", ", "))

    def test_trailing_newline_rejected(self):
        self.assert_rejected(self.VALID + "\n")
        self.assert_rejected(self.VALID + "\r\n")

    def test_integer_float_forms_rejected(self):
        self.assert_rejected(
            '{"parameters":[{"data":1,"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":1.0,"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":[1.000000,2],"requires_grad":true}]}'
        )

    def test_exponent_notation_rejected(self):
        self.assert_rejected(
            '{"parameters":[{"data":1e0,"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":1.00000e0,"requires_grad":true}]}'
        )

    def test_negative_zero_rejected(self):
        self.assert_rejected(
            '{"parameters":[{"data":-0.000000,"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":[-0.000000],"requires_grad":true}]}'
        )

    def test_non_finite_rejected(self):
        self.assert_rejected(
            '{"parameters":[{"data":NaN,"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":Infinity,"requires_grad":true}]}'
        )

    def test_duplicate_and_extra_keys_rejected(self):
        entry = '{"data":1.000000,"requires_grad":true}'
        self.assert_rejected(
            '{"parameters":[{"data":1.000000,"requires_grad":true,'
            '"requires_grad":false}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":1.000000,"data":2.000000,'
            '"requires_grad":true}]}'
        )
        self.assert_rejected(
            '{"parameters":[{"data":1.000000,"requires_grad":true,'
            '"extra":0.000000}]}'
        )
        self.assert_rejected(
            '{"parameters":[' + entry + '],"lr":0.100000}'
        )
        self.assert_rejected(
            '{"lr":0.100000,"parameters":[' + entry + "]}"
        )

    def test_key_order_rejected(self):
        self.assert_rejected(
            '{"parameters":[{"requires_grad":true,"data":1.000000}]}'
        )

    def test_count_mismatch_rejected(self):
        two = (
            '{"parameters":['
            '{"data":1.000000,"requires_grad":true},'
            '{"data":2.000000,"requires_grad":true}]}'
        )
        one_param = SGD([Tensor(0.0, True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(one_param, two)
        two_param = SGD([Tensor(0.0, True), Tensor(0.0, True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(two_param, self.VALID)

    def test_shape_mismatch_rejected(self):
        vector_param = SGD([Tensor([0.0, 0.0], True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(vector_param, self.VALID)
        scalar_param = SGD([Tensor(0.0, True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(
                scalar_param,
                '{"parameters":[{"data":[1.000000,2.000000],'
                '"requires_grad":true}]}',
            )
        wrong_length = SGD([Tensor([0.0, 0.0, 0.0], True)], 0.1)
        with self.assertRaises(ValueError):
            load_sgd(
                wrong_length,
                '{"parameters":[{"data":[1.000000,2.000000],'
                '"requires_grad":true}]}',
            )


class TypeErrorTests(unittest.TestCase):

    def test_dump_rejects_non_sgd(self):
        tensor = Tensor([1.0, 2.0], True)
        for non_sgd in (
            Adam([tensor], 0.1),
            RMSprop([tensor], 0.1),
            tensor,
            object(),
            SGD,
            None,
        ):
            with self.subTest(non_sgd=non_sgd):
                with self.assertRaises(TypeError):
                    dump_sgd(non_sgd)

    def test_load_rejects_non_sgd(self):
        tensor = Tensor([1.0, 2.0], True)
        for non_sgd in (
            Adam([tensor], 0.1),
            RMSprop([tensor], 0.1),
            tensor,
            object(),
            SGD,
            None,
        ):
            with self.subTest(non_sgd=non_sgd):
                with self.assertRaises(TypeError):
                    load_sgd(non_sgd, "anything")

    def test_load_rejects_non_str_text(self):
        optimizer = make_optimizer()
        for non_text in (b"", 1, 1.0, True, None, [], {}):
            with self.subTest(non_text=non_text):
                with self.assertRaises(TypeError):
                    load_sgd(optimizer, non_text)

    def test_dump_rejects_corrupt_optimizer_types(self):
        def corrupt(**kwargs):
            optimizer = make_optimizer()
            for key, value in kwargs.items():
                setattr(optimizer, key, value)
            return optimizer

        with self.assertRaises(TypeError):
            dump_sgd(corrupt(parameters="x"))
        with self.assertRaises(TypeError):
            dump_sgd(corrupt(parameters=[object()]))
        with self.assertRaises(TypeError):
            dump_sgd(corrupt(lr=1))
        with self.assertRaises(TypeError):
            dump_sgd(corrupt(lr=True))

        bad_tensor = Tensor(1.0)
        bad_tensor.requires_grad = 1
        with self.assertRaises(TypeError):
            dump_sgd(corrupt(parameters=[bad_tensor]))
        bad_tensor = Tensor(1.0)
        bad_tensor.data = 1
        with self.assertRaises(TypeError):
            dump_sgd(SGD([bad_tensor], 0.1))

    def test_dump_rejects_corrupt_optimizer_values(self):
        def corrupt(**kwargs):
            optimizer = make_optimizer()
            for key, value in kwargs.items():
                setattr(optimizer, key, value)
            return optimizer

        with self.assertRaises(ValueError):
            dump_sgd(corrupt(parameters=[]))
        tensor = Tensor(1.0)
        with self.assertRaises(ValueError):
            dump_sgd(corrupt(parameters=[tensor, tensor]))
        with self.assertRaises(ValueError):
            dump_sgd(corrupt(lr=0.0))
        with self.assertRaises(ValueError):
            dump_sgd(corrupt(lr=-1.0))
        with self.assertRaises(ValueError):
            dump_sgd(corrupt(lr=float("inf")))
        with self.assertRaises(ValueError):
            dump_sgd(corrupt(lr=float("nan")))

        bad_tensor = Tensor(1.0)
        bad_tensor.data = []
        with self.assertRaises(ValueError):
            dump_sgd(SGD([bad_tensor], 0.1))
        bad_tensor = Tensor(1.0)
        bad_tensor.data = float("nan")
        with self.assertRaises(ValueError):
            dump_sgd(SGD([bad_tensor], 0.1))
        bad_list = Tensor([1.0])
        bad_list.data = [1.0, float("inf")]
        with self.assertRaises(ValueError):
            dump_sgd(SGD([bad_list], 0.1))

    def test_constructor_type_errors(self):
        with self.assertRaises(TypeError):
            SGD((Tensor(1.0),), 0.1)
        with self.assertRaises(TypeError):
            SGD([1.0], 0.1)
        with self.assertRaises(TypeError):
            SGD([Tensor(1.0)], 1)
        with self.assertRaises(ValueError):
            SGD([], 0.1)
        with self.assertRaises(ValueError):
            SGD([Tensor(1.0)], 0.0)
        tensor = Tensor(1.0)
        with self.assertRaises(ValueError):
            SGD([tensor, tensor], 0.1)


class FailureAtomicityTests(unittest.TestCase):

    def test_failed_load_leaves_state_untouched(self):
        optimizer = make_optimizer()
        optimizer.parameters[0].grad = 7.5
        optimizer.parameters[1].grad = [0.5, -0.25]
        parameter_objects = list(optimizer.parameters)
        parameter_list = optimizer.parameters
        bad_texts = [
            123,
            "",
            '{"parameters":[]}',
            '{"parameters":[{"data":1.000000,"requires_grad":true},'
            '{"data":2.000000,"requires_grad":true}]}',
            '{"parameters":[{"data":[1.000000,2.000000],'
            '"requires_grad":true}]}',
            '{"parameters":[{"data":1.000000,"requires_grad":true,'
            '"extra":null}]}',
            '{"parameters":[{"requires_grad":true,"data":1.000000}]}',
            '{"parameters":[{"data":-0.000000,"requires_grad":true}]}',
        ]
        for text in bad_texts:
            with self.subTest(text=text):
                with self.assertRaises((TypeError, ValueError)):
                    load_sgd(optimizer, text)
                self.assertEqual(optimizer.parameters[0].data, 1.5)
                self.assertIs(
                    optimizer.parameters[0].requires_grad, True
                )
                self.assertEqual(
                    optimizer.parameters[1].data, [-0.0, 2.0]
                )
                self.assertIs(
                    optimizer.parameters[1].requires_grad, False
                )
                self.assertEqual(optimizer.parameters[0].grad, 7.5)
                self.assertEqual(
                    optimizer.parameters[1].grad, [0.5, -0.25]
                )
                self.assertIs(optimizer.parameters, parameter_list)
                self.assertIs(
                    optimizer.parameters[0], parameter_objects[0]
                )
                self.assertIs(
                    optimizer.parameters[1], parameter_objects[1]
                )
                self.assertEqual(optimizer.lr, 0.1)

    def test_later_entry_failure_leaves_earlier_untouched(self):
        # The second entry has the wrong vector length: validation must
        # complete before any parameter is mutated.
        optimizer = SGD(
            [Tensor(1.0, True), Tensor([2.0, 3.0], False)], 0.1
        )
        optimizer.parameters[0].grad = 9.0
        text = (
            '{"parameters":['
            '{"data":8.000000,"requires_grad":false},'
            '{"data":[4.000000],"requires_grad":true}]}'
        )
        with self.assertRaises(ValueError):
            load_sgd(optimizer, text)
        self.assertEqual(optimizer.parameters[0].data, 1.0)
        self.assertIs(optimizer.parameters[0].requires_grad, True)
        self.assertEqual(optimizer.parameters[0].grad, 9.0)
        self.assertEqual(optimizer.parameters[1].data, [2.0, 3.0])
        self.assertIs(optimizer.parameters[1].requires_grad, False)


if __name__ == "__main__":
    unittest.main()
