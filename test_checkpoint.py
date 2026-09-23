"""Contract tests for dump_checkpoint/load_checkpoint.

Covers AMSGrad v1, Adam v2, AdamW v3, Adamax v4, SGD v5,
MomentumSGD v6, Adagrad v7 and RMSprop v8. Standard library only;
discovered by ``python -m unittest discover``.
"""

import hashlib
import json
import unittest

from autograd import (
    AMSGrad,
    SGD,
    Adagrad,
    Adam,
    MomentumSGD,
    RMSprop,
    Tensor,
    dump_adagrad,
    dump_adam,
    dump_amsgrad,
    dump_checkpoint,
    dump_momentum_sgd,
    dump_rmsprop,
    dump_sgd,
    load_checkpoint,
)


def make_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = AMSGrad([w, b], 0.01)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


def make_adam_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = Adam([w, b], 0.01)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


class DumpFormatTests(unittest.TestCase):

    def test_textual_form(self):
        parameters, optimizer = make_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload = json.loads(text)
        self.assertEqual(
            list(payload.keys()), ["version", "names", "amsgrad"]
        )
        self.assertEqual(payload["version"], 1)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["names"], ["w", "b"])
        inner_start = text.index('"amsgrad":') + len('"amsgrad":')
        self.assertEqual(text[inner_start:-1], dump_amsgrad(optimizer))

    def test_adam_textual_form(self):
        parameters, optimizer = make_adam_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload = json.loads(text)
        self.assertEqual(
            list(payload.keys()), ["version", "type", "names", "state"]
        )
        self.assertEqual(payload["version"], 2)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["type"], "adam")
        self.assertEqual(payload["names"], ["w", "b"])
        inner_start = text.index('"state":') + len('"state":')
        self.assertEqual(text[inner_start:-1], dump_adam(optimizer))

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, object())

    def test_parameters_follow_dump_state_contract(self):
        _, optimizer = make_state()
        w, b = optimizer.parameters
        with self.assertRaises(TypeError):
            dump_checkpoint([("w", w), ("b", b)], optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({1: w, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"1": w, "b": b}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": 1.0, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w, "x": w}, optimizer)

    def test_values_must_be_optimizer_parameters_in_order(self):
        _, optimizer = make_state()
        w, b = optimizer.parameters
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": b, "b": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint(
                {"w": Tensor(list(w.data), True), "b": b}, optimizer
            )


class LoadRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_amsgrad_semantics(self):
        parameters, optimizer = make_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = AMSGrad([qw, qb], 0.5, beta1=0.8, beta2=0.9, eps=1e-6)
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        self.assertEqual(qw.data, [0.09, -0.19, 0.29])
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        self.assertEqual(target.m, optimizer_dump_slots(optimizer, "m"))
        self.assertEqual(target.v, optimizer_dump_slots(optimizer, "v"))
        self.assertEqual(
            target.v_max, optimizer_dump_slots(optimizer, "v_max")
        )
        self.assertEqual(target.t, 1)
        # Identity and hyperparameters survive.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)
        self.assertEqual(target.beta1, 0.8)
        self.assertEqual(target.beta2, 0.9)
        self.assertEqual(target.eps, 1e-6)

    def test_adam_round_trip_restores_semantics(self):
        parameters, optimizer = make_adam_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = Adam([qw, qb], 0.5, beta1=0.8, beta2=0.9, eps=1e-6)
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        self.assertEqual(qw.data, [0.09, -0.19, 0.29])
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        payload = json.loads(dump_adam(optimizer))
        self.assertEqual(target.m, payload["m"])
        self.assertEqual(target.v, payload["v"])
        self.assertEqual(target.t, 1)
        # Identity and hyperparameters survive.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)
        self.assertEqual(target.beta1, 0.8)
        self.assertEqual(target.beta2, 0.9)
        self.assertEqual(target.eps, 1e-6)

    def test_adam_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_adam_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = Adam([w, b], 0.01)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_cross_kind_load_is_rejected(self):
        amsgrad_parameters, amsgrad_optimizer = make_state()
        amsgrad_text = dump_checkpoint(amsgrad_parameters, amsgrad_optimizer)
        adam_parameters, adam_optimizer = make_adam_state()
        adam_text = dump_checkpoint(adam_parameters, adam_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, Adam([w, b], 0.01), amsgrad_text
            )
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, AMSGrad([w, b], 0.01), adam_text
            )

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = AMSGrad([w, b], 0.01)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())


def optimizer_dump_slots(optimizer, name):
    """Six-decimal-quantized slot values as a reload observes them."""
    payload = json.loads(dump_amsgrad(optimizer))
    return payload[name]


class LoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_non_str_text(self):
        for bad in (None, b"x", 1, 1.0, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, object(), self.text)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
        w, b = self.optimizer.parameters
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": b, "b": w}, self.optimizer, self.text
            )

    def test_rejects_malformed_text(self):
        amsgrad = dump_amsgrad(self.optimizer)
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            '{"names":["w","b"],"version":1,"amsgrad":' + amsgrad + "}",
            '{"version":2,"names":["w","b"],"amsgrad":' + amsgrad + "}",
            '{"version":01,"names":["w","b"],"amsgrad":' + amsgrad + "}",
            '{"version":1.0,"names":["w","b"],"amsgrad":' + amsgrad + "}",
            '{"version":1,"names":["w"],"amsgrad":' + amsgrad + "}",
            '{"version":1,"names":["b","w"],"amsgrad":' + amsgrad + "}",
            '{"version":1,"names":["w","b"],"extra":0,"amsgrad":'
            + amsgrad
            + "}",
            '{"version":1,"names":["w","b"],"amsgrad":'
            '{"parameters":[],"m":[],"v":[],"v_max":[],"t":0}}',
            '{"version":1,"names":["w","b"],"amsgrad":'
            + amsgrad.replace('"t":1', '"t":1,"t":2', 1)
            + "}",
            self.text + " ",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        data_before = list(w.data)
        grad_before = list(w.grad)
        m_before = list(self.optimizer.m[0])
        t_before = self.optimizer.t
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":1', '"version":2'),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(self.optimizer.m[0], m_before)
                self.assertEqual(self.optimizer.t, t_before)


class AdamLoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_adam_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_malformed_text(self):
        adam = dump_adam(self.optimizer)
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            '{"version":2,"names":["w","b"],"type":"adam","state":'
            + adam
            + "}",
            '{"version":2,"type":"adam","state":' + adam + ',"names":["w","b"]}',
            '{"version":2,"type":"amsgrad","names":["w","b"],"state":'
            + adam
            + "}",
            '{"version":2,"type":"adam","names":["w","b"],"amsgrad":'
            + adam
            + "}",
            '{"version":2,"type":"adam","names":["w"],"state":' + adam + "}",
            '{"version":2,"type":"adam","names":["b","w"],"state":'
            + adam
            + "}",
            '{"version":2,"type":"adam","names":["w","b"],"extra":0,"state":'
            + adam
            + "}",
            '{"version":2,"type":"adam","names":["w","b"],"state":'
            '{"parameters":[],"m":[],"v":[],"t":0}}',
            '{"version":2,"type":"adam","names":["w","b"],"state":'
            + adam.replace('"t":1', '"t":1,"t":2', 1)
            + "}",
            '{"version":1,"type":"adam","names":["w","b"],"state":'
            + adam
            + "}",
            self.text + " ",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        data_before = list(w.data)
        grad_before = list(w.grad)
        m_before = list(self.optimizer.m[0])
        t_before = self.optimizer.t
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":2', '"version":3'),
            self.text.replace('"type":"adam"', '"type":"sgd"'),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(self.optimizer.m[0], m_before)
                self.assertEqual(self.optimizer.t, t_before)


def make_sgd_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = SGD([w, b], 0.01)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


class SGDDumpFormatTests(unittest.TestCase):

    def test_textual_form(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload = json.loads(text)
        self.assertEqual(
            list(payload.keys()), ["version", "type", "names", "state"]
        )
        self.assertEqual(payload["version"], 5)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["type"], "sgd")
        self.assertEqual(payload["names"], ["w", "b"])
        self.assertEqual(list(payload["state"].keys()), ["parameters"])
        inner_start = text.index('"state":') + len('"state":')
        self.assertEqual(text[inner_start:-1], dump_sgd(optimizer))

    def test_exact_byte_form(self):
        w = Tensor(1.5, True)
        b = Tensor([-0.0, 2.0], False)
        optimizer = SGD([w, b], 0.1)
        text = dump_checkpoint({"w": w, "b": b}, optimizer)
        self.assertEqual(
            text,
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":['
            '{"data":1.500000,"requires_grad":true},'
            '{"data":[0.000000,2.000000],"requires_grad":false}'
            "]}}",
        )

    def test_six_decimals_no_exponent_no_negative_zero(self):
        tiny = Tensor(1e-9, True)
        huge = Tensor(1e20, True)
        negzero = Tensor(-0.0, False)
        optimizer = SGD([tiny, huge, negzero], 1.0)
        text = dump_checkpoint(
            {"tiny": tiny, "huge": huge, "negzero": negzero}, optimizer
        )
        self.assertIn("0.000000", text)
        self.assertIn("100000000000000000000.000000", text)
        self.assertNotIn("e+", text)
        self.assertNotIn("e-", text)
        self.assertNotIn("E", text)
        self.assertNotIn("-0.000000", text)

    def test_parameters_follow_dump_state_contract(self):
        _, optimizer = make_sgd_state()
        w, b = optimizer.parameters
        with self.assertRaises(TypeError):
            dump_checkpoint([("w", w), ("b", b)], optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({1: w, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"1": w, "b": b}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": 1.0, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w, "x": w}, optimizer)

    def test_values_must_be_optimizer_parameters_in_order(self):
        _, optimizer = make_sgd_state()
        w, b = optimizer.parameters
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": b, "b": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint(
                {"w": Tensor(list(w.data), True), "b": b}, optimizer
            )


class SGDLoadRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_semantics(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = SGD([qw, qb], 0.5)
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        self.assertEqual(qw.data, [0.096, -0.195, 0.294])
        # b had requires_grad False, so the step left it at 1.5.
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        # List/Tensor identities and lr survive.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = SGD([w, b], 0.01)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_cross_kind_load_is_rejected(self):
        sgd_parameters, sgd_optimizer = make_sgd_state()
        sgd_text = dump_checkpoint(sgd_parameters, sgd_optimizer)
        adam_parameters, adam_optimizer = make_adam_state()
        adam_text = dump_checkpoint(adam_parameters, adam_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": w, "b": b}, SGD([w, b], 0.01), adam_text)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": w, "b": b}, Adam([w, b], 0.01), sgd_text)
        amsgrad_parameters, amsgrad_optimizer = make_state()
        amsgrad_text = dump_checkpoint(amsgrad_parameters, amsgrad_optimizer)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": w, "b": b}, SGD([w, b], 0.01), amsgrad_text)


class SGDLoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_sgd_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_non_str_text(self):
        for bad in (None, b"x", 1, 1.0, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_bad_parameters(self):
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
        w, b = self.optimizer.parameters
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": b, "b": w}, self.optimizer, self.text
            )

    def test_rejects_malformed_text(self):
        sgd = dump_sgd(self.optimizer)
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            '{"version":5,"names":["w","b"],"type":"sgd","state":'
            + sgd
            + "}",
            '{"type":"sgd","version":5,"names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","state":' + sgd + ',"names":["w","b"]}',
            '{"version":5,"names":["w","b"],"state":' + sgd + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"sgd":'
            + sgd
            + "}",
            '{"version":5,"type":"adam","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":4,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":05,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":5.0,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w"],"state":' + sgd + "}",
            '{"version":5,"type":"sgd","names":["b","w"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","w"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","x"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd.replace('"parameters"', '"parameters"', 1)
            .replace('{"parameters"', '{"extra":0,"parameters"', 1)
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":[{"requires_grad":true,"data":1.0}]}}',
            '{"version":6,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":[{"data":1e9,"requires_grad":true}]}}',
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":[{"data":-0.000000,"requires_grad":true}]}}',
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":[{"data":1.0000000,"requires_grad":true}]}}',
            self.text + " ",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_state_shape_mismatch(self):
        w, b = self.optimizer.parameters
        qw = Tensor(9.0, True)
        qb = Tensor([9.0, 9.0], False)
        target = SGD([qw, qb], 0.5)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, self.text)
        with self.assertRaises(ValueError):
            load_checkpoint({"b": qb, "w": qw}, target, self.text)

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        b.grad = 0.9
        data_before = list(w.data)
        b_data_before = b.data
        grad_before = list(w.grad)
        lr_before = self.optimizer.lr
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":5', '"version":2'),
            self.text.replace('"type":"sgd"', '"type":"adam"'),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(b.data, b_data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(b.grad, 0.9)
                self.assertIs(self.optimizer.lr, lr_before)


def make_momentum_sgd_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = MomentumSGD([w, b], 0.01, momentum=0.9, nesterov=True)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


class MomentumSGDDumpFormatTests(unittest.TestCase):

    def test_textual_form(self):
        parameters, optimizer = make_momentum_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload = json.loads(text)
        self.assertEqual(
            list(payload.keys()), ["version", "type", "names", "state"]
        )
        self.assertEqual(payload["version"], 6)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["type"], "momentum_sgd")
        self.assertEqual(payload["names"], ["w", "b"])
        self.assertEqual(
            list(payload["state"].keys()), ["parameters", "velocity"]
        )
        inner_start = text.index('"state":') + len('"state":')
        self.assertEqual(text[inner_start:-1], dump_momentum_sgd(optimizer))

    def test_exact_byte_form(self):
        w = Tensor(1.5, True)
        b = Tensor([-0.0, 2.0], False)
        optimizer = MomentumSGD([w, b], 0.1, momentum=0.5)
        text = dump_checkpoint({"w": w, "b": b}, optimizer)
        self.assertEqual(
            text,
            '{"version":6,"type":"momentum_sgd","names":["w","b"],"state":'
            '{"parameters":['
            '{"data":1.500000,"requires_grad":true},'
            '{"data":[0.000000,2.000000],"requires_grad":false}'
            '],"velocity":[0.000000,[0.000000,0.000000]]}}',
        )

    def test_stepped_state_exact_byte_form(self):
        w = Tensor(1.0, True)
        optimizer = MomentumSGD([w], 0.1, momentum=0.9)
        w.grad = 0.5
        optimizer.step()
        text = dump_checkpoint({"w": w}, optimizer)
        self.assertEqual(
            text,
            '{"version":6,"type":"momentum_sgd","names":["w"],"state":'
            '{"parameters":['
            '{"data":0.950000,"requires_grad":true}'
            '],"velocity":[0.500000]}}',
        )

    def test_six_decimals_no_exponent_no_negative_zero(self):
        tiny = Tensor(1e-9, True)
        huge = Tensor(1e20, True)
        negzero = Tensor(-0.0, False)
        optimizer = MomentumSGD([tiny, huge, negzero], 1.0)
        text = dump_checkpoint(
            {"tiny": tiny, "huge": huge, "negzero": negzero}, optimizer
        )
        self.assertIn("0.000000", text)
        self.assertIn("100000000000000000000.000000", text)
        self.assertNotIn("e+", text)
        self.assertNotIn("e-", text)
        self.assertNotIn("E", text)
        self.assertNotIn("-0.000000", text)

    def test_parameters_follow_dump_state_contract(self):
        _, optimizer = make_momentum_sgd_state()
        w, b = optimizer.parameters
        with self.assertRaises(TypeError):
            dump_checkpoint([("w", w), ("b", b)], optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({1: w, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"1": w, "b": b}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": 1.0, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w, "x": w}, optimizer)

    def test_values_must_be_optimizer_parameters_in_order(self):
        _, optimizer = make_momentum_sgd_state()
        w, b = optimizer.parameters
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": b, "b": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint(
                {"w": Tensor(list(w.data), True), "b": b}, optimizer
            )

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, object())


class MomentumSGDLoadRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_semantics(self):
        parameters, optimizer = make_momentum_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = MomentumSGD(
            [qw, qb], 0.5, momentum=0.25, nesterov=False
        )
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        # v = g with zero initial velocity and Nesterov direction
        # 1.9*g, so w moves by 0.01 * 1.9 * g; values are stored at the
        # six-decimal serialization precision.
        self.assertEqual(qw.data, [0.0924, -0.1905, 0.2886])
        # b had requires_grad False, so the step left it at 1.5.
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        self.assertEqual(target.velocity, [[0.4, -0.5, 0.6], 0.0])
        # List/Tensor identities and hyperparameters survive.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)
        self.assertEqual(target.momentum, 0.25)
        self.assertFalse(target.nesterov)

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_momentum_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = MomentumSGD([w, b], 0.01, momentum=0.9, nesterov=True)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_dump_is_deterministic(self):
        text1 = dump_checkpoint(*make_momentum_sgd_state())
        text2 = dump_checkpoint(*make_momentum_sgd_state())
        self.assertEqual(text1, text2)

    def test_cross_kind_load_is_rejected(self):
        momentum_parameters, momentum_optimizer = make_momentum_sgd_state()
        momentum_text = dump_checkpoint(
            momentum_parameters, momentum_optimizer
        )
        sgd_parameters, sgd_optimizer = make_sgd_state()
        sgd_text = dump_checkpoint(sgd_parameters, sgd_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, SGD([w, b], 0.01), momentum_text
            )
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b},
                MomentumSGD([w, b], 0.01),
                sgd_text,
            )


class MomentumSGDLoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_momentum_sgd_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_non_str_text(self):
        for bad in (None, b"x", 1, 1.0, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, object(), self.text)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
        w, b = self.optimizer.parameters
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": b, "b": w}, self.optimizer, self.text
            )

    def test_rejects_malformed_text(self):
        momentum = dump_momentum_sgd(self.optimizer)
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            '{"version":6,"names":["w","b"],"type":"momentum_sgd",'
            '"state":' + momentum + "}",
            '{"type":"momentum_sgd","version":6,"names":["w","b"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","state":'
            + momentum
            + ',"names":["w","b"]}',
            '{"version":6,"names":["w","b"],"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"velocity":' + momentum + "}",
            '{"version":6,"type":"sgd","names":["w","b"],"state":'
            + momentum
            + "}",
            '{"version":5,"type":"momentum_sgd","names":["w","b"],'
            '"state":' + momentum + "}",
            '{"version":06,"type":"momentum_sgd","names":["w","b"],'
            '"state":' + momentum + "}",
            '{"version":6.0,"type":"momentum_sgd","names":["w","b"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["w"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["b","w"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["w","w"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["w","x"],'
            '"state":' + momentum + "}",
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"extra":0,"state":' + momentum + "}",
            # Inner state key order: velocity before parameters.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"velocity":[],"parameters":[]}}',
            # Inner parameter entry key order: requires_grad before data.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"requires_grad":true,"data":1.0}'
            '],"velocity":[1.0]}}',
            # Extra inner state key.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true}'
            '],"velocity":[1.0],"t":0}}',
            # Missing velocity.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true}]}}',
            # Velocity length mismatch.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true},'
            '{"data":2.0,"requires_grad":false}'
            '],"velocity":[1.0]}}',
            # Lexical/numeric number violations.
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1e9,"requires_grad":true}'
            '],"velocity":[1.0]}}',
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":-0.000000,"requires_grad":true}'
            '],"velocity":[1.0]}}',
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0000000,"requires_grad":true}'
            '],"velocity":[1.0]}}',
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.000000,"requires_grad":true}'
            '],"velocity":[1e9]}}',
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.000000,"requires_grad":true}'
            '],"velocity":[-0.000000]}}',
            '{"version":6,"type":"momentum_sgd","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":true,"requires_grad":true}'
            '],"velocity":[1.0]}}',
            self.text + " ",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_state_shape_mismatch(self):
        qw = Tensor(9.0, True)
        qb = Tensor([9.0, 9.0], False)
        target = MomentumSGD([qw, qb], 0.5)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, self.text)
        with self.assertRaises(ValueError):
            load_checkpoint({"b": qb, "w": qw}, target, self.text)

    def test_rejects_velocity_shape_mismatch(self):
        # Data shapes match the target but a vector parameter's velocity
        # slot is a scalar, which must be rejected with state untouched.
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, False)
        target = MomentumSGD([qw, qb], 0.5)
        bad = (
            '{"version":6,"type":"momentum_sgd","names":["w","b"],"state":'
            '{"parameters":['
            '{"data":[1.0,2.0,3.0],"requires_grad":true},'
            '{"data":1.0,"requires_grad":false}'
            '],"velocity":[0.000000,0.000000]}}'
        )
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, bad)

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        b.grad = 0.9
        data_before = list(w.data)
        b_data_before = b.data
        grad_before = list(w.grad)
        velocity_before = [
            list(slot) if isinstance(slot, list) else slot
            for slot in self.optimizer.velocity
        ]
        lr_before = self.optimizer.lr
        momentum_before = self.optimizer.momentum
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":6', '"version":5'),
            self.text.replace('"type":"momentum_sgd"', '"type":"sgd"'),
            self.text.replace('"velocity":[', '"velocity":[9.9,', 1),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(b.data, b_data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(b.grad, 0.9)
                self.assertEqual(
                    self.optimizer.velocity, velocity_before
                )
                self.assertIs(self.optimizer.lr, lr_before)
                self.assertIs(
                    self.optimizer.momentum, momentum_before
                )


def make_adagrad_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = Adagrad([w, b], 0.01)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


class AdagradDumpFormatTests(unittest.TestCase):

    def test_textual_form(self):
        parameters, optimizer = make_adagrad_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload = json.loads(text)
        self.assertEqual(
            list(payload.keys()), ["version", "type", "names", "state"]
        )
        self.assertEqual(payload["version"], 7)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["type"], "adagrad")
        self.assertEqual(payload["names"], ["w", "b"])
        self.assertEqual(
            list(payload["state"].keys()), ["parameters", "sum_sq"]
        )
        inner_start = text.index('"state":') + len('"state":')
        self.assertEqual(text[inner_start:-1], dump_adagrad(optimizer))

    def test_exact_byte_form(self):
        w = Tensor(1.5, True)
        b = Tensor([-0.0, 2.0], False)
        optimizer = Adagrad([w, b], 0.1)
        text = dump_checkpoint({"w": w, "b": b}, optimizer)
        self.assertEqual(
            text,
            '{"version":7,"type":"adagrad","names":["w","b"],"state":'
            '{"parameters":['
            '{"data":1.500000,"requires_grad":true},'
            '{"data":[0.000000,2.000000],"requires_grad":false}'
            '],"sum_sq":[0.000000,[0.000000,0.000000]]}}',
        )

    def test_stepped_state_exact_byte_form(self):
        w = Tensor(1.0, True)
        optimizer = Adagrad([w], 0.1)
        w.grad = 0.5
        optimizer.step()
        text = dump_checkpoint({"w": w}, optimizer)
        self.assertEqual(
            text,
            '{"version":7,"type":"adagrad","names":["w"],"state":'
            '{"parameters":['
            '{"data":0.900000,"requires_grad":true}'
            '],"sum_sq":[0.250000]}}',
        )

    def test_six_decimals_no_exponent_no_negative_zero(self):
        tiny = Tensor(1e-9, True)
        huge = Tensor(1e20, True)
        negzero = Tensor(-0.0, False)
        optimizer = Adagrad([tiny, huge, negzero], 1.0)
        text = dump_checkpoint(
            {"tiny": tiny, "huge": huge, "negzero": negzero}, optimizer
        )
        self.assertIn("0.000000", text)
        self.assertIn("100000000000000000000.000000", text)
        self.assertNotIn("e+", text)
        self.assertNotIn("e-", text)
        self.assertNotIn("E", text)
        self.assertNotIn("-0.000000", text)

    def test_parameters_follow_dump_state_contract(self):
        _, optimizer = make_adagrad_state()
        w, b = optimizer.parameters
        with self.assertRaises(TypeError):
            dump_checkpoint([("w", w), ("b", b)], optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({1: w, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"1": w, "b": b}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": 1.0, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w, "x": w}, optimizer)

    def test_values_must_be_optimizer_parameters_in_order(self):
        _, optimizer = make_adagrad_state()
        w, b = optimizer.parameters
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": b, "b": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint(
                {"w": Tensor(list(w.data), True), "b": b}, optimizer
            )

    def test_optimizer_parameters_must_be_a_list(self):
        w = Tensor([0.1], True)
        optimizer = Adagrad([w], 0.01)
        optimizer.parameters = tuple(optimizer.parameters)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, optimizer)


class AdagradLoadRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_semantics(self):
        parameters, optimizer = make_adagrad_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = Adagrad([qw, qb], 0.5)
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        # sum_sq = g*g, so w moves by lr * g / (sqrt(g*g) + eps);
        # values are stored at the six-decimal serialization precision.
        self.assertEqual(qw.data, [0.09, -0.19, 0.29])
        # b had requires_grad False, so the step left it at 1.5.
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        self.assertEqual(target.sum_sq, [[0.16, 0.25, 0.36], 0.0])
        # List/Tensor identities and hyperparameters survive.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)
        self.assertEqual(target.eps, 1e-8)

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_adagrad_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = Adagrad([w, b], 0.01)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_dump_is_deterministic(self):
        text1 = dump_checkpoint(*make_adagrad_state())
        text2 = dump_checkpoint(*make_adagrad_state())
        self.assertEqual(text1, text2)

    def test_cross_kind_load_is_rejected(self):
        adagrad_parameters, adagrad_optimizer = make_adagrad_state()
        adagrad_text = dump_checkpoint(adagrad_parameters, adagrad_optimizer)
        sgd_parameters, sgd_optimizer = make_sgd_state()
        sgd_text = dump_checkpoint(sgd_parameters, sgd_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, SGD([w, b], 0.01), adagrad_text
            )
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b},
                Adagrad([w, b], 0.01),
                sgd_text,
            )


class AdagradLoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_adagrad_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_non_str_text(self):
        for bad in (None, b"x", 1, 1.0, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, object(), self.text)

    def test_optimizer_parameters_must_be_a_list(self):
        optimizer = Adagrad([Tensor([0.1], True)], 0.01)
        optimizer.parameters = tuple(optimizer.parameters)
        with self.assertRaises(TypeError):
            load_checkpoint(
                {"w": optimizer.parameters[0]}, optimizer, self.text
            )

    def test_rejects_bad_parameters(self):
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
        w, b = self.optimizer.parameters
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": b, "b": w}, self.optimizer, self.text
            )

    def test_rejects_malformed_text(self):
        state = dump_adagrad(self.optimizer)
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            '{"version":7,"names":["w","b"],"type":"adagrad",'
            '"state":' + state + "}",
            '{"type":"adagrad","version":7,"names":["w","b"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","state":'
            + state
            + ',"names":["w","b"]}',
            '{"version":7,"names":["w","b"],"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"sum_sq":' + state + "}",
            '{"version":7,"type":"sgd","names":["w","b"],"state":'
            + state
            + "}",
            '{"version":5,"type":"adagrad","names":["w","b"],'
            '"state":' + state + "}",
            '{"version":07,"type":"adagrad","names":["w","b"],'
            '"state":' + state + "}",
            '{"version":7.0,"type":"adagrad","names":["w","b"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["w"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["b","w"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["w","w"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["w","x"],'
            '"state":' + state + "}",
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"extra":0,"state":' + state + "}",
            # Inner state key order: sum_sq before parameters.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"sum_sq":[],"parameters":[]}}',
            # Inner parameter entry key order: requires_grad before data.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"requires_grad":true,"data":1.0}'
            '],"sum_sq":[1.0]}}',
            # Extra inner state key.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true}'
            '],"sum_sq":[1.0],"t":0}}',
            # Missing sum_sq.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true}]}}',
            # sum_sq length mismatch.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0,"requires_grad":true},'
            '{"data":2.0,"requires_grad":false}'
            '],"sum_sq":[1.0]}}',
            # Lexical/numeric number violations.
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1e9,"requires_grad":true}'
            '],"sum_sq":[1.0]}}',
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":-0.000000,"requires_grad":true}'
            '],"sum_sq":[1.0]}}',
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.0000000,"requires_grad":true}'
            '],"sum_sq":[1.0]}}',
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.000000,"requires_grad":true}'
            '],"sum_sq":[1e9]}}',
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":1.000000,"requires_grad":true}'
            '],"sum_sq":[-0.000000]}}',
            '{"version":7,"type":"adagrad","names":["w","b"],'
            '"state":{"parameters":['
            '{"data":true,"requires_grad":true}'
            '],"sum_sq":[1.0]}}',
            self.text + " ",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_state_shape_mismatch(self):
        qw = Tensor(9.0, True)
        qb = Tensor([9.0, 9.0], False)
        target = Adagrad([qw, qb], 0.5)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, self.text)
        with self.assertRaises(ValueError):
            load_checkpoint({"b": qb, "w": qw}, target, self.text)

    def test_rejects_sum_sq_shape_mismatch(self):
        # Data shapes match the target but a vector parameter's sum_sq
        # slot is a scalar, which must be rejected with state untouched.
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, False)
        target = Adagrad([qw, qb], 0.5)
        bad = (
            '{"version":7,"type":"adagrad","names":["w","b"],"state":'
            '{"parameters":['
            '{"data":[1.0,2.0,3.0],"requires_grad":true},'
            '{"data":1.0,"requires_grad":false}'
            '],"sum_sq":[0.000000,0.000000]}}'
        )
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, bad)

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        b.grad = 0.9
        data_before = list(w.data)
        b_data_before = b.data
        grad_before = list(w.grad)
        sum_sq_before = [
            list(slot) if isinstance(slot, list) else slot
            for slot in self.optimizer.sum_sq
        ]
        lr_before = self.optimizer.lr
        eps_before = self.optimizer.eps
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":7', '"version":6'),
            self.text.replace('"type":"adagrad"', '"type":"sgd"'),
            self.text.replace('"sum_sq":[', '"sum_sq":[9.9,', 1),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(b.data, b_data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(b.grad, 0.9)
                self.assertEqual(self.optimizer.sum_sq, sum_sq_before)
                self.assertIs(self.optimizer.lr, lr_before)
                self.assertIs(self.optimizer.eps, eps_before)


def make_rmsprop_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = RMSprop([w, b], 0.01, alpha=0.9, eps=0.001)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    optimizer.step()
    return {"w": w, "b": b}, optimizer


def redigest_v8(text):
    """Reattach a valid digest after tampering with a v8 payload."""
    payload, _ = text.rsplit(',"digest":"', 1)
    payload += "}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload[:-1] + ',"digest":"' + digest + '"}'


class RMSpropDumpFormatTests(unittest.TestCase):

    def test_textual_form(self):
        parameters, optimizer = make_rmsprop_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsInstance(text, str)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        payload_text, digest_suffix = text.rsplit(',"digest":"', 1)
        payload_text += "}"
        self.assertTrue(digest_suffix.endswith('"}'))
        digest = digest_suffix[:-2]
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, digest.lower())
        self.assertEqual(
            digest,
            hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        )
        payload = json.loads(payload_text)
        self.assertEqual(
            list(payload.keys()),
            ["version", "type", "names", "hyperparameters", "state"],
        )
        self.assertEqual(payload["version"], 8)
        self.assertIsInstance(payload["version"], int)
        self.assertEqual(payload["type"], "rmsprop")
        self.assertEqual(payload["names"], ["w", "b"])
        self.assertEqual(
            list(payload["hyperparameters"].keys()), ["lr", "alpha", "eps"]
        )
        self.assertEqual(payload["hyperparameters"]["lr"], 0.01)
        self.assertEqual(payload["hyperparameters"]["alpha"], 0.9)
        self.assertEqual(payload["hyperparameters"]["eps"], 0.001)
        self.assertEqual(
            list(payload["state"].keys()), ["parameters", "square_avg"]
        )
        inner_start = text.index('"state":') + len('"state":')
        inner_end = text.rindex(',"digest":"')
        self.assertEqual(text[inner_start:inner_end], dump_rmsprop(optimizer))

    def test_exact_byte_form(self):
        w = Tensor(1.5, True)
        b = Tensor([-0.0, 2.0], False)
        optimizer = RMSprop([w, b], 0.1, alpha=0.5, eps=0.25)
        text = dump_checkpoint({"w": w, "b": b}, optimizer)
        self.assertEqual(
            text,
            '{"version":8,"type":"rmsprop","names":["w","b"],'
            '"hyperparameters":'
            '{"lr":0.100000,"alpha":0.500000,"eps":0.250000},'
            '"state":'
            '{"parameters":['
            '{"data":1.500000,"requires_grad":true},'
            '{"data":[0.000000,2.000000],"requires_grad":false}'
            '],"square_avg":[0.000000,[0.000000,0.000000]]},'
            '"digest":'
            '"754833bb47a79a1093dadcb553df9db11f446fb1f3d694d286aec04f'
            '33871cc6"}',
        )

    def test_six_decimals_no_exponent_no_negative_zero(self):
        tiny = Tensor(1e-9, True)
        huge = Tensor(1e20, True)
        negzero = Tensor(-0.0, False)
        optimizer = RMSprop([tiny, huge, negzero], 1.0)
        text = dump_checkpoint(
            {"tiny": tiny, "huge": huge, "negzero": negzero}, optimizer
        )
        self.assertIn("0.000000", text)
        self.assertIn("100000000000000000000.000000", text)
        self.assertNotIn("e+", text)
        self.assertNotIn("e-", text)
        self.assertNotIn("E", text)
        self.assertNotIn("-0.000000", text)

    def test_parameters_follow_dump_state_contract(self):
        _, optimizer = make_rmsprop_state()
        w, b = optimizer.parameters
        with self.assertRaises(TypeError):
            dump_checkpoint([("w", w), ("b", b)], optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({1: w, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"1": w, "b": b}, optimizer)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": 1.0, "b": b}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w, "x": w}, optimizer)

    def test_values_must_be_optimizer_parameters_in_order(self):
        _, optimizer = make_rmsprop_state()
        w, b = optimizer.parameters
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": b, "b": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint({"w": w}, optimizer)
        with self.assertRaises(ValueError):
            dump_checkpoint(
                {"w": Tensor(list(w.data), True), "b": b}, optimizer
            )

    def test_optimizer_parameters_must_be_a_list(self):
        w = Tensor([0.1], True)
        optimizer = RMSprop([w], 0.01)
        optimizer.parameters = tuple(optimizer.parameters)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, optimizer)

    def test_rejects_invalid_hyperparameters(self):
        w = Tensor([0.1], True)
        for attr, bad, error in (
            ("lr", 0.0, ValueError),
            ("lr", -0.5, ValueError),
            ("lr", float("inf"), ValueError),
            ("lr", 1, TypeError),
            ("alpha", -0.1, ValueError),
            ("alpha", 1.0, ValueError),
            ("alpha", float("nan"), ValueError),
            ("alpha", True, TypeError),
            ("eps", 0.0, ValueError),
            ("eps", -1e-8, ValueError),
            ("eps", float("inf"), ValueError),
            ("eps", "x", TypeError),
        ):
            with self.subTest(attr=attr, bad=bad):
                optimizer = RMSprop([w], 0.01)
                setattr(optimizer, attr, bad)
                with self.assertRaises(error):
                    dump_checkpoint({"w": w}, optimizer)


class RMSpropLoadRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_semantics(self):
        parameters, optimizer = make_rmsprop_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], False)
        qb = Tensor(9.0, True)
        target = RMSprop([qw, qb], 0.5, alpha=0.1, eps=0.25)
        qw.grad = [1.0, 1.0, 1.0]
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        # square_avg = 0.1 * g * g after one step from zero; data moves
        # by lr * g / (sqrt(square_avg) + eps); values are stored at
        # the six-decimal serialization precision.
        self.assertEqual(qw.data, [0.068625, -0.168576, 0.268543])
        # b had requires_grad False, so the step left it at 1.5.
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        self.assertEqual(target.square_avg, [[0.016, 0.025, 0.036], 0.0])
        # List/Tensor identities survive; hyperparameters are replaced.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.01)
        self.assertEqual(target.alpha, 0.9)
        self.assertEqual(target.eps, 0.001)

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_rmsprop_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.11, 0.22, -0.33], 0.44),
            ([0.5, -0.6, 0.7], -0.8),
            ([1.0, 1.0, 1.0], 2.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = RMSprop([w, b], 0.5, alpha=0.1, eps=0.25)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_dump_is_deterministic(self):
        text1 = dump_checkpoint(*make_rmsprop_state())
        text2 = dump_checkpoint(*make_rmsprop_state())
        self.assertEqual(text1, text2)

    def test_cross_kind_load_is_rejected(self):
        rmsprop_parameters, rmsprop_optimizer = make_rmsprop_state()
        rmsprop_text = dump_checkpoint(rmsprop_parameters, rmsprop_optimizer)
        sgd_parameters, sgd_optimizer = make_sgd_state()
        sgd_text = dump_checkpoint(sgd_parameters, sgd_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, SGD([w, b], 0.01), rmsprop_text
            )
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b},
                RMSprop([w, b], 0.01),
                sgd_text,
            )


class RMSpropLoadValidationTests(unittest.TestCase):

    def setUp(self):
        self.parameters, self.optimizer = make_rmsprop_state()
        self.text = dump_checkpoint(self.parameters, self.optimizer)

    def test_rejects_non_str_text(self):
        for bad in (None, b"x", 1, 1.0, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_unsupported_optimizer(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, object(), self.text)

    def test_optimizer_parameters_must_be_a_list(self):
        optimizer = RMSprop([Tensor([0.1], True)], 0.01)
        optimizer.parameters = tuple(optimizer.parameters)
        with self.assertRaises(TypeError):
            load_checkpoint(
                {"w": optimizer.parameters[0]}, optimizer, self.text
            )

    def test_rejects_bad_parameters(self):
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
        w, b = self.optimizer.parameters
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": b, "b": w}, self.optimizer, self.text
            )

    def test_rejects_malformed_text(self):
        payload = self.text.rsplit(',"digest":"', 1)[0] + "}"
        state = dump_rmsprop(self.optimizer)
        hyper = '"hyperparameters":{"lr":0.010000,"alpha":0.900000,"eps":0.001000}'
        bad_texts = (
            "",
            "{",
            "not json",
            "null",
            "[]",
            # Missing digest member entirely.
            payload,
            # Truncated/extended/uppercase/non-hex digests.
            self.text[:-3] + 'a"}',
            self.text[:-66] + "A" * 64 + '"}',
            self.text[:-66] + "g" * 64 + '"}',
            self.text[:-66] + "a" * 63 + '"}',
            self.text + "}",
            # Digest does not match the payload.
            redigest_v8(self.text)[:-66] + "0" * 64 + '"}',
            self.text.replace('"alpha":0.900000', '"alpha":0.800000'),
            # Top-level key order, missing and extra keys.
            redigest_v8(self.text).replace(
                '"version":8,"type":"rmsprop",', '"type":"rmsprop","version":8,'
            ),
            redigest_v8(self.text).replace('"type":"rmsprop",', ""),
            redigest_v8(self.text).replace(
                ',"hyperparameters"', ',"hyper"', 1
            ),
            redigest_v8(self.text).replace(
                ',"state"', ',"extra":0,"state"', 1
            ),
            # Hyperparameter member order, missing and extra keys.
            redigest_v8(self.text).replace(
                hyper,
                '"hyperparameters":{"alpha":0.900000,"lr":0.010000,"eps":0.001000}',
            ),
            redigest_v8(self.text).replace(
                hyper,
                '"hyperparameters":{"lr":0.010000,"alpha":0.900000}',
            ),
            redigest_v8(self.text).replace(
                hyper,
                '"hyperparameters":{"lr":0.010000,"alpha":0.900000,"eps":0.001000,"mu":0.100000}',
            ),
            # Names: missing, reordered, duplicated, wrong.
            redigest_v8(self.text).replace('["w","b"]', '["w"]'),
            redigest_v8(self.text).replace('["w","b"]', '["b","w"]'),
            redigest_v8(self.text).replace('["w","b"]', '["w","w"]'),
            redigest_v8(self.text).replace('["w","b"]', '["w","x"]'),
            # Version and type mismatches.
            redigest_v8(self.text).replace('"version":8', '"version":7'),
            redigest_v8(self.text).replace('"type":"rmsprop"', '"type":"sgd"'),
            redigest_v8(self.text).replace('"version":8', '"version":08'),
            # Lexical violations in hyperparameters.
            redigest_v8(self.text).replace('"lr":0.010000', '"lr":0.01000'),
            redigest_v8(self.text).replace('"lr":0.010000', '"lr":1e-2'),
            redigest_v8(self.text).replace('"lr":0.010000', '"lr": 0.010000'),
            redigest_v8(self.text).replace('"alpha":0.900000', '"alpha":-0.000000'),
            # Inner state key order and lexical violations.
            redigest_v8(self.text).replace(
                state,
                '{"square_avg":[[0.016000,0.025000,0.036000],0.000000],'
                '"parameters":[]}',
            ),
            redigest_v8(self.text).replace(
                '"data":[0.068625,-0.168576,0.268543]',
                '"data":[0.068625,-0.168576]',
            ),
            redigest_v8(self.text).replace(
                '"square_avg":[[0.016000,0.025000,0.036000],0.000000]',
                '"square_avg":[0.016000,0.000000]',
            ),
            self.text + " ",
            self.text + "\n",
            self.text[:-1],
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_rejects_bad_hyperparameters(self):
        bad_texts = (
            self.text.replace('"lr":0.010000', '"lr":0.000000'),
            self.text.replace('"lr":0.010000', '"lr":-0.010000'),
            self.text.replace('"alpha":0.900000', '"alpha":1.000000'),
            self.text.replace('"alpha":0.900000', '"alpha":-0.500000'),
            self.text.replace('"eps":0.001000', '"eps":0.000000'),
            self.text.replace('"eps":0.001000', '"eps":-0.001000'),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, redigest_v8(bad)
                    )

    def test_rejects_state_shape_mismatch(self):
        qw = Tensor(9.0, True)
        qb = Tensor([9.0, 9.0], False)
        target = RMSprop([qw, qb], 0.5)
        with self.assertRaises(ValueError):
            load_checkpoint({"w": qw, "b": qb}, target, self.text)
        with self.assertRaises(ValueError):
            load_checkpoint({"b": qb, "w": qw}, target, self.text)

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        b.grad = 0.9
        data_before = list(w.data)
        b_data_before = b.data
        grad_before = list(w.grad)
        square_avg_before = [
            list(slot) if isinstance(slot, list) else slot
            for slot in self.optimizer.square_avg
        ]
        lr_before = self.optimizer.lr
        alpha_before = self.optimizer.alpha
        eps_before = self.optimizer.eps
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":8', '"version":7'),
            self.text.replace('"type":"rmsprop"', '"type":"sgd"'),
            self.text.replace('"alpha":0.900000', '"alpha":0.800000'),
            redigest_v8(
                self.text.replace('"lr":0.010000', '"lr":-0.010000')
            ),
            redigest_v8(
                self.text.replace('"alpha":0.900000', '"alpha":1.000000')
            ),
            redigest_v8(
                self.text.replace('"eps":0.001000', '"eps":0.000000')
            ),
            redigest_v8(
                self.text.replace('"square_avg":[', '"square_avg":[9.9,', 1)
            ),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(b.data, b_data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(b.grad, 0.9)
                self.assertEqual(self.optimizer.square_avg, square_avg_before)
                self.assertIs(self.optimizer.lr, lr_before)
                self.assertIs(self.optimizer.alpha, alpha_before)
                self.assertIs(self.optimizer.eps, eps_before)


if __name__ == "__main__":
    unittest.main()