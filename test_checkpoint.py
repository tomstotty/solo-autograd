"""Contract tests for dump_checkpoint/load_checkpoint (AMSGrad v1, Adam v2).

Standard library only; discovered by ``python -m unittest discover``.
"""

import json
import unittest

from autograd import (
    AMSGrad,
    SGD,
    Adam,
    Tensor,
    dump_adam,
    dump_amsgrad,
    dump_checkpoint,
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


def make_sgd_state():
    w = Tensor([0.1, -0.2, 0.3], True)
    b = Tensor(1.5, False)
    optimizer = SGD([w, b], 0.01)
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
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, None)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, SGD)

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
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, None, self.text)
        with self.assertRaises(TypeError):
            load_checkpoint({"w": w}, SGD, self.text)

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
        inner_start = text.index('"state":') + len('"state":')
        self.assertEqual(text[inner_start:-1], dump_sgd(optimizer))

    def test_exact_bytes(self):
        a = Tensor(1.5, True)
        b = Tensor([-0.0, 2.0], False)
        optimizer = SGD([a, b], 0.1)
        text = dump_checkpoint({"a": a, "b": b}, optimizer)
        self.assertEqual(
            text,
            '{"version":5,"type":"sgd","names":["a","b"],"state":'
            '{"parameters":[{"data":1.500000,"requires_grad":true},'
            '{"data":[0.000000,2.000000],"requires_grad":false}]}}',
        )

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


class SGDRoundTripTests(unittest.TestCase):

    def test_round_trip_restores_semantics(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        qw = Tensor([9.0, 9.0, 9.0], True)
        qb = Tensor(9.0, True)
        target = SGD([qw, qb], 0.5)
        result = load_checkpoint({"w": qw, "b": qb}, target, text)
        self.assertIsNone(result)
        self.assertEqual(qw.data, [0.096, -0.195, 0.294])
        self.assertEqual(qb.data, 1.5)
        self.assertTrue(qw.requires_grad)
        self.assertFalse(qb.requires_grad)
        self.assertIsNone(qw.grad)
        self.assertIsNone(qb.grad)
        # Identity and lr survive; SGD has no other slots.
        self.assertIs(target.parameters[0], qw)
        self.assertIs(target.parameters[1], qb)
        self.assertEqual(target.lr, 0.5)

    def test_resume_is_byte_identical_from_shared_text(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        gradients = [
            ([0.125, 0.25, -0.125], 0.5),
            ([0.25, -0.5, 0.75], -0.25),
            ([0.375, 0.625, -0.875], 1.0),
        ]

        def branch():
            w = Tensor([0.0, 0.0, 0.0], True)
            b = Tensor(0.0, False)
            opt = SGD([w, b], 0.125)
            load_checkpoint({"w": w, "b": b}, opt, text)
            for grad_w, grad_b in gradients:
                w.grad = list(grad_w)
                b.grad = grad_b
                opt.step()
            return dump_checkpoint({"w": w, "b": b}, opt)

        self.assertEqual(branch(), branch())

    def test_self_round_trip_is_stable(self):
        parameters, optimizer = make_sgd_state()
        text = dump_checkpoint(parameters, optimizer)
        self.assertIsNone(load_checkpoint(parameters, optimizer, text))
        self.assertEqual(dump_checkpoint(parameters, optimizer), text)

    def test_cross_kind_load_is_rejected(self):
        sgd_parameters, sgd_optimizer = make_sgd_state()
        sgd_text = dump_checkpoint(sgd_parameters, sgd_optimizer)
        adam_parameters, adam_optimizer = make_adam_state()
        adam_text = dump_checkpoint(adam_parameters, adam_optimizer)
        w = Tensor([0.0, 0.0, 0.0], True)
        b = Tensor(0.0, False)
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, SGD([w, b], 0.01), adam_text
            )
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, Adam([w, b], 0.01), sgd_text
            )
        # Version 1 AMSGrad text into SGD is rejected as well.
        ams_text = dump_checkpoint(*make_state())
        with self.assertRaises(ValueError):
            load_checkpoint(
                {"w": w, "b": b}, SGD([w, b], 0.01), ams_text
            )


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
        w, b = self.optimizer.parameters
        with self.assertRaises(TypeError):
            load_checkpoint(
                list(self.parameters.items()), self.optimizer, self.text
            )
        with self.assertRaises(ValueError):
            load_checkpoint({}, self.optimizer, self.text)
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
            # Reordered top-level keys.
            '{"type":"sgd","version":5,"names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":5,"names":["w","b"],"type":"sgd","state":'
            + sgd
            + "}",
            # Wrong type literal / version.
            '{"version":5,"type":"adam","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":2,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            '{"version":6,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + "}",
            # Wrong key names / extra keys.
            '{"version":5,"type":"sgd","names":["w","b"],"amsgrad":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd
            + ',"extra":0}',
            # Name count/order problems.
            '{"version":5,"type":"sgd","names":["w"],"state":' + sgd + "}",
            '{"version":5,"type":"sgd","names":["b","w"],"state":'
            + sgd
            + "}",
            '{"version":5,"type":"sgd","names":["w","b","w"],"state":'
            + sgd
            + "}",
            # Empty names array.
            '{"version":5,"type":"sgd","names":[],"state":'
            '{"parameters":[]}}',
            # Embedded state lexically / structurally invalid.
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            '{"parameters":[]}}',
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd.replace(
                '"requires_grad":true', '"requires_grad":True', 1
            )
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd.replace('1.500000', '-0.000000', 1)
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd.replace('"data":1.500000', '"data":1.5', 1)
            + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"state":'
            + sgd.replace('1.500000', '1.50000e0', 1)
            + "}",
            self.text + " ",
            self.text + "\n",
            self.text[:-1],
            self.text + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_duplicate_top_level_keys_rejected(self):
        sgd = dump_sgd(self.optimizer)
        bad_texts = (
            '{"version":5,"version":5,"type":"sgd","names":["w","b"],'
            '"state":' + sgd + "}",
            '{"version":5,"type":"sgd","type":"sgd","names":["w","b"],'
            '"state":' + sgd + "}",
            '{"version":5,"type":"sgd","names":["w","b"],"names":["w","b"],'
            '"state":' + sgd + "}",
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )

    def test_names_must_match_target_in_order(self):
        # V5 binds positionally like v1-v3, not by name like v4.
        reordered = self.text.replace('["w","b"]', '["b","w"]')
        with self.assertRaises(ValueError):
            load_checkpoint(self.parameters, self.optimizer, reordered)

    def test_failed_load_changes_nothing(self):
        w, b = self.optimizer.parameters
        w.grad = [0.9, 0.9, 0.9]
        b.grad = 0.5
        data_before = list(w.data)
        grad_before = list(w.grad)
        b_data_before = b.data
        b_grad_before = b.grad
        lr_before = self.optimizer.lr
        parameter_objects = list(self.optimizer.parameters)
        bad_texts = (
            "{",
            self.text.replace('["w","b"]', '["x","b"]'),
            self.text.replace('"version":5', '"version":4'),
            self.text.replace('"type":"sgd"', '"type":"adam"'),
        )
        for bad in bad_texts:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        self.parameters, self.optimizer, bad
                    )
                self.assertEqual(w.data, data_before)
                self.assertEqual(w.grad, grad_before)
                self.assertEqual(b.data, b_data_before)
                self.assertEqual(b.grad, b_grad_before)
                self.assertEqual(self.optimizer.lr, lr_before)
                self.assertIs(
                    self.optimizer.parameters[0], parameter_objects[0]
                )
                self.assertIs(
                    self.optimizer.parameters[1], parameter_objects[1]
                )

    def test_load_clears_existing_grads(self):
        w, b = self.optimizer.parameters
        w.grad = [3.0, 3.0, 3.0]
        b.grad = 4.0
        self.assertIsNone(
            load_checkpoint(self.parameters, self.optimizer, self.text)
        )
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)


if __name__ == "__main__":
    unittest.main()