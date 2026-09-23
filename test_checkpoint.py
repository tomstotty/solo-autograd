"""Contract tests for dump_checkpoint/load_checkpoint (AMSGrad only).

Standard library only; discovered by ``python -m unittest discover``.
"""

import json
import unittest

from autograd import (
    AMSGrad,
    Adam,
    Tensor,
    dump_amsgrad,
    dump_checkpoint,
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

    def test_rejects_non_amsgrad(self):
        parameters, optimizer = make_state()
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            dump_checkpoint({"w": w}, Adam([w], 0.01))

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

    def test_rejects_non_amsgrad(self):
        w = Tensor([0.1], True)
        with self.assertRaises(TypeError):
            load_checkpoint(
                {"w": w}, Adam([w], 0.01), self.text
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


if __name__ == "__main__":
    unittest.main()
