import copy
import json
import unittest

import autograd as ag

Tensor = ag.Tensor


def inner_checkpoint(text):
    marker = '"checkpoint":'
    start = text.index(marker) + len(marker)
    return text[start:-1]


def make_adamax():
    a = Tensor(1.5, requires_grad=True)
    b = Tensor([1.0, -2.0, 0.25], requires_grad=False)
    opt = ag.Adamax([a, b], lr=0.01, beta1=0.8, beta2=0.95, eps=1e-7)
    a.grad = 0.5
    b.grad = None
    opt.step()
    return {"a": a, "b": b}, opt


def make_rmsprop():
    a = Tensor(1.5, requires_grad=True)
    b = Tensor([1.0, -2.0, 0.25], requires_grad=True)
    opt = ag.RMSprop([a, b], lr=0.01, alpha=0.9, eps=1e-7)
    a.grad = 0.5
    b.grad = [0.1, -0.2, 0.3]
    opt.step()
    return {"a": a, "b": b}, opt


class DumpAdamaxTests(unittest.TestCase):
    def test_textual_form(self):
        parameters, opt = make_adamax()
        text = ag.dump_training_state(parameters, opt, 7, 42)
        parsed = json.loads(text)
        self.assertEqual(
            list(parsed.keys()),
            ["version", "global_step", "rng_state", "checkpoint"],
        )
        self.assertEqual(parsed["version"], 2)
        self.assertEqual(parsed["global_step"], 7)
        self.assertEqual(parsed["rng_state"], 42)
        cp = parsed["checkpoint"]
        self.assertEqual(cp["version"], 4)
        self.assertEqual(cp["type"], "adamax")
        self.assertEqual(cp["names"], ["a", "b"])
        self.assertEqual(list(cp["state"].keys()), ["parameters", "m", "u", "t"])
        self.assertEqual(cp["state"]["t"], opt.t)
        self.assertFalse(text.endswith("\n"))
        self.assertNotIn(" ", text)
        # Inner checkpoint bytes are exactly dump_checkpoint's.
        inner = inner_checkpoint(text)
        self.assertEqual(inner, ag.dump_checkpoint(parameters, opt))

    def test_inner_bytes_match_version4(self):
        parameters, opt = make_adamax()
        text = ag.dump_training_state(parameters, opt, 0, 0)
        inner = inner_checkpoint(text)
        self.assertTrue(inner.startswith('{"version":4,"type":"adamax"'))

    def test_rmsprop_still_version1(self):
        parameters, opt = make_rmsprop()
        text = ag.dump_training_state(parameters, opt, 3, 9)
        parsed = json.loads(text)
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["checkpoint"]["version"], 8)
        self.assertEqual(parsed["checkpoint"]["type"], "rmsprop")
        inner = inner_checkpoint(text)
        self.assertEqual(inner, ag.dump_checkpoint(parameters, opt))

    def test_rejects_other_optimizer(self):
        a = Tensor(1.0, requires_grad=True)
        other = ag.SGD([a], lr=0.1)
        parameters = {"a": a}
        with self.assertRaises(TypeError):
            ag.dump_training_state(parameters, other, 0, 0)

    def test_loop_value_types_and_ranges(self):
        parameters, opt = make_adamax()
        for bad in (True, False, 1.0, "0", None, 1 + 0j):
            with self.assertRaises(TypeError):
                ag.dump_training_state(parameters, opt, bad, 0)
            with self.assertRaises(TypeError):
                ag.dump_training_state(parameters, opt, 0, bad)
        with self.assertRaises(ValueError):
            ag.dump_training_state(parameters, opt, -1, 0)
        with self.assertRaises(ValueError):
            ag.dump_training_state(parameters, opt, 0, -1)
        with self.assertRaises(ValueError):
            ag.dump_training_state(parameters, opt, 0, 4294967296)
        # Boundaries accepted.
        ag.dump_training_state(parameters, opt, 0, 0)
        ag.dump_training_state(parameters, opt, 10 ** 40, 4294967295)

    def test_adamax_hyperparameter_validation(self):
        parameters, opt = make_adamax()
        cases = [
            ("lr", True, TypeError), ("lr", 0.0, ValueError),
            ("lr", float("nan"), ValueError), ("lr", -1.0, ValueError),
            ("beta1", True, TypeError), ("beta1", 1.0, ValueError),
            ("beta1", -0.1, ValueError), ("beta1", float("inf"), ValueError),
            ("beta2", 1, TypeError), ("beta2", 1.0, ValueError),
            ("eps", 0.0, ValueError), ("eps", -1e-9, ValueError),
            ("eps", True, TypeError),
        ]
        for attr, value, error in cases:
            original = getattr(opt, attr)
            setattr(opt, attr, value)
            with self.assertRaises(error):
                ag.dump_training_state(parameters, opt, 0, 0)
            setattr(opt, attr, original)


class RoundTripAdamaxTests(unittest.TestCase):
    def test_round_trip_restores_state_and_returns_loop_tuple(self):
        parameters, opt = make_adamax()
        identities = [id(p) for p in opt.parameters]
        text = ag.dump_training_state(parameters, opt, 11, 123456)
        a, b = opt.parameters
        a.data = 99.0
        a.requires_grad = False
        a.grad = 7.0
        b.data = [9.0, 9.0, 9.0]
        b.grad = [1.0, 1.0, 1.0]
        opt.m = [0.0, [0.0, 0.0, 0.0]]
        opt.u = [0.0, [0.0, 0.0, 0.0]]
        opt.t = 0
        result = ag.load_training_state(parameters, opt, text)
        self.assertEqual(result, (11, 123456))
        self.assertTrue(all(isinstance(v, int) for v in result))
        expected = json.loads(text)
        self.assertAlmostEqual(a.data, expected["checkpoint"]["state"]["parameters"][0]["data"])
        self.assertEqual(a.requires_grad, True)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertEqual([id(p) for p in opt.parameters], identities)
        self.assertEqual(opt.t, 1)

    def test_round_trip_values(self):
        parameters, opt = make_adamax()
        text = ag.dump_training_state(parameters, opt, 5, 6)
        state = json.loads(text)["checkpoint"]["state"]
        a, b = opt.parameters
        a.data = 0.0
        b.data = [0.0, 0.0, 0.0]
        opt.m = [0.0, [0.0, 0.0, 0.0]]
        opt.u = [0.0, [0.0, 0.0, 0.0]]
        opt.t = 0
        ag.load_training_state(parameters, opt, text)
        # The six-decimal checkpoint contract rounds; compare with the
        # serialized values rather than the pre-dump floats.
        def close(x, y):
            if isinstance(x, list):
                return isinstance(y, list) and len(x) == len(y) and all(
                    abs(p - q) < 1e-9 for p, q in zip(x, y)
                )
            return abs(x - y) < 1e-9

        self.assertTrue(close(opt.m[0], state["m"][0]))
        self.assertTrue(all(
            close(x, y) for x, y in zip(opt.m[1], state["m"][1])
        ))
        self.assertTrue(close(opt.u[0], state["u"][0]))
        self.assertTrue(all(
            close(x, y) for x, y in zip(opt.u[1], state["u"][1])
        ))
        self.assertEqual(opt.t, state["t"])
        self.assertEqual(
            (opt.lr, opt.beta1, opt.beta2, opt.eps),
            (0.01, 0.8, 0.95, 1e-7),
        )
        self.assertEqual(a.requires_grad, True)
        self.assertEqual(b.requires_grad, False)

    def test_resume_is_byte_identical(self):
        parameters, opt = make_adamax()
        a, b = opt.parameters
        a.grad = -0.75
        b.grad = [0.4, 0.0, -0.9]
        opt.step()
        saved = ag.dump_training_state(parameters, opt, 4, 4242)
        opt.zero_grad()

        # Resume into a fresh optimizer with identical structure/hyperparams.
        a2 = Tensor(a.data, requires_grad=a.requires_grad)
        b2 = Tensor(b.data, requires_grad=b.requires_grad)
        opt2 = ag.Adamax([a2, b2], lr=opt.lr, beta1=opt.beta1,
                         beta2=opt.beta2, eps=opt.eps)
        p2 = {"a": a2, "b": b2}
        global_step, rng = ag.load_training_state(p2, opt2, saved)
        self.assertEqual((global_step, rng), (4, 4242))
        # Continue with the same gradients.
        a2.grad = -0.75
        b2.grad = [0.4, 0.0, -0.9]
        opt2.step()
        a.grad = -0.75
        b.grad = [0.4, 0.0, -0.9]
        opt.step()
        again_original = ag.dump_training_state(parameters, opt, 5, 4242)
        again_resumed = ag.dump_training_state(p2, opt2, 5, 4242)
        self.assertEqual(again_original, again_resumed)

    def test_dump_is_deterministic(self):
        parameters, opt = make_adamax()
        self.assertEqual(
            ag.dump_training_state(parameters, opt, 2, 3),
            ag.dump_training_state(parameters, opt, 2, 3),
        )


class LoadValidationTests(unittest.TestCase):
    def setUp(self):
        self.parameters, self.opt = make_adamax()
        self.text = ag.dump_training_state(self.parameters, self.opt, 1, 2)

    def test_rejects_non_str(self):
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, b"x")
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, None)
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, 42)

    def test_rejects_other_optimizer(self):
        a = Tensor(1.0, requires_grad=True)
        other = ag.SGD([a], lr=0.1)
        with self.assertRaises(TypeError):
            ag.load_training_state({"a": a}, other, self.text)
        # Adam is a supported optimizer, but a v2 text is an Adamax
        # checkpoint, so the mismatch surfaces as ValueError.
        adam = ag.Adam([a], lr=0.1)
        with self.assertRaises(ValueError):
            ag.load_training_state({"a": a}, adam, self.text)

    def test_cross_version_mismatch(self):
        rp, ropt = make_rmsprop()
        v1 = ag.dump_training_state(rp, ropt, 0, 0)
        # v1 text into Adamax target -> ValueError
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, v1)
        # v2 text into RMSprop target -> ValueError
        with self.assertRaises(ValueError):
            ag.load_training_state(rp, ropt, self.text)

    def test_rejects_version3(self):
        bad = self.text.replace('"version":2', '"version":3', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_other_inner_version(self):
        # v2 wrapper but inner checkpoint edited to version 8 bytes.
        rp, ropt = make_rmsprop()
        inner8 = ag.dump_checkpoint(rp, ropt)
        head = self.text[: self.text.index(',"checkpoint":')]
        frankenstein = head + ',"checkpoint":' + inner8 + "}"
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, frankenstein)

    def test_rejects_bad_outer_form(self):
        variants = [
            self.text.replace('{"version":2', '{"version" :2'),
            self.text.replace('"global_step"', '"step"'),
            self.text.replace(',"rng_state"', ', "rng_state"'),
            self.text + " ",
            self.text[:-1],
            self.text.replace('{"version":2,', '{ "version":2,'),
            '{"version":2,"global_step":0,"rng_state":0}',
            '{"version":2,"global_step":0,"rng_state":0,"extra":1,'
            '"checkpoint":' + self._inner() + "}",
            '{"version":2,"rng_state":2,"global_step":1,'
            '"checkpoint":' + self._inner() + "}",
        ]
        for bad in variants:
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def _inner(self):
        return ag.dump_checkpoint(self.parameters, self.opt)

    def test_rejects_integer_lexical_and_range_errors(self):
        inner = self._inner()
        prefix = '{"version":2,"global_step":'
        cases = [
            prefix + '-1,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '01,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '1.0,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '0,"rng_state":4294967296,"checkpoint":' + inner + "}",
            prefix + '0,"rng_state":00,"checkpoint":' + inner + "}",
        ]
        for bad in cases:
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_inner_contract_violations(self):
        inner = self._inner()
        # Tamper with inner bytes; digest-free v4 still strictly parsed.
        bad_inner = inner.replace('"adamax"', '"adam"')
        head = self.text[: self.text.index('"checkpoint":') + len('"checkpoint":')]
        tail_index = self.text.index(inner)
        bad = self.text[:tail_index] + bad_inner + self.text[tail_index + len(inner):]
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)
        # Truncated inner
        bad2 = head + inner[:-2] + "}}"
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad2)

    def test_name_mismatch(self):
        inner = self._inner()
        renamed = inner.replace('"names":["a","b"]', '"names":["a","c"]')
        bad = self.text.replace(inner, renamed)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)

    def test_failed_load_changes_nothing(self):
        a, b = self.opt.parameters
        before = (
            a.data, b.data, copy.deepcopy(self.opt.m),
            copy.deepcopy(self.opt.u), self.opt.t, a.requires_grad,
            b.requires_grad,
        )
        bad = self.text.replace('"adamax"', '"sgd"')
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)
        self.assertEqual(a.data, before[0])
        self.assertEqual(b.data, before[1])
        self.assertEqual(self.opt.m, before[2])
        self.assertEqual(self.opt.u, before[3])
        self.assertEqual(self.opt.t, before[4])
        self.assertEqual(a.requires_grad, before[5])
        self.assertEqual(b.requires_grad, before[6])

    def test_hyperparameters_kept_on_load(self):
        a, b = self.opt.parameters
        opt2 = ag.Adamax([a, b], lr=0.5, beta1=0.1, beta2=0.2, eps=9.0)
        ag.load_training_state(self.parameters, opt2, self.text)
        self.assertEqual((opt2.lr, opt2.beta1, opt2.beta2, opt2.eps),
                         (0.5, 0.1, 0.2, 9.0))

    def test_reordered_names_round_trip(self):
        # Build a genuine v4 checkpoint whose names order is [b, a] with
        # correspondingly ordered state entries, using real dump bytes.
        bp = Tensor([1.0, -2.0, 0.25], requires_grad=False)
        ap = Tensor(1.5, requires_grad=True)
        opt_rev = ag.Adamax([bp, ap], lr=0.01, beta1=0.8, beta2=0.95,
                            eps=1e-7)
        ap.grad = 0.5
        bp.grad = None
        opt_rev.step()
        reordered = ag.dump_training_state(
            {"b": bp, "a": ap}, opt_rev, 8, 77
        )
        self.assertIn('"names":["b","a"]', reordered)

        a, b = self.opt.parameters
        a.data, b.data = 0.0, [0.0, 0.0, 0.0]
        self.opt.t = 0
        step, rng = ag.load_training_state(
            self.parameters, self.opt, reordered
        )
        self.assertEqual((step, rng), (8, 77))
        self.assertAlmostEqual(a.data, 1.49)
        self.assertEqual(b.data, [1.0, -2.0, 0.25])
        self.assertAlmostEqual(self.opt.m[0], 0.1)
        self.assertEqual(self.opt.t, 1)


if __name__ == "__main__":
    unittest.main()
