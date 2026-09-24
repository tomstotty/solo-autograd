import copy
import hashlib
import json
import unittest

import autograd as ag

Tensor = ag.Tensor


def inner_checkpoint(text):
    marker = '"checkpoint":'
    start = text.index(marker) + len(marker)
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("no checkpoint object found")


def reseal(text, new_inner):
    """Replace the inner checkpoint and recompute the v5 digest."""
    marker = ',"checkpoint":'
    head = text[: text.index(marker)] + marker
    payload = head + new_inner + "}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload[:-1] + ',"digest":"' + digest + '"}'


def make_amsgrad():
    a = Tensor(1.5, requires_grad=True)
    b = Tensor([1.0, -2.0, 0.25], requires_grad=False)
    opt = ag.AMSGrad([a, b], lr=0.01, beta1=0.8, beta2=0.95, eps=1e-7)
    a.grad = 0.5
    b.grad = None
    opt.step()
    a.grad = -0.25
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


class DumpAMSGradTests(unittest.TestCase):
    def test_textual_form(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 7, 42)
        parsed = json.loads(text)
        self.assertEqual(
            list(parsed.keys()),
            ["version", "global_step", "rng_state", "checkpoint", "digest"],
        )
        self.assertEqual(parsed["version"], 5)
        self.assertEqual(parsed["global_step"], 7)
        self.assertEqual(parsed["rng_state"], 42)
        self.assertFalse(text.endswith("\n"))
        self.assertNotIn(" ", text)
        cp = parsed["checkpoint"]
        self.assertEqual(list(cp.keys()), ["version", "names", "amsgrad"])
        self.assertEqual(cp["version"], 1)
        self.assertEqual(cp["names"], ["a", "b"])
        ams = cp["amsgrad"]
        self.assertEqual(
            list(ams.keys()), ["parameters", "m", "v", "v_max", "t"]
        )
        self.assertEqual(
            list(ams["parameters"][0].keys()), ["data", "requires_grad"]
        )
        self.assertEqual(ams["t"], opt.t)
        # Inner checkpoint bytes are exactly dump_checkpoint's.
        self.assertEqual(inner_checkpoint(text),
                         ag.dump_checkpoint(parameters, opt))

    def test_digest_covers_four_key_document(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 7, 42)
        parsed = json.loads(text)
        payload = text[: text.index(',"digest":')] + "}"
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(parsed["digest"], expected)
        self.assertEqual(len(parsed["digest"]), 64)
        self.assertEqual(parsed["digest"], parsed["digest"].lower())

    def test_versions_1_to_4_unchanged(self):
        rp, ropt = make_rmsprop()
        text = ag.dump_training_state(rp, ropt, 3, 9)
        parsed = json.loads(text)
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(
            list(parsed.keys()),
            ["version", "global_step", "rng_state", "checkpoint"],
        )
        self.assertEqual(parsed["checkpoint"]["version"], 8)
        self.assertEqual(inner_checkpoint(text),
                         ag.dump_checkpoint(rp, ropt))

    def test_rejects_other_optimizer(self):
        a = Tensor(1.0, requires_grad=True)
        other = ag.SGD([a], lr=0.1)
        parameters = {"a": a}
        with self.assertRaises(TypeError):
            ag.dump_training_state(parameters, other, 0, 0)
        with self.assertRaises(TypeError):
            ag.dump_training_state(parameters, ag.Adagrad([a], lr=0.1), 0, 0)

    def test_loop_value_types_and_ranges(self):
        parameters, opt = make_amsgrad()
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

    def test_amsgrad_hyperparameter_validation(self):
        parameters, opt = make_amsgrad()
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


class RoundTripAMSGradTests(unittest.TestCase):
    def test_round_trip_restores_state_and_returns_loop_tuple(self):
        parameters, opt = make_amsgrad()
        identities = [id(p) for p in opt.parameters]
        text = ag.dump_training_state(parameters, opt, 11, 123456)
        a, b = opt.parameters
        a.data = 99.0
        a.requires_grad = False
        a.grad = 7.0
        b.data = [9.0, 9.0, 9.0]
        b.grad = [1.0, 1.0, 1.0]
        opt.m = [0.0, [0.0, 0.0, 0.0]]
        opt.v = [0.0, [0.0, 0.0, 0.0]]
        opt.v_max = [0.0, [0.0, 0.0, 0.0]]
        opt.t = 0
        result = ag.load_training_state(parameters, opt, text)
        self.assertEqual(result, (11, 123456))
        self.assertTrue(all(isinstance(value, int) for value in result))
        expected = json.loads(text)
        state = expected["checkpoint"]["amsgrad"]
        self.assertAlmostEqual(
            a.data, state["parameters"][0]["data"]
        )
        self.assertEqual(a.requires_grad, True)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertEqual([id(p) for p in opt.parameters], identities)
        self.assertEqual(opt.t, 2)

    def test_round_trip_values(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 5, 6)
        state = json.loads(text)["checkpoint"]["amsgrad"]
        a, b = opt.parameters
        a.data = 0.0
        b.data = [0.0, 0.0, 0.0]
        opt.m = [0.0, [0.0, 0.0, 0.0]]
        opt.v = [0.0, [0.0, 0.0, 0.0]]
        opt.v_max = [0.0, [0.0, 0.0, 0.0]]
        opt.t = 0
        ag.load_training_state(parameters, opt, text)

        def close(x, y):
            if isinstance(x, list):
                return isinstance(y, list) and len(x) == len(y) and all(
                    abs(p - q) < 1e-9 for p, q in zip(x, y)
                )
            return abs(x - y) < 1e-9

        for slot in ("m", "v", "v_max"):
            self.assertTrue(close(getattr(opt, slot)[0], state[slot][0]))
            self.assertTrue(all(
                close(x, y)
                for x, y in zip(getattr(opt, slot)[1], state[slot][1])
            ))
        self.assertEqual(opt.t, state["t"])
        self.assertEqual(
            (opt.lr, opt.beta1, opt.beta2, opt.eps),
            (0.01, 0.8, 0.95, 1e-7),
        )
        self.assertEqual(a.requires_grad, True)
        self.assertEqual(b.requires_grad, False)

    def test_resume_is_byte_identical(self):
        parameters, opt = make_amsgrad()
        a, b = opt.parameters
        saved = ag.dump_training_state(parameters, opt, 4, 4242)
        opt.zero_grad()

        a2 = Tensor(a.data, requires_grad=a.requires_grad)
        b2 = Tensor(b.data, requires_grad=b.requires_grad)
        opt2 = ag.AMSGrad([a2, b2], lr=opt.lr, beta1=opt.beta1,
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
        parameters, opt = make_amsgrad()
        self.assertEqual(
            ag.dump_training_state(parameters, opt, 2, 3),
            ag.dump_training_state(parameters, opt, 2, 3),
        )

    def test_reordered_names_round_trip(self):
        bp = Tensor([1.0, -2.0, 0.25], requires_grad=False)
        ap = Tensor(1.5, requires_grad=True)
        opt_rev = ag.AMSGrad([bp, ap], lr=0.01, beta1=0.8, beta2=0.95,
                             eps=1e-7)
        ap.grad = 0.5
        opt_rev.step()
        reordered = ag.dump_training_state(
            {"b": bp, "a": ap}, opt_rev, 8, 77
        )
        self.assertIn('"names":["b","a"]', reordered)

        parameters, opt = make_amsgrad()
        a, b = opt.parameters
        a.data, b.data = 0.0, [0.0, 0.0, 0.0]
        opt.m = [0.0, [0.0, 0.0, 0.0]]
        opt.v = [0.0, [0.0, 0.0, 0.0]]
        opt.v_max = [0.0, [0.0, 0.0, 0.0]]
        opt.t = 0
        step, rng = ag.load_training_state(parameters, opt, reordered)
        self.assertEqual((step, rng), (8, 77))
        self.assertAlmostEqual(a.data, 1.49)
        self.assertEqual(b.data, [1.0, -2.0, 0.25])
        self.assertGreater(opt.m[0], 0.0)
        self.assertEqual(opt.m[1], [0.0, 0.0, 0.0])
        self.assertEqual(opt.t, 1)


class LoadValidationTests(unittest.TestCase):
    def setUp(self):
        self.parameters, self.opt = make_amsgrad()
        self.text = ag.dump_training_state(self.parameters, self.opt, 1, 2)

    def _inner(self):
        return ag.dump_checkpoint(self.parameters, self.opt)

    def test_rejects_non_str(self):
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, b"x")
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, None)
        with self.assertRaises(TypeError):
            ag.load_training_state(self.parameters, self.opt, 42)

    def test_rejects_other_optimizer(self):
        a = Tensor(1.0, requires_grad=True)
        with self.assertRaises(TypeError):
            ag.load_training_state(
                {"a": a}, ag.SGD([a], lr=0.1), self.text
            )
        # Adam is a supported optimizer, but a v5 text is an AMSGrad
        # checkpoint, so the mismatch surfaces as ValueError.
        with self.assertRaises(ValueError):
            ag.load_training_state(
                {"a": a}, ag.Adam([a], lr=0.1), self.text
            )

    def test_cross_version_mismatch(self):
        rp, ropt = make_rmsprop()
        v1 = ag.dump_training_state(rp, ropt, 0, 0)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, v1)
        with self.assertRaises(ValueError):
            ag.load_training_state(rp, ropt, self.text)

    def test_rejects_other_versions(self):
        for bad_version in ("0", "6", "99"):
            bad = self.text.replace('"version":5',
                                    '"version":' + bad_version, 1)
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_other_inner_version(self):
        rp, ropt = make_rmsprop()
        frankenstein = reseal(self.text, ag.dump_checkpoint(rp, ropt))
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, frankenstein)

    def test_rejects_bad_outer_form(self):
        variants = [
            self.text.replace('{"version":5', '{"version" :5'),
            self.text.replace('"global_step"', '"step"'),
            self.text.replace(',"rng_state"', ', "rng_state"'),
            self.text + " ",
            self.text[:-1],
            self.text.replace('{"version":5,', '{ "version":5,'),
            '{"version":5,"global_step":0,"rng_state":0}',
            '{"version":5,"global_step":0,"rng_state":0,"extra":1,'
            '"checkpoint":' + self._inner() + "}",
            '{"version":5,"rng_state":2,"global_step":1,'
            '"checkpoint":' + self._inner() + "}",
        ]
        for bad in variants:
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_integer_lexical_and_range_errors(self):
        inner = self._inner()
        prefix = '{"version":5,"global_step":'
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

    def test_rejects_bad_digest(self):
        bad = self.text[:-2] + ("0" if self.text[-2] != "0" else "1") + "}"
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)
        # A digest member missing altogether (four-key document only).
        payload = self.text[: self.text.index(',"digest":')] + "}"
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, payload)

    def test_rejects_inner_contract_violations(self):
        inner = self._inner()
        # Duplicated inner key: strict parser rejects; reseal the digest.
        bad_inner = inner.replace(
            '"v_max":', '"v_extra":[], "v_max":', 1
        )
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, bad_inner)
            )
        # Truncated inner state.
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, inner[:-2])
            )

    def test_rejects_negative_v_and_v_max(self):
        inner = self._inner()
        for slot in ('"v":', '"v_max":'):
            bad_inner = inner.replace(slot + "[0.0", slot + "[-0.5", 1)
            with self.assertRaises(ValueError, msg=slot):
                ag.load_training_state(
                    self.parameters, self.opt, reseal(self.text, bad_inner)
                )

    def test_rejects_v_max_below_v(self):
        inner = self._inner()
        # Force v above v_max for the scalar parameter.
        bad_inner = inner.replace('"v":[0.0', '"v":[0.5', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, bad_inner)
            )

    def test_rejects_negative_t(self):
        inner = self._inner()
        bad_inner = inner.replace('"t":2', '"t":-1', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, bad_inner)
            )

    def test_name_mismatch(self):
        inner = self._inner()
        renamed = inner.replace('"names":["a","b"]', '"names":["a","c"]')
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, renamed)
            )

    def test_duplicate_names_rejected(self):
        inner = self._inner()
        renamed = inner.replace('"names":["a","b"]', '"names":["a","a"]')
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, renamed)
            )

    def test_failed_load_changes_nothing(self):
        a, b = self.opt.parameters
        before = (
            a.data, b.data, copy.deepcopy(self.opt.m),
            copy.deepcopy(self.opt.v), copy.deepcopy(self.opt.v_max),
            self.opt.t, a.requires_grad, b.requires_grad,
        )
        bad_inner = self._inner().replace('"v":[0.0', '"v":[0.5', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, reseal(self.text, bad_inner)
            )
        self.assertEqual(a.data, before[0])
        self.assertEqual(b.data, before[1])
        self.assertEqual(self.opt.m, before[2])
        self.assertEqual(self.opt.v, before[3])
        self.assertEqual(self.opt.v_max, before[4])
        self.assertEqual(self.opt.t, before[5])
        self.assertEqual(a.requires_grad, before[6])
        self.assertEqual(b.requires_grad, before[7])

    def test_hyperparameters_kept_on_load(self):
        a, b = self.opt.parameters
        opt2 = ag.AMSGrad([a, b], lr=0.5, beta1=0.1, beta2=0.2, eps=9.0)
        ag.load_training_state(self.parameters, opt2, self.text)
        self.assertEqual(
            (opt2.lr, opt2.beta1, opt2.beta2, opt2.eps),
            (0.5, 0.1, 0.2, 9.0),
        )


if __name__ == "__main__":
    unittest.main()
