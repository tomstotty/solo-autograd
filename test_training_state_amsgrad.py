"""Contract tests for dump_training_state/load_training_state with AMSGrad.

Covers the version-5 training state wrapping the version-1 AMSGrad
checkpoint: textual form, digest, round-trip semantics, identity and
hyperparameter preservation, and strict parse/validation/atomicity
behaviour. Standard library only; discovered by
``python -m unittest discover``.
"""

import copy
import hashlib
import json
import unittest

import autograd as ag

Tensor = ag.Tensor


def make_amsgrad():
    w = Tensor([0.1, -0.2, 0.3], requires_grad=True)
    b = Tensor(1.5, requires_grad=False)
    opt = ag.AMSGrad([w, b], lr=0.01, beta1=0.8, beta2=0.9, eps=1e-6)
    w.grad = [0.4, -0.5, 0.6]
    b.grad = 0.7
    opt.step()
    return {"w": w, "b": b}, opt


def make_rmsprop():
    a = Tensor(1.5, requires_grad=True)
    b = Tensor([1.0, -2.0, 0.25], requires_grad=True)
    opt = ag.RMSprop([a, b], lr=0.01, alpha=0.9, eps=1e-7)
    a.grad = 0.5
    b.grad = [0.1, -0.2, 0.3]
    opt.step()
    return {"a": a, "b": b}, opt


def four_key_doc(text):
    """Return the signed four-key document behind a v3/v4/v5 dump."""
    return text[: text.index(',"digest":')] + "}"


def inner_checkpoint(text):
    start = text.index('"checkpoint":') + len('"checkpoint":')
    end = text.index(',"digest":')
    return text[start:end]


def resign(checkpoint_text, step=1, rng=2):
    """Wrap a (possibly tampered) checkpoint in a valid v5 document."""
    payload = (
        '{"version":5,"global_step":'
        + str(step)
        + ',"rng_state":'
        + str(rng)
        + ',"checkpoint":'
        + checkpoint_text
        + "}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload[:-1] + ',"digest":"' + digest + '"}'


def six(value):
    """Format a float exactly as dump_* does: six decimals, no -0."""
    text = f"{float(value):.6f}"
    return "0.000000" if text == "-0.000000" else text


def dump_six(node):
    """Serialize a parsed amsgrad state back in the exact compact form."""
    if isinstance(node, dict):
        return "{" + ",".join(
            '"' + key + '":' + dump_six(node[key]) for key in node
        ) + "}"
    if isinstance(node, list):
        return "[" + ",".join(dump_six(item) for item in node) + "]"
    if isinstance(node, bool):
        return "true" if node else "false"
    if isinstance(node, int):
        return str(node)
    if isinstance(node, float):
        return six(node)
    raise TypeError(type(node))


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
        self.assertIsInstance(parsed["version"], int)
        self.assertEqual(parsed["global_step"], 7)
        self.assertEqual(parsed["rng_state"], 42)
        cp = parsed["checkpoint"]
        self.assertEqual(list(cp.keys()), ["version", "names", "amsgrad"])
        self.assertEqual(cp["version"], 1)
        self.assertEqual(cp["names"], ["w", "b"])
        state = cp["amsgrad"]
        self.assertEqual(
            list(state.keys()), ["parameters", "m", "v", "v_max", "t"]
        )
        self.assertEqual(
            [list(entry.keys()) for entry in state["parameters"]],
            [["data", "requires_grad"], ["data", "requires_grad"]],
        )
        self.assertEqual(state["t"], opt.t)
        self.assertFalse(text.endswith("\n"))
        for whitespace in (" ", "\n", "\t", "\r"):
            self.assertNotIn(whitespace, text)
        self.assertEqual(
            inner_checkpoint(text), ag.dump_checkpoint(parameters, opt)
        )

    def test_inner_is_version1_amsgrad(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 0, 0)
        inner = inner_checkpoint(text)
        self.assertTrue(inner.startswith('{"version":1,"names":['))
        self.assertIn('"amsgrad":', inner)

    def test_digest_signs_four_key_document(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 9, 4242)
        parsed = json.loads(text)
        payload = four_key_doc(text)
        self.assertEqual(
            parsed["digest"],
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        self.assertRegex(parsed["digest"], r"^[0-9a-f]{64}$")

    def test_digest_changes_with_loop_state(self):
        parameters, opt = make_amsgrad()
        a = ag.dump_training_state(parameters, opt, 1, 0)
        b = ag.dump_training_state(parameters, opt, 2, 0)
        self.assertNotEqual(json.loads(a)["digest"], json.loads(b)["digest"])

    def test_dump_is_deterministic(self):
        parameters, opt = make_amsgrad()
        self.assertEqual(
            ag.dump_training_state(parameters, opt, 2, 3),
            ag.dump_training_state(parameters, opt, 2, 3),
        )

    def test_rejects_other_optimizer(self):
        w = Tensor(1.0, requires_grad=True)
        for other in (
            ag.SGD([w], lr=0.1),
            ag.Adagrad([w], lr=0.1),
            object(),
        ):
            with self.subTest(other=type(other).__name__):
                with self.assertRaises(TypeError):
                    ag.dump_training_state({"w": w}, other, 0, 0)
        # Adam is supported (version 3), but a v5 AMSGrad text aimed at
        # it is rejected at load time, not dump time.
        adam = ag.Adam([w], lr=0.1)
        self.assertEqual(
            json.loads(ag.dump_training_state({"w": w}, adam, 0, 0))[
                "version"
            ],
            3,
        )

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

    def test_versions_one_to_four_unchanged(self):
        # AMSGrad is brand new at version 5; RMSprop stays version 1 and
        # carries no digest.
        rp, ropt = make_rmsprop()
        text = ag.dump_training_state(rp, ropt, 3, 9)
        parsed = json.loads(text)
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["checkpoint"]["version"], 8)
        self.assertNotIn("digest", parsed)
        self.assertEqual(
            list(parsed.keys()),
            ["version", "global_step", "rng_state", "checkpoint"],
        )


class RoundTripAMSGradTests(unittest.TestCase):
    def test_returns_loop_tuple_of_ints(self):
        parameters, opt = make_amsgrad()
        text = ag.dump_training_state(parameters, opt, 11, 123456)
        result = ag.load_training_state(parameters, opt, text)
        self.assertEqual(result, (11, 123456))
        self.assertIsInstance(result, tuple)
        self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool)
                            for v in result))

    def test_restores_data_flags_slots_and_clears_grads(self):
        parameters, opt = make_amsgrad()
        identities = [id(p) for p in opt.parameters]
        hyper = (opt.lr, opt.beta1, opt.beta2, opt.eps)
        text = ag.dump_training_state(parameters, opt, 1, 2)
        state = json.loads(text)["checkpoint"]["amsgrad"]
        w, b = opt.parameters
        w.data = [9.0, 9.0, 9.0]
        w.requires_grad = False
        w.grad = [7.0, 7.0, 7.0]
        b.data = 8.0
        b.requires_grad = True
        b.grad = 7.0
        opt.m = [[0.0, 0.0, 0.0], 0.0]
        opt.v = [[0.0, 0.0, 0.0], 0.0]
        opt.v_max = [[0.0, 0.0, 0.0], 0.0]
        opt.t = 0
        ag.load_training_state(parameters, opt, text)
        self.assertEqual(w.data, state["parameters"][0]["data"])
        self.assertEqual(b.data, state["parameters"][1]["data"])
        self.assertTrue(w.requires_grad)
        self.assertFalse(b.requires_grad)
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)
        self.assertEqual(opt.m, state["m"])
        self.assertEqual(opt.v, state["v"])
        self.assertEqual(opt.v_max, state["v_max"])
        self.assertEqual(opt.t, state["t"])
        self.assertEqual([id(p) for p in opt.parameters], identities)
        self.assertEqual((opt.lr, opt.beta1, opt.beta2, opt.eps), hyper)

    def test_load_keeps_target_hyperparameters(self):
        parameters, opt = make_amsgrad()
        w, b = opt.parameters
        target = ag.AMSGrad([w, b], lr=0.5, beta1=0.1, beta2=0.2, eps=9.0)
        ag.load_training_state(parameters, target,
                               ag.dump_training_state(parameters, opt, 0, 0))
        self.assertEqual(
            (target.lr, target.beta1, target.beta2, target.eps),
            (0.5, 0.1, 0.2, 9.0),
        )

    def test_resume_is_byte_identical(self):
        parameters, opt = make_amsgrad()
        w, b = opt.parameters
        w.grad = [-0.75, 0.4, 0.0]
        b.grad = -0.9
        opt.step()
        saved = ag.dump_training_state(parameters, opt, 4, 4242)
        opt.zero_grad()

        w2 = Tensor(w.data, requires_grad=w.requires_grad)
        b2 = Tensor(b.data, requires_grad=b.requires_grad)
        opt2 = ag.AMSGrad([w2, b2], lr=opt.lr, beta1=opt.beta1,
                          beta2=opt.beta2, eps=opt.eps)
        p2 = {"w": w2, "b": b2}
        step, rng = ag.load_training_state(p2, opt2, saved)
        self.assertEqual((step, rng), (4, 4242))
        w2.grad = [-0.75, 0.4, 0.0]
        b2.grad = -0.9
        opt2.step()
        w.grad = [-0.75, 0.4, 0.0]
        b.grad = -0.9
        opt.step()
        self.assertEqual(
            ag.dump_training_state(parameters, opt, 5, 4242),
            ag.dump_training_state(p2, opt2, 5, 4242),
        )


class LoadValidationTests(unittest.TestCase):
    def setUp(self):
        self.parameters, self.opt = make_amsgrad()
        self.text = ag.dump_training_state(self.parameters, self.opt, 1, 2)

    def test_rejects_non_str(self):
        for bad in (b"x", None, 42, 1.0, [], {}):
            with self.assertRaises(TypeError):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_other_optimizer(self):
        w = Tensor(1.0, requires_grad=True)
        with self.assertRaises(TypeError):
            ag.load_training_state({"w": w}, ag.SGD([w], 0.1), self.text)

    def test_cross_version_mismatch(self):
        rp, ropt = make_rmsprop()
        v1 = ag.dump_training_state(rp, ropt, 0, 0)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, v1)
        with self.assertRaises(ValueError):
            ag.load_training_state(rp, ropt, self.text)
        # v5 text into an Adam target is a ValueError, not a TypeError.
        w = self.opt.parameters[0]
        with self.assertRaises(ValueError):
            ag.load_training_state(
                {"w": w}, ag.Adam([w], 0.1), self.text
            )

    def test_rejects_wrong_version_number(self):
        for bad_number in ("6", "0", "05", "5.0"):
            bad = self.text.replace('"version":5',
                                    '"version":' + bad_number, 1)
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_missing_or_tampered_digest(self):
        # Dropping the digest member leaves a bare four-key document.
        bare = four_key_doc(self.text)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bare)
        # Flip one hex digit of the digest.
        parsed = json.loads(self.text)
        flipped = "0" if parsed["digest"][0] != "0" else "1"
        tampered = self.text.replace(
            '"digest":"' + parsed["digest"],
            '"digest":"' + flipped + parsed["digest"][1:],
            1,
        )
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, tampered)
        # Any edit of the signed body invalidates the digest too.
        edited = self.text.replace('"global_step":1', '"global_step":9', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, edited)

    def test_rejects_other_inner_version(self):
        rp, ropt = make_rmsprop()
        frankenstein = resign(ag.dump_checkpoint(rp, ropt))
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, frankenstein)

    def test_rejects_bad_outer_form(self):
        inner = inner_checkpoint(self.text)
        variants = [
            self.text.replace('{"version":5', '{"version" :5'),
            self.text.replace('"global_step"', '"step"'),
            self.text.replace(',"rng_state"', ', "rng_state"'),
            self.text + " ",
            self.text[:-1],
            self.text.replace('{"version":5,', '{ "version":5,'),
            resign(inner).replace(',"checkpoint":', ', "checkpoint":'),
            # Key order / extra key at the outer level.
            '{"version":5,"rng_state":2,"global_step":1,'
            '"checkpoint":' + inner + "}",
        ]
        for bad in variants:
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_integer_lexical_and_range_errors(self):
        inner = inner_checkpoint(self.text)
        prefix = '{"version":5,"global_step":'
        cases = [
            prefix + '-1,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '01,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '1.0,"rng_state":0,"checkpoint":' + inner + "}",
            prefix + '0,"rng_state":4294967296,"checkpoint":' + inner + "}",
            prefix + '0,"rng_state":00,"checkpoint":' + inner + "}",
        ]
        for bad in cases:
            bad = resign_after(bad)
            with self.assertRaises(ValueError, msg=bad):
                ag.load_training_state(self.parameters, self.opt, bad)

    def test_rejects_inner_contract_violations(self):
        # Inner tampering must be rejected: the digest fails as produced,
        # and after re-signing the strict checkpoint/state parser catches
        # it. Either way it is a ValueError with no state applied.
        inner = inner_checkpoint(self.text)
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt,
                resign(inner.replace('"amsgrad"', '"adam"')),
            )
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt,
                resign(inner[:-1]),
            )

    def test_name_order_must_match_target(self):
        # Version 1 AMSGrad checkpoints bind positionally, unlike the
        # version-4 Adamax name permutation: swapped names are rejected.
        inner = inner_checkpoint(self.text)
        swapped = inner.replace('"names":["w","b"]', '"names":["b","w"]')
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, resign(swapped)
            )

    def test_rejects_bad_state_values_after_resign(self):
        inner = inner_checkpoint(self.text)
        state = json.loads(inner)["amsgrad"]

        def wrap(mutated_state):
            bad_inner = (
                '{"version":1,"names":["w","b"],"amsgrad":'
                + dump_six(mutated_state)
                + "}"
            )
            return resign(bad_inner)

        def slot_copy(value):
            return list(value) if isinstance(value, list) else value

        bad_v = dict(state)
        neg_v = [slot_copy(state["v"][0]), slot_copy(state["v"][1])]
        neg_v[0][0] = -0.5
        bad_v["v"] = neg_v
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, wrap(bad_v)
            )
        # v_max below v elementwise (and negative) is rejected.
        bad_max = dict(state)
        lowered = [
            slot_copy(state["v_max"][0]), slot_copy(state["v_max"][1])
        ]
        lowered[0][0] = state["v"][0][0] - 1.0
        bad_max["v_max"] = lowered
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, wrap(bad_max)
            )
        # t must be a non-negative lexical integer.
        bad_t = dict(state)
        bad_t["t"] = -1
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, wrap(bad_t)
            )
        # A count mismatch between the state and parameters is rejected.
        bad_count = dict(state)
        short_m = list(state["m"])
        short_m.pop()
        bad_count["m"] = short_m
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, wrap(bad_count)
            )
        # Sanity: the untouched state, rebuilt in the exact six-decimal
        # form, is accepted.
        ag.load_training_state(
            self.parameters, self.opt, wrap(state)
        )

    def test_failed_load_changes_nothing(self):
        w, b = self.opt.parameters
        before = (
            w.data, b.data, copy.deepcopy(self.opt.m),
            copy.deepcopy(self.opt.v), copy.deepcopy(self.opt.v_max),
            self.opt.t, w.requires_grad, b.requires_grad,
        )
        bad = self.text.replace('"version":5', '"version":1', 1)
        with self.assertRaises(ValueError):
            ag.load_training_state(self.parameters, self.opt, bad)
        self.assertEqual(w.data, before[0])
        self.assertEqual(b.data, before[1])
        self.assertEqual(self.opt.m, before[2])
        self.assertEqual(self.opt.v, before[3])
        self.assertEqual(self.opt.v_max, before[4])
        self.assertEqual(self.opt.t, before[5])
        self.assertEqual(w.requires_grad, before[6])
        self.assertEqual(b.requires_grad, before[7])

    def test_failed_resigned_load_changes_nothing(self):
        w, b = self.opt.parameters
        w.grad = [3.0, 3.0, 3.0]
        b.grad = 3.0
        before = (
            w.data, b.data, copy.deepcopy(self.opt.m),
            copy.deepcopy(self.opt.v), copy.deepcopy(self.opt.v_max),
            self.opt.t,
        )
        inner = inner_checkpoint(self.text)
        state = json.loads(inner)["amsgrad"]
        state["v"][0][0] = -0.5
        bad_inner = (
            '{"version":1,"names":["w","b"],"amsgrad":'
            + dump_six(state)
            + "}"
        )
        with self.assertRaises(ValueError):
            ag.load_training_state(
                self.parameters, self.opt, resign(bad_inner)
            )
        self.assertEqual(w.data, before[0])
        self.assertEqual(b.data, before[1])
        self.assertEqual(self.opt.m, before[2])
        self.assertEqual(self.opt.v, before[3])
        self.assertEqual(self.opt.v_max, before[4])
        self.assertEqual(self.opt.t, before[5])
        # Grad state is also untouched on failure.
        self.assertEqual(w.grad, [3.0, 3.0, 3.0])
        self.assertEqual(b.grad, 3.0)


def resign_after(four_key_text):
    """Re-sign a hand-built four-key v5 document whose bytes changed."""
    digest = hashlib.sha256(four_key_text.encode("utf-8")).hexdigest()
    return four_key_text[:-1] + ',"digest":"' + digest + '"}'


if __name__ == "__main__":
    unittest.main()
