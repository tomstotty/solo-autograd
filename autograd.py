#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

import json
import math
import os
import re
import sys
import tempfile


def _as_finite_float(value):
    """Validate a scalar float; TypeError on bad type, ValueError if non-finite."""
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError("expected a finite float")
    if not math.isfinite(value):
        raise ValueError("expected a finite float")
    return value


def _validate_data(data):
    """Return data normalized to a float scalar or a non-empty list of floats."""
    if isinstance(data, bool):
        raise TypeError(
            "data must be a finite float scalar or a non-empty 1D float list"
        )
    if isinstance(data, float):
        if not math.isfinite(data):
            raise ValueError("data must be finite")
        return data
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("data list must be non-empty")
        return [_as_finite_float(x) for x in data]
    raise TypeError(
        "data must be a finite float scalar or a non-empty 1D float list"
    )


def _ensure_finite_data(data):
    """ValueError if a forward result contains a non-finite value."""
    values = data if isinstance(data, list) else [data]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("forward result must be finite")


def _ensure_finite_grad(value):
    """ValueError if a gradient to accumulate contains a non-finite value."""
    values = value if isinstance(value, list) else [value]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("gradient must be finite")


def _broadcast_apply(a, b, op):
    """Apply op over scalar/vector operands with scalar broadcasting."""
    a_vector = isinstance(a, list)
    b_vector = isinstance(b, list)
    if not a_vector and not b_vector:
        return op(a, b)
    if a_vector and not b_vector:
        return [op(x, b) for x in a]
    if not a_vector and b_vector:
        return [op(a, y) for y in b]
    if len(a) != len(b):
        raise ValueError("vector lengths must match")
    return [op(x, y) for x, y in zip(a, b)]


def _map_unary(data, op):
    """Apply op elementwise over a scalar or vector."""
    if isinstance(data, list):
        return [op(x) for x in data]
    return op(data)


def _reduce_like(parent_data, value):
    """Sum a broadcast gradient back down to the parent's shape."""
    if not isinstance(parent_data, list) and isinstance(value, list):
        return sum(value)
    return value


def _merge_grad(a, b):
    """Combine two gradient contributions of identical shape."""
    if isinstance(a, list):
        return [x + y for x, y in zip(a, b)]
    return a + b


def _validate_grad(grad, like):
    """Validate an externally supplied outgoing gradient against tensor data."""
    if isinstance(grad, bool):
        raise TypeError("grad must be a float or a list of floats")
    if isinstance(like, float):
        if isinstance(grad, list):
            raise ValueError("grad shape must match tensor shape")
        if not isinstance(grad, float):
            raise TypeError("grad must be a float for a scalar tensor")
        if not math.isfinite(grad):
            raise ValueError("grad must be finite")
        return grad
    if isinstance(grad, list):
        if len(grad) != len(like):
            raise ValueError("grad shape must match tensor shape")
        return [_as_finite_float(x) for x in grad]
    if isinstance(grad, float):
        raise ValueError("grad shape must match tensor shape")
    raise TypeError("grad must be a float or a list of floats")


class Tensor:
    def __init__(self, data, requires_grad=False):
        normalized = _validate_data(data)
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        self.data = normalized
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward_fn = None

    @classmethod
    def _make(cls, data, requires_grad, parents, backward_fn):
        obj = cls.__new__(cls)
        obj.data = data
        obj.requires_grad = requires_grad
        obj.grad = None
        obj._parents = parents
        obj._backward_fn = backward_fn
        return obj

    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, bool) or not isinstance(other, float):
            raise TypeError("other must be a Tensor or a finite float")
        if not math.isfinite(other):
            raise ValueError("other must be finite")
        return Tensor(other)

    def add(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x + y
        )
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            return [
                (parent_self, _reduce_like(parent_self.data, grad)),
                (parent_other, _reduce_like(parent_other.data, grad)),
            ]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def mul(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x * y
        )
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        a, b = parent_self.data, parent_other.data

        def backward_fn(grad):
            grad_self = _broadcast_apply(grad, b, lambda x, y: x * y)
            grad_other = _broadcast_apply(grad, a, lambda x, y: x * y)
            return [
                (parent_self, _reduce_like(a, grad_self)),
                (parent_other, _reduce_like(b, grad_other)),
            ]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def tanh(self):
        out_data = _map_unary(self.data, math.tanh)
        _ensure_finite_data(out_data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        out = out_data

        def backward_fn(grad):
            contribution = _broadcast_apply(
                grad, out, lambda g, y: g * (1.0 - y * y)
            )
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def sum(self):
        if isinstance(self.data, list):
            out_data = sum(self.data)
        else:
            out_data = self.data
        _ensure_finite_data(out_data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            if isinstance(parent.data, list):
                return [(parent, [grad] * len(parent.data))]
            return [(parent, grad)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def zero_grad(self):
        self.grad = None
        return None

    def backward(self, grad=None):
        if not self._parents:
            raise ValueError("cannot call backward on a tensor without a graph")
        if grad is None:
            if isinstance(self.data, list):
                raise ValueError(
                    "grad must be provided for a non-scalar tensor"
                )
            grad = 1.0
        else:
            grad = _validate_grad(grad, self.data)

        topo = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)

        # Phase 1: compute every gradient without mutating any node, so a
        # non-finite gradient aborts the pass with all grads untouched.
        grads = {id(self): grad}
        for node in reversed(topo):
            if node._backward_fn is None:
                continue
            node_grad = grads.get(id(node))
            if node_grad is None:
                continue
            for parent, contribution in node._backward_fn(node_grad):
                if not parent.requires_grad:
                    continue
                _ensure_finite_grad(contribution)
                existing = grads.get(id(parent))
                if existing is None:
                    grads[id(parent)] = contribution
                else:
                    merged = _merge_grad(existing, contribution)
                    _ensure_finite_grad(merged)
                    grads[id(parent)] = merged

        # Phase 2: resolve the final grad for every node, validate, then
        # apply. Non-leaf grads are recomputed; leaf grads accumulate.
        assignments = []
        for node in topo:
            if node._parents:
                assignments.append((node, grads.get(id(node))))
            elif node.requires_grad:
                value = grads.get(id(node))
                if value is not None:
                    if node.grad is not None:
                        value = _merge_grad(node.grad, value)
                    assignments.append((node, value))
        for node, value in assignments:
            if value is not None:
                _ensure_finite_grad(value)
        for node, value in assignments:
            node.grad = value

    def to_json(self):
        return (
            '{"data":'
            + _json_value(self.data)
            + ',"grad":'
            + ("null" if self.grad is None else _json_value(self.grad))
            + ',"requires_grad":'
            + ("true" if self.requires_grad else "false")
            + "}"
        )


class SGD:
    """Stochastic gradient descent over a fixed list of Tensors."""

    def __init__(self, parameters, lr):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        # Compute and validate every new value before mutating anything, so
        # a failure leaves all data, grad, and requires_grad untouched.
        updates = []
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            updates.append(
                (parameter, self._updated_data(parameter.data, parameter.grad))
            )
        for parameter, new_data in updates:
            parameter.data = new_data
        return None

    def _updated_data(self, data, grad):
        if isinstance(data, list):
            if isinstance(grad, list):
                if len(grad) != len(data):
                    raise ValueError("grad shape must match tensor shape")
                return [
                    self._updated_value(value, g)
                    for value, g in zip(data, grad)
                ]
            if isinstance(grad, float) and not isinstance(grad, bool):
                raise ValueError("grad shape must match tensor shape")
            raise TypeError("grad must be a list of floats")
        if isinstance(grad, list):
            raise ValueError("grad shape must match tensor shape")
        return self._updated_value(data, grad)

    def _updated_value(self, value, grad):
        if isinstance(grad, bool) or not isinstance(grad, float):
            raise TypeError("grad elements must be floats")
        if not math.isfinite(grad):
            raise ValueError("grad must be finite")
        new_value = value - self.lr * grad
        if not math.isfinite(new_value):
            raise ValueError("updated data must be finite")
        return new_value

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


def gradcheck(fn, data, eps=1e-6, atol=1e-5):
    """Compare analytic gradients from backward() with central differences.

    Returns (passed, max_abs_error) where passed is True only when every
    per-element absolute error is at most atol.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if isinstance(eps, bool) or not isinstance(eps, float):
        raise TypeError("eps must be a finite float")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    if isinstance(atol, bool) or not isinstance(atol, float):
        raise TypeError("atol must be a finite float")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a non-negative finite float")
    tensor = Tensor(data, True)

    def scalar_data(result):
        if not isinstance(result, Tensor):
            raise TypeError("fn must return a Tensor")
        if isinstance(result.data, list):
            raise ValueError("fn must return a scalar Tensor")
        return result.data

    result = fn(tensor)
    scalar_data(result)
    if not result._parents:
        raise ValueError("fn result must be part of a graph")
    result.backward()
    if tensor.grad is None:
        raise ValueError("fn result is not connected to the input")

    vector = isinstance(tensor.data, list)
    base = tensor.data if vector else [tensor.data]
    analytic = tensor.grad if vector else [tensor.grad]

    max_abs_error = 0.0
    passed = True
    for index in range(len(base)):
        plus = list(base)
        minus = list(base)
        plus[index] = base[index] + eps
        minus[index] = base[index] - eps
        f_plus = scalar_data(fn(Tensor(plus if vector else plus[0])))
        f_minus = scalar_data(fn(Tensor(minus if vector else minus[0])))
        numeric = (f_plus - f_minus) / (2.0 * eps)
        error = abs(analytic[index] - numeric)
        if error > max_abs_error:
            max_abs_error = error
        if error > atol:
            passed = False
    return (passed, max_abs_error)


_STATE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STATE_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")


def _check_state_parameters(parameters):
    """Validate a dump_state/load_state parameters mapping."""
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dict of name -> Tensor")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    seen = set()
    for name, tensor in parameters.items():
        if not isinstance(name, str):
            raise TypeError("parameter names must be strings")
        if _STATE_NAME.fullmatch(name) is None:
            raise ValueError("invalid parameter name")
        if not isinstance(tensor, Tensor):
            raise TypeError("parameter values must be Tensors")
        if id(tensor) in seen:
            raise ValueError("parameters must not contain duplicate Tensors")
        seen.add(id(tensor))
        _validate_data(tensor.data)
        if not isinstance(tensor.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")


def dump_state(parameters):
    """Serialize named Tensors to a compact JSON state string.

    The output contains no whitespace and no trailing newline; floats are
    written with exactly six decimals and negative zero as 0.000000.
    """
    _check_state_parameters(parameters)
    entries = []
    for name, tensor in parameters.items():
        entries.append(
            '{"name":"'
            + name
            + '","data":'
            + _json_value(tensor.data)
            + ',"requires_grad":'
            + ("true" if tensor.requires_grad else "false")
            + "}"
        )
    return '{"parameters":[' + ",".join(entries) + "]}"


class _StateParser:
    """Strict parser for the exact textual form produced by dump_state."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_state format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_entry(self):
        self._expect('{"name":"')
        match = _STATE_NAME.match(self._text, self._pos)
        if match is None:
            self._fail()
        name = match.group(0)
        self._pos = match.end()
        self._expect('","data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (name, data, requires_grad)

    def parse(self):
        self._expect('{"parameters":[')
        entries = [self._parse_entry()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            entries.append(self._parse_entry())
        self._expect("]}")
        if self._pos != len(self._text):
            self._fail()
        return entries


def load_state(parameters, text):
    """Restore named Tensors from a string produced by dump_state.

    On full validation success, overwrite each Tensor's data and
    requires_grad and clear its grad; on any failure no Tensor is touched.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_state_parameters(parameters)
    entries = _StateParser(text).parse()
    if len(entries) != len(parameters):
        raise ValueError("state must contain exactly one entry per parameter")
    updates = []
    for (name, data, requires_grad), (expected_name, tensor) in zip(
        entries, parameters.items()
    ):
        if name != expected_name:
            raise ValueError("state names must match parameters in order")
        current = tensor.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match tensor shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match tensor shape")
        updates.append((tensor, data, requires_grad))
    for tensor, data, requires_grad in updates:
        tensor.data = data
        tensor.requires_grad = requires_grad
        tensor.grad = None
    return None


def _format_float(value):
    if not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite float")
    text = f"{value:.6f}"
    if text == "-0.000000":
        text = "0.000000"
    return text


def _json_value(value):
    if isinstance(value, list):
        return "[" + ",".join(_format_float(float(x)) for x in value) + "]"
    return _format_float(float(value))


def _cli_tensor():
    line = sys.stdin.readline()
    try:
        payload = json.loads(line)
    except (ValueError, OSError):
        print("invalid JSON input", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("input must be a JSON object", file=sys.stderr)
        return 2
    if set(payload) - {"data", "requires_grad"}:
        print("only data and requires_grad are allowed", file=sys.stderr)
        return 2
    if "data" not in payload:
        print("data is required", file=sys.stderr)
        return 2
    requires_grad = payload.get("requires_grad", False)
    if not isinstance(requires_grad, bool):
        print("requires_grad must be a bool", file=sys.stderr)
        return 2
    try:
        tensor = Tensor(payload["data"], requires_grad)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(tensor.to_json() + "\n")
    return 0


class _ObjectPairs(list):
    """Marker for a JSON object parsed as ordered (key, value) pairs."""


def _parse_json_pairs(text):
    """Parse JSON preserving object member pairs so duplicates are visible."""
    return json.loads(text, object_pairs_hook=_ObjectPairs)


def _has_duplicate_key(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            return True
        seen.add(key)
    return False


def _read_stdin_line():
    """Read stdin, requiring exactly one line; returns (text, error)."""
    try:
        text = sys.stdin.read()
    except OSError:
        return None, "failed to read stdin"
    if text.endswith("\n"):
        text = text[:-1]
    if "\n" in text:
        return None, "stdin must contain exactly one line of JSON"
    return text, None


def _cli_checkpoint_parameters():
    """Parse the checkpoint stdin payload into an ordered name -> Tensor dict.

    Returns (parameters, error); exactly one of the two is None.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object with only the parameters key"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    if [key for key, _ in payload] != ["parameters"]:
        return None, "input must be an object with only the parameters key"
    members = payload[0][1]
    if not isinstance(members, _ObjectPairs):
        return None, "parameters must be an object"
    if _has_duplicate_key(members):
        return None, "duplicate key in JSON input"
    if len(members) == 0:
        return None, "parameters must be non-empty"
    parameters = {}
    for name, spec in members:
        if not isinstance(spec, _ObjectPairs):
            return (
                None,
                "each parameter must contain only data and requires_grad"
                " in that order",
            )
        if _has_duplicate_key(spec):
            return None, "duplicate key in JSON input"
        if [key for key, _ in spec] != ["data", "requires_grad"]:
            return (
                None,
                "each parameter must contain only data and requires_grad"
                " in that order",
            )
        try:
            parameters[name] = Tensor(spec[0][1], spec[1][1])
        except (TypeError, ValueError) as exc:
            return None, str(exc)
    return parameters, None


def _cli_checkpoint_save(path):
    parameters, error = _cli_checkpoint_parameters()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        content = dump_state(parameters)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


def _cli_checkpoint_load(path):
    try:
        with open(path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parameters, error = _cli_checkpoint_parameters()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        load_state(parameters, text)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(dump_state(parameters) + "\n")
    return 0


def _cli_evaluate_samples():
    """Parse the evaluate stdin payload into a list of (x, y) float pairs.

    Returns (samples, error); exactly one of the two is None.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object with only the samples key"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    if [key for key, _ in payload] != ["samples"]:
        return None, "input must be an object with only the samples key"
    members = payload[0][1]
    if not isinstance(members, list) or isinstance(members, _ObjectPairs):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(members) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in members:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    return samples, None


def _cli_evaluate(path):
    try:
        with open(path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    samples, error = _cli_evaluate_samples()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        entries = _StateParser(text).parse()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len(entries) != 2 or entries[0][0] != "w" or entries[1][0] != "b":
        print("state must contain exactly the parameters w and b in order",
              file=sys.stderr)
        return 2
    if isinstance(entries[0][1], list) or isinstance(entries[1][1], list):
        print("parameters w and b must be scalar Tensors", file=sys.stderr)
        return 2
    try:
        w_tensor = Tensor(entries[0][1], entries[0][2])
        b_tensor = Tensor(entries[1][1], entries[1][2])
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    w = w_tensor.data
    b = b_tensor.data
    total = 0.0
    for x, y in samples:
        prediction = w * x
        if not math.isfinite(prediction):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        prediction = prediction + b
        if not math.isfinite(prediction):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        residual = prediction - y
        if not math.isfinite(residual):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        squared = residual * residual
        if not math.isfinite(squared):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        total = total + squared
        if not math.isfinite(total):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
    loss = total / len(samples)
    if not math.isfinite(loss):
        print("evaluation produced a non-finite value", file=sys.stderr)
        return 2
    sys.stdout.write('{"loss":' + _format_float(loss) + "}\n")
    return 0


def _atomic_write_checkpoint(path, content):
    """Atomically replace path with UTF-8 content; None on success, error text."""
    directory = os.path.dirname(os.path.abspath(path))
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
        with os.fdopen(fd, "wb") as tmp_file:
            fd = None  # the file object owns the descriptor now
            tmp_file.write(content.encode("utf-8"))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        return str(exc)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


def _train_forward(w, b, samples):
    """Mean-squared-error scalar Tensor: sum((w*x+b-y)^2) / n, sample order."""
    total = None
    for x, y in samples:
        residual = w.mul(x).add(b).add(-y)
        squared = residual.mul(residual)
        total = squared if total is None else total.add(squared)
    return total.mul(1.0 / len(samples))


def _cli_train_config(include_momentum=False):
    """Parse the train/momentum stdin payload.

    Returns (config, error); exactly one is None. config is
    (samples, steps, lr, seed) for train, or
    (samples, steps, lr, momentum, seed) when include_momentum is True.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    expected_keys = ["samples", "steps", "lr"]
    if include_momentum:
        expected_keys.append("momentum")
    expected_keys.append("seed")
    if [key for key, _ in payload] != expected_keys:
        if include_momentum:
            return (
                None,
                "input must contain only samples, steps, lr, momentum"
                " and seed in that order",
            )
        return (
            None,
            "input must contain only samples, steps, lr and seed in that order",
        )
    samples_raw = payload[0][1]
    steps = payload[1][1]
    lr = payload[2][1]
    if include_momentum:
        momentum_value = payload[3][1]
        seed = payload[4][1]
    else:
        seed = payload[3][1]
    if not isinstance(samples_raw, list) or isinstance(
        samples_raw, _ObjectPairs
    ):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(samples_raw) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in samples_raw:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    if isinstance(steps, bool) or not isinstance(steps, int):
        return None, "steps must be a positive integer"
    if steps <= 0:
        return None, "steps must be a positive integer"
    if isinstance(lr, bool) or not isinstance(lr, float):
        return None, "lr must be a positive finite float"
    if not math.isfinite(lr) or lr <= 0.0:
        return None, "lr must be a positive finite float"
    if include_momentum:
        momentum_error = (
            "momentum must be a finite float between 0.0 (inclusive)"
            " and 1.0 (exclusive)"
        )
        if isinstance(momentum_value, bool) or not isinstance(
            momentum_value, float
        ):
            return None, momentum_error
        if (
            not math.isfinite(momentum_value)
            or momentum_value < 0.0
            or momentum_value >= 1.0
        ):
            return None, momentum_error
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, "seed must be an integer between 0 and 4294967295"
    if seed < 0 or seed > 4294967295:
        return None, "seed must be an integer between 0 and 4294967295"
    if include_momentum:
        return (samples, steps, lr, momentum_value, seed), None
    return (samples, steps, lr, seed), None


def _read_state_file(input_path):
    """Read a checkpoint path as strict UTF-8.

    Returns (text, exit_code, error); on success exit_code is 0 and the
    other two are None-ish. Missing/unreadable files exit 1 like other I/O
    failures; non-UTF-8 state is a usage error and exits 2.
    """
    try:
        with open(input_path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        return None, 1, str(exc)
    try:
        return raw.decode("utf-8"), 0, None
    except UnicodeDecodeError as exc:
        return None, 2, str(exc)


def _cli_train(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_train_config()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if (
            len(entries) != 2
            or entries[0][0] != "w"
            or entries[1][0] != "b"
            or isinstance(entries[0][1], list)
            or isinstance(entries[1][1], list)
            or entries[0][2] is not True
            or entries[1][2] is not True
        ):
            print(
                "state must contain exactly the scalar parameters w and b"
                " with requires_grad true in order",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
    try:
        optimizer = SGD([w, b], lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            optimizer.step()
            # Quantize after every step so a reloaded checkpoint resumes the
            # exact same trajectory as uninterrupted training.
            w.data = float(_format_float(w.data))
            b.data = float(_format_float(b.data))
        final_loss = _train_forward(w, b, samples).data
        content = dump_state({"w": w, "b": b})
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def _cli_momentum(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_train_config(include_momentum=True)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, momentum, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if len(entries) != 4 or [entry[0] for entry in entries] != [
            "w",
            "b",
            "vw",
            "vb",
        ] or any(isinstance(entry[1], list) for entry in entries):
            print(
                "state must contain exactly the scalar parameters w, b,"
                " vw and vb in order",
                file=sys.stderr,
            )
            return 2
        if (
            entries[0][2] is not True
            or entries[1][2] is not True
            or entries[2][2] is not False
            or entries[3][2] is not False
        ):
            print(
                "w and b must have requires_grad true; vw and vb must have"
                " requires_grad false",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
            vw = entries[2][1]
            vb = entries[3][1]
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
        vw = 0.0
        vb = 0.0
    try:
        for _ in range(steps):
            w.zero_grad()
            b.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            # Compute the velocity buffers and new parameters before
            # committing anything, so a non-finite result aborts the step
            # with all four values untouched.
            new_vw = momentum * vw + w.grad
            new_vb = momentum * vb + b.grad
            if not math.isfinite(new_vw) or not math.isfinite(new_vb):
                raise ValueError("momentum buffer must be finite")
            new_w = w.data - lr * new_vw
            new_b = b.data - lr * new_vb
            if not math.isfinite(new_w) or not math.isfinite(new_b):
                raise ValueError("updated data must be finite")
            # Quantize after every step so a reloaded checkpoint resumes the
            # exact same trajectory as uninterrupted training.
            vw = float(_format_float(new_vw))
            vb = float(_format_float(new_vb))
            w.data = float(_format_float(new_w))
            b.data = float(_format_float(new_b))
        final_loss = _train_forward(w, b, samples).data
        vw_tensor = Tensor(vw, False)
        vb_tensor = Tensor(vb, False)
        content = dump_state(
            {"w": w, "b": b, "vw": vw_tensor, "vb": vb_tensor}
        )
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def _cli_adam_config():
    """Parse the adam stdin payload.

    Returns (config, error); exactly one is None. config is
    (samples, steps, lr, beta1, beta2, epsilon, seed) with samples a list
    of (x, y) float pairs.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    expected_keys = [
        "samples",
        "steps",
        "lr",
        "beta1",
        "beta2",
        "epsilon",
        "seed",
    ]
    if [key for key, _ in payload] != expected_keys:
        return (
            None,
            "input must contain only samples, steps, lr, beta1, beta2,"
            " epsilon and seed in that order",
        )
    samples_raw = payload[0][1]
    steps = payload[1][1]
    lr = payload[2][1]
    beta1 = payload[3][1]
    beta2 = payload[4][1]
    epsilon_value = payload[5][1]
    seed = payload[6][1]
    if not isinstance(samples_raw, list) or isinstance(
        samples_raw, _ObjectPairs
    ):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(samples_raw) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in samples_raw:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    if isinstance(steps, bool) or not isinstance(steps, int):
        return None, "steps must be a positive integer"
    if steps <= 0:
        return None, "steps must be a positive integer"
    if isinstance(lr, bool) or not isinstance(lr, float):
        return None, "lr must be a positive finite float"
    if not math.isfinite(lr) or lr <= 0.0:
        return None, "lr must be a positive finite float"
    for name, value in (("beta1", beta1), ("beta2", beta2)):
        beta_error = (
            name
            + " must be a finite float between 0.0 (inclusive) and 1.0"
            " (exclusive)"
        )
        if isinstance(value, bool) or not isinstance(value, float):
            return None, beta_error
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            return None, beta_error
    if isinstance(epsilon_value, bool) or not isinstance(epsilon_value, float):
        return None, "epsilon must be a positive finite float"
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        return None, "epsilon must be a positive finite float"
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, "seed must be an integer between 0 and 4294967295"
    if seed < 0 or seed > 4294967295:
        return None, "seed must be an integer between 0 and 4294967295"
    return (samples, steps, lr, beta1, beta2, epsilon_value, seed), None


def _cli_adam(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_adam_config()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, beta1, beta2, epsilon, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        names = ["w", "b", "mw", "mb", "vw", "vb", "t"]
        requires_flags = (True, True, False, False, False, False, False)
        if (
            len(entries) != 7
            or [entry[0] for entry in entries] != names
            or any(isinstance(entry[1], list) for entry in entries)
            or any(
                entry[2] is not flag
                for entry, flag in zip(entries, requires_flags)
            )
        ):
            print(
                "state must contain exactly the scalar parameters w, b, mw,"
                " mb, vw, vb and t in order with requires_grad true, true,"
                " false, false, false, false, false",
                file=sys.stderr,
            )
            return 2
        t_value = entries[6][1]
        if (
            not isinstance(t_value, float)
            or not math.isfinite(t_value)
            or t_value < 0.0
            or not t_value.is_integer()
        ):
            print(
                "t must be a non-negative integer-valued float",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        mw = entries[2][1]
        mb = entries[3][1]
        vw = entries[4][1]
        vb = entries[5][1]
        t = t_value
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
        mw = 0.0
        mb = 0.0
        vw = 0.0
        vb = 0.0
        t = 0.0
    try:
        one_minus_b1 = 1.0 - beta1
        one_minus_b2 = 1.0 - beta2
        for _ in range(steps):
            w.zero_grad()
            b.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            # Advance t and derive the bias-correction denominators before
            # touching any parameter or buffer.
            new_t = t + 1.0
            if not math.isfinite(new_t):
                raise ValueError("t must remain finite")
            b1_power = beta1**new_t
            b2_power = beta2**new_t
            if not math.isfinite(b1_power) or not math.isfinite(b2_power):
                raise ValueError("beta power must be finite")
            bias1 = 1.0 - b1_power
            bias2 = 1.0 - b2_power
            if not math.isfinite(bias1) or not math.isfinite(bias2):
                raise ValueError("bias correction must be finite")
            # Compute every new moment and parameter up front, so a
            # non-finite result aborts the step with all seven values
            # untouched.
            updates = []
            for data, grad, m, v in (
                (w.data, w.grad, mw, vw),
                (b.data, b.grad, mb, vb),
            ):
                new_m = beta1 * m + one_minus_b1 * grad
                new_v = beta2 * v + one_minus_b2 * (grad * grad)
                if not math.isfinite(new_m) or not math.isfinite(new_v):
                    raise ValueError("adam buffers must be finite")
                m_hat = new_m / bias1
                v_hat = new_v / bias2
                if not math.isfinite(m_hat) or not math.isfinite(v_hat):
                    raise ValueError(
                        "bias-corrected moments must be finite"
                    )
                denom = math.sqrt(v_hat) + epsilon
                if not math.isfinite(denom):
                    raise ValueError("update denominator must be finite")
                new_data = data - lr * m_hat / denom
                if not math.isfinite(new_data):
                    raise ValueError("updated data must be finite")
                updates.append((new_m, new_v, new_data))
            # Quantize all seven values, then commit, so a reloaded
            # checkpoint resumes the exact same trajectory as uninterrupted
            # training.
            mw = float(_format_float(updates[0][0]))
            vw = float(_format_float(updates[0][1]))
            w_new = float(_format_float(updates[0][2]))
            mb = float(_format_float(updates[1][0]))
            vb = float(_format_float(updates[1][1]))
            b_new = float(_format_float(updates[1][2]))
            t = float(_format_float(new_t))
            w.data = w_new
            b.data = b_new
        final_loss = _train_forward(w, b, samples).data
        content = dump_state(
            {
                "w": w,
                "b": b,
                "mw": Tensor(mw, False),
                "mb": Tensor(mb, False),
                "vw": Tensor(vw, False),
                "vb": Tensor(vb, False),
                "t": Tensor(t, False),
            }
        )
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def main(argv):
    if len(argv) == 1 and argv[0] == "tensor":
        try:
            return _cli_tensor()
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if (
        len(argv) == 3
        and argv[0] == "checkpoint"
        and argv[1] in ("save", "load")
    ):
        try:
            if argv[1] == "save":
                return _cli_checkpoint_save(argv[2])
            return _cli_checkpoint_load(argv[2])
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) == 2 and argv[0] == "evaluate":
        try:
            return _cli_evaluate(argv[1])
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "train":
        try:
            return _cli_train(argv[1], argv[2] if len(argv) == 3 else None)
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "momentum":
        try:
            return _cli_momentum(
                argv[1], argv[2] if len(argv) == 3 else None
            )
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "adam":
        try:
            return _cli_adam(
                argv[1], argv[2] if len(argv) == 3 else None
            )
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    print(
        "usage: python autograd.py tensor | checkpoint {save|load} PATH"
        " | evaluate PATH | train OUTPUT [INPUT]"
        " | momentum OUTPUT [INPUT] | adam OUTPUT [INPUT]",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
