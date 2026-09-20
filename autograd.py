#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

import json
import math
import re
import sys


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


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Exactly the text _format_float can emit: an optional minus, an integer with
# no leading zeroes, a dot and exactly 6 fractional digits.
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")


def _validate_parameters(parameters):
    """Validate a non-empty dict of name -> unique Tensor; return its items."""
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dict of named Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    items = list(parameters.items())
    for name, tensor in items:
        if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
            raise ValueError("parameter names must match [A-Za-z_][A-Za-z0-9_]*")
        if not isinstance(tensor, Tensor):
            raise TypeError("parameter values must be Tensors")
    if len({id(tensor) for _, tensor in items}) != len(items):
        raise ValueError("parameters must not contain duplicate Tensors")
    return items


def dump_state(parameters):
    """Serialize named Tensors as a compact JSON string."""
    items = _validate_parameters(parameters)
    parts = []
    for name, tensor in items:
        _validate_data(tensor.data)
        if not isinstance(tensor.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        parts.append(
            '{"name":"'
            + name
            + '","data":'
            + _json_value(tensor.data)
            + ',"requires_grad":'
            + ("true" if tensor.requires_grad else "false")
            + "}"
        )
    return '{"parameters":[' + ",".join(parts) + "]}"


class _StrictStateParser:
    """Hand-written parser accepting only dump_state's exact JSON dialect."""

    def __init__(self, text):
        self.text = text
        self.pos = 0

    def error(self):
        raise ValueError("state text does not match the expected format")

    def take(self, char):
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            self.error()
        self.pos += 1

    def parse_value(self):
        if self.pos >= len(self.text):
            self.error()
        char = self.text[self.pos]
        if char == "{":
            return self.parse_object()
        if char == "[":
            return self.parse_array()
        if char == '"':
            return self.parse_string()
        if char == "-" or char.isdigit():
            return self.parse_number()
        if self.text.startswith("true", self.pos):
            self.pos += 4
            return True
        if self.text.startswith("false", self.pos):
            self.pos += 5
            return False
        if self.text.startswith("null", self.pos):
            self.pos += 4
            return None
        self.error()

    def parse_number(self):
        match = _NUMBER_RE.match(self.text, self.pos)
        if match is None:
            self.error()
        token = match.group(0)
        # dump_state renders negative zero as "0.000000".
        if token.startswith("-") and float(token) == 0.0:
            self.error()
        end = match.end()
        # Reject any JSON number that is not exactly a 6-decimal float.
        if end < len(self.text) and self.text[end] in "0123456789.eE+-":
            self.error()
        self.pos = end
        return float(token)

    def parse_string(self):
        # dump_state only emits unescaped [A-Za-z_][A-Za-z0-9_]* names, so
        # escapes and control characters are outside the accepted dialect.
        self.pos += 1
        start = self.pos
        while True:
            if self.pos >= len(self.text):
                self.error()
            char = self.text[self.pos]
            if char == '"':
                content = self.text[start:self.pos]
                self.pos += 1
                return content
            if char == "\\" or ord(char) < 0x20:
                self.error()
            self.pos += 1

    def parse_array(self):
        self.take("[")
        result = []
        if self.pos < len(self.text) and self.text[self.pos] == "]":
            self.pos += 1
            return result
        while True:
            result.append(self.parse_value())
            if self.pos >= len(self.text):
                self.error()
            if self.text[self.pos] == "]":
                self.pos += 1
                return result
            self.take(",")

    def parse_object(self):
        self.take("{")
        result = {}
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
            return result
        while True:
            if self.pos >= len(self.text) or self.text[self.pos] != '"':
                self.error()
            key = self.parse_string()
            if key in result:
                self.error()
            self.take(":")
            result[key] = self.parse_value()
            if self.pos >= len(self.text):
                self.error()
            if self.text[self.pos] == "}":
                self.pos += 1
                return result
            self.take(",")

    def parse(self):
        value = self.parse_value()
        if self.pos != len(self.text):
            self.error()
        return value


def _loaded_number(raw):
    """A parsed data element must be a finite float (never bool/null/...)."""
    if not isinstance(raw, float) or isinstance(raw, bool) or not math.isfinite(raw):
        raise ValueError("data elements must be finite floats")
    return raw


def _loaded_data(raw, current):
    """Validate parsed data and match its shape against the current Tensor."""
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError("data list must be non-empty")
        data = [_loaded_number(x) for x in raw]
    else:
        data = _loaded_number(raw)
    if isinstance(current, list):
        if not isinstance(data, list) or len(data) != len(current):
            raise ValueError("data shape must match the current tensor")
    elif isinstance(data, list):
        raise ValueError("data shape must match the current tensor")
    return data


def load_state(parameters, text):
    """Restore Tensors from text produced by dump_state.

    Everything is parsed and validated before any Tensor is touched, so a
    failure leaves all Tensors unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    items = _validate_parameters(parameters)
    parsed = _StrictStateParser(text).parse()

    if not isinstance(parsed, dict) or list(parsed) != ["parameters"]:
        raise ValueError("state text must have a single top-level 'parameters' key")
    raw_parameters = parsed["parameters"]
    if not isinstance(raw_parameters, list) or len(raw_parameters) != len(items):
        raise ValueError("parameter count must match")

    updates = []
    for (name, tensor), entry in zip(items, raw_parameters):
        if not isinstance(entry, dict) or list(entry) != [
            "name",
            "data",
            "requires_grad",
        ]:
            raise ValueError("each parameter must have name, data, requires_grad")
        if entry["name"] != name:
            raise ValueError("parameter names and order must match")
        if not isinstance(entry["requires_grad"], bool):
            raise ValueError("requires_grad must be a bool")
        _validate_data(tensor.data)
        new_data = _loaded_data(entry["data"], tensor.data)
        updates.append((tensor, new_data, entry["requires_grad"]))

    for tensor, new_data, requires_grad in updates:
        tensor.data = new_data
        tensor.requires_grad = requires_grad
        tensor.grad = None
    return None


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


def main(argv):
    if len(argv) == 1 and argv[0] == "tensor":
        try:
            return _cli_tensor()
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    print("usage: python autograd.py tensor", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
