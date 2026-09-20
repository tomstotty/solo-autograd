#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

import json
import math
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


def _require_finite(value, message):
    """ValueError if a scalar or list of floats holds a non-finite entry."""
    if isinstance(value, list):
        for item in value:
            if not math.isfinite(item):
                raise ValueError(message)
    elif not math.isfinite(value):
        raise ValueError(message)


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


def _reduce_like(parent_data, value):
    """Sum a broadcast gradient back down to the parent's shape."""
    if not isinstance(parent_data, list) and isinstance(value, list):
        return sum(value)
    return value


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
        _require_finite(out_data, "add produced a non-finite result")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad, accumulate):
            accumulate(
                parent_self, _reduce_like(parent_self.data, grad)
            )
            accumulate(
                parent_other, _reduce_like(parent_other.data, grad)
            )

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def mul(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x * y
        )
        _require_finite(out_data, "mul produced a non-finite result")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        a, b = parent_self.data, parent_other.data

        def backward_fn(grad, accumulate):
            grad_self = _broadcast_apply(grad, b, lambda x, y: x * y)
            grad_other = _broadcast_apply(grad, a, lambda x, y: x * y)
            accumulate(parent_self, _reduce_like(a, grad_self))
            accumulate(parent_other, _reduce_like(b, grad_other))

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def tanh(self):
        if isinstance(self.data, list):
            out_data = [math.tanh(x) for x in self.data]
        else:
            out_data = math.tanh(self.data)
        _require_finite(out_data, "tanh produced a non-finite result")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        out = out_data

        def backward_fn(grad, accumulate):
            grad_self = _broadcast_apply(
                grad, out, lambda g, y: g * (1.0 - y * y)
            )
            accumulate(parent, grad_self)

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def sum(self):
        if isinstance(self.data, list):
            total = 0.0
            for x in self.data:
                total += x
            out_data = total
        else:
            out_data = self.data
        _require_finite(out_data, "sum produced a non-finite result")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad, accumulate):
            if isinstance(parent.data, list):
                accumulate(parent, [grad] * len(parent.data))
            else:
                accumulate(parent, grad)

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

        # Phase 1: accumulate every gradient into a local store, validating
        # finiteness as we go. A non-finite contribution aborts the pass
        # before any node's grad is touched.
        local = {}

        def accumulate(node, value):
            if not node.requires_grad:
                return
            _require_finite(value, "gradient must be finite")
            key = id(node)
            if key in local:
                base = local[key]
            elif not node._parents:
                base = node.grad
            else:
                base = None
            if base is None:
                merged = value
            elif isinstance(value, list):
                merged = [x + y for x, y in zip(base, value)]
            else:
                merged = base + value
            _require_finite(merged, "gradient must be finite")
            local[key] = merged

        accumulate(self, grad)
        for node in reversed(topo):
            node_grad = local.get(id(node))
            if node._backward_fn is not None and node_grad is not None:
                node._backward_fn(node_grad, accumulate)

        # Phase 2: commit. Intermediates are recomputed each pass so that
        # repeated backward() calls accumulate only into graph leaves.
        for node in topo:
            key = id(node)
            if key in local:
                node.grad = local[key]
            elif node._parents:
                node.grad = None

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


def _validate_tolerance(value, name):
    """Validate an eps/atol style parameter; return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name + " must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def gradcheck(fn, data, eps=1e-6, atol=1e-5):
    """Compare analytic gradients from backward() with central differences.

    Returns (passed, max_abs_error) as (bool, float).
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    eps = _validate_tolerance(eps, "eps")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    atol = _validate_tolerance(atol, "atol")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")

    def evaluate(values):
        result = fn(Tensor(values, True))
        if not isinstance(result, Tensor):
            raise TypeError("fn must return a Tensor")
        return result

    # Analytic pass: one backward() through the graph built by fn.
    tensor = Tensor(data, True)
    result = fn(tensor)
    if not isinstance(result, Tensor):
        raise TypeError("fn must return a Tensor")
    if isinstance(result.data, list):
        raise ValueError("fn must return a scalar Tensor")
    if not result._parents:
        raise ValueError("fn result must be part of a graph")
    result.backward()
    if tensor.grad is None:
        raise ValueError("fn result does not depend on the input")
    analytic = tensor.grad if isinstance(tensor.grad, list) else [tensor.grad]

    normalized = tensor.data
    scalar_input = not isinstance(normalized, list)
    base = [normalized] if scalar_input else list(normalized)

    # Numeric pass: central difference per input index on fresh tensors.
    max_error = 0.0
    passed = True
    for i in range(len(base)):
        plus = list(base)
        plus[i] = plus[i] + eps
        minus = list(base)
        minus[i] = minus[i] - eps
        plus_result = evaluate(plus[0] if scalar_input else plus)
        minus_result = evaluate(minus[0] if scalar_input else minus)
        if isinstance(plus_result.data, list) or isinstance(
            minus_result.data, list
        ):
            raise ValueError("fn must return a scalar Tensor")
        numeric = (plus_result.data - minus_result.data) / (2.0 * eps)
        error = abs(analytic[i] - numeric)
        if error > max_error:
            max_error = error
        if error > atol:
            passed = False
    return (passed, float(max_error))


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
