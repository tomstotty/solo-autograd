#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

import json
import math
import sys


def _as_finite_float(value):
    """Validate a scalar number; TypeError on bad type, ValueError if non-finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite number")
    return result


def _validate_data(data):
    """Return data normalized to a float scalar or a non-empty list of floats."""
    if isinstance(data, bool):
        raise TypeError(
            "data must be a finite float scalar or a non-empty 1D float list"
        )
    if isinstance(data, (int, float)):
        result = float(data)
        if not math.isfinite(result):
            raise ValueError("data must be finite")
        return result
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("data list must be non-empty")
        return [_as_finite_float(x) for x in data]
    raise TypeError(
        "data must be a finite float scalar or a non-empty 1D float list"
    )


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


def _accumulate(node, value):
    """Accumulate a gradient into a node that participates in the graph."""
    if not node.requires_grad:
        return
    if node.grad is None:
        node.grad = value
    elif isinstance(value, list):
        node.grad = [x + y for x, y in zip(node.grad, value)]
    else:
        node.grad = node.grad + value


def _validate_grad(grad, like):
    """Validate an externally supplied outgoing gradient against tensor data."""
    if isinstance(grad, bool):
        raise TypeError("grad must be a number or a list of numbers")
    if isinstance(like, float):
        if isinstance(grad, list):
            raise ValueError("grad shape must match tensor shape")
        if not isinstance(grad, (int, float)):
            raise TypeError("grad must be a number for a scalar tensor")
        result = float(grad)
        if not math.isfinite(result):
            raise ValueError("grad must be finite")
        return result
    if isinstance(grad, list):
        if len(grad) != len(like):
            raise ValueError("grad shape must match tensor shape")
        return [_as_finite_float(x) for x in grad]
    if isinstance(grad, (int, float)):
        raise ValueError("grad shape must match tensor shape")
    raise TypeError("grad must be a number or a list of numbers")


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
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            raise TypeError("other must be a Tensor or a finite number")
        if not math.isfinite(float(other)):
            raise ValueError("other must be finite")
        return Tensor(float(other))

    def add(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x + y
        )
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            _accumulate(
                parent_self, _reduce_like(parent_self.data, grad)
            )
            _accumulate(
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
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        a, b = parent_self.data, parent_other.data

        def backward_fn(grad):
            grad_self = _broadcast_apply(grad, b, lambda x, y: x * y)
            grad_other = _broadcast_apply(grad, a, lambda x, y: x * y)
            _accumulate(parent_self, _reduce_like(a, grad_self))
            _accumulate(parent_other, _reduce_like(b, grad_other))

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

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

        # Recompute the output and every intermediate on each pass so that
        # repeated backward() calls accumulate only into graph leaves.
        for node in topo:
            if node._parents:
                node.grad = None
        _accumulate(self, grad)
        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn(node.grad)

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
