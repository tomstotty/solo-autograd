"""Contract tests for jacobiancheck's analytic-gradient type errors.

Standard library only; discovered by ``python -m unittest discover``.
"""

import unittest

from autograd import Tensor, jacobiancheck


class MissingAnalyticGradTest(unittest.TestCase):

    def test_disconnected_input_raises_type_error(self):
        # The result is part of a graph, but the graph is rooted at a
        # different leaf, so the probed input receives no grad (None).
        other = Tensor([0.5, -0.5], True)

        def fn(tensor):
            return other.add(other)

        with self.assertRaises(TypeError):
            jacobiancheck(fn, [0.1, 0.2])

    def test_scalar_disconnected_input_raises_type_error(self):
        other = Tensor([0.5, -0.5], True)

        def fn(tensor):
            return other.mul(2.0)

        with self.assertRaises(TypeError):
            jacobiancheck(fn, 0.1)

    def test_graphless_result_is_value_error(self):
        # No graph at all (requires_grad False leaf) is still a ValueError.
        def fn(tensor):
            return Tensor([1.0, 2.0])

        with self.assertRaises(ValueError):
            jacobiancheck(fn, [0.1, 0.2])


class PassingCheckTest(unittest.TestCase):

    def test_linear_function_passes(self):
        passed, err = jacobiancheck(
            lambda t: t.add(t),
            [0.1, 0.2, -0.3],
        )
        self.assertTrue(passed, err)

    def test_scalar_input_vector_output_passes(self):
        # Linear graph: output [t, 2t], a graph-backed function of t.
        def graph_fn(t):
            return Tensor([1.0, 2.0]).mul(t)

        passed, err = jacobiancheck(graph_fn, 0.5)
        self.assertTrue(passed, err)


if __name__ == "__main__":
    unittest.main()
