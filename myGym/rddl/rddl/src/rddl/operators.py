"""Logical/scoring operators that compose operands into an expression tree.

Operators provide two parallel channels over their children:

- `decide()` -> bool: whether the (sub)expression is satisfied.
- `evaluate()` -> float: additive reward shaping (children's scores summed).

`gather_variables()` walks the tree and collects `Variable`s from
`LogicalOperand` leaves (predicates) only, giving an expression its full
variable set. `set_symbolic_value()` pushes a desired truth assignment down to
the leaves so the sampler can construct worlds satisfying the expression;
`NotOp` flips the value (De Morgan) on the way down.

Arity bases (`NullaryOperator`/`UnaryOperator`/`BinaryOperator`/`NAryOperator`)
hold the children and supply `__repr__`/`gather_variables`/`set_symbolic_value`
scaffolding; concrete operators add `_SYMBOL`, `__decide__`, and `__evaluate__`.
"""
# from entities import Entity
from typing import Optional

from rddl import Operator
from rddl.core import LogicalOperand, Operand, Variable


class NullaryOperator(Operator):
    """Operator scaffold with no operands (arity 0)."""

    _ARITY = 0

    def __init__(self) -> None:
        """Initialize a nullary operator."""
        super().__init__()

    def __repr__(self) -> str:
        """Render as just the operator symbol."""
        return self._SYMBOL


class UnaryOperator(Operator):
    """Operator scaffold over a single operand (arity 1)."""

    _ARITY = 1

    def __init__(self, operand: Operand) -> None:
        """Store the single operand and register it as an argument."""
        super().__init__()
        self._operand: Operand = operand
        self._append_arguments(operand)

    def __repr__(self) -> str:
        """Render as ``SYMBOL(operand)``."""
        return f"{self._SYMBOL}({self._operand})"

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        """Push the desired truth `value` down to the operand."""
        return self._operand.set_symbolic_value(value, only_if_contains)

    def gather_variables(self) -> list[Variable]:
        """Collect variables from the operand if it is a logical (predicate) leaf."""
        return self._operand.gather_variables() if isinstance(self._operand, LogicalOperand) else []


class BinaryOperator(Operator):
    """Operator scaffold over a left and right operand (arity 2)."""

    _ARITY = 2

    def __init__(self, left: Operand, right: Operand) -> None:
        """Store both operands and register them as arguments."""
        super().__init__()
        self._left: Operand = left
        self._right: Operand = right
        self._append_arguments(left, right)

    def __repr__(self) -> str:
        """Render as ``(left SYMBOL right)``."""
        return f"({self._left} {self._SYMBOL} {self._right})"

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        """Push the desired truth `value` down to both operands."""
        self._left.set_symbolic_value(value, only_if_contains)
        self._right.set_symbolic_value(value, only_if_contains)

    def gather_variables(self) -> list[Variable]:
        """Collect variables from whichever of left/right are logical (predicate) leaves."""
        return (self._left.gather_variables() if isinstance(self._left, LogicalOperand) else []) + (self._right.gather_variables() if isinstance(self._right, LogicalOperand) else [])


class NAryOperator(Operator):
    """Operator scaffold over an arbitrary number of operands (variadic arity)."""

    _ARITY = None

    def __init__(self, operands: list[Operand]) -> None:
        """Store the operand list and register each as an argument."""
        super().__init__()
        self._operands: list[Operand] = operands
        self._append_arguments(*operands)

    def __repr__(self) -> str:
        """Render the operands joined by the operator symbol."""
        return f' {self._SYMBOL} '.join([str(op) for op in self._operands])

    def gather_variables(self) -> list[Variable]:
        """Collect variables from every operand that is a logical (predicate) leaf."""
        gathered: list[Variable] = []
        for operand in self._operands:
            if isinstance(operand, LogicalOperand):
                gathered += operand.gather_variables()
        return gathered

# class SequentialAndOp(BinaryOperator):
#     _SYMBOL = "&>"

#     def __init__(self, left: Predicate, right: Predicate) -> None:
#         super().__init__(left, right)

#     def __evaluate__(self) -> float:
#         return self._left.evaluate() and self._right.evaluate()


class ParallelAndOp(BinaryOperator):
    """Conjunction of two operands: ``decide = L ∧ R``, ``evaluate = L + R``."""

    _SYMBOL = "&"

    def __init__(self, left: Operand, right: Operand) -> None:
        """Construct a parallel-AND of `left` and `right`."""
        super().__init__(left, right)

    def __decide__(self):
        """Satisfied iff both operands decide True."""
        left_check = self._left.decide()
        right_check = self._right.decide()
        # print(f"Checking and operator; left: {left_check}, right: {right_check}, result: {left_check and right_check}")
        result = left_check and right_check
        return result

    def __evaluate__(self):
        """Score is the sum of both operands' scores."""
        left_eval = self._left.evaluate()
        right_eval = self._right.evaluate()
        # print(f"Evaluating and operator; left: {left_eval}, right: {right_eval}, result: {left_eval + right_eval}")
        result = left_eval + right_eval
        return result


class SequentialOp(BinaryOperator):
    """"Do L, then R": ``decide = R`` (after L); ``evaluate = L + (R if L else 0)``."""

    _SYMBOL = "->"

    def __init__(self, left: Operand, right: Operand) -> None:
        """Construct a sequential composition of `left` followed by `right`."""
        super().__init__(left, right)

    def __decide__(self):
        """Satisfied by the second operand's decision (the first is still computed)."""
        first_result = self._left.decide()
        after_result = self._right.decide()
        # print(f"Checking sequential operator; first: {first_result}, after: {after_result}")
        return after_result

    def __evaluate__(self):
        """Score the first operand, then add the second only once the first is satisfied."""
        first_evaluation = self._left.evaluate()
        after_evaluation = self._right.evaluate()
        # print(f"Evaluating sequential operator; first: {first_evaluation}, after: {after_evaluation}")
        return first_evaluation + after_evaluation if self._left.decide() else first_evaluation


class NotOp(UnaryOperator):
    """Negation: ``decide = ¬operand``, ``evaluate = −operand``."""

    _SYMBOL = "~"

    def __init__(self, operand: Operand) -> None:
        """Construct the negation of `operand`."""
        super().__init__(operand)

    def __decide__(self):
        """Satisfied iff the operand is not satisfied."""
        return not self._operand.decide()

    def __evaluate__(self):
        """Negate the operand's score."""
        return -self._operand.evaluate()

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        """Push the negated target value down (De Morgan), so leaves get ``¬value``."""
        self._operand.set_symbolic_value(not value, only_if_contains)


class NAryAndOp(NAryOperator):
    """Variadic conjunction: ``decide = ⋀ ops``, ``evaluate = Σ ops``."""

    _SYMBOL = "&"

    def __init__(self, operands: list[Operand]) -> None:
        """Construct an N-ary AND over `operands`."""
        super().__init__(operands)

    def __decide__(self):
        """Satisfied iff every operand decides True."""
        return all(op.decide() for op in self._operands)

    def __evaluate__(self):
        """Score is the sum over all operands' scores."""
        return sum(op.evaluate() for op in self._operands)

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        """Push the desired truth `value` down to every operand."""
        for op in self._operands:
            op.set_symbolic_value(value, only_if_contains)
