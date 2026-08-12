"""Textual DSL parser: the action-definition front-end.

``RDDLParser`` is a small recursive-descent / shunting parser for textual predicate
expressions (e.g. ``GripperAt(g, o) and not(...)``). Built from three mappings
(operator symbol -> Operator, predicate name -> Predicate, type name -> Entity), it
alternately matches predicates and operators onto stacks, then folds operators by their
``ARITY`` into a nested ``Operator`` tree. Not a serializer and not on the sampling path.
"""
import re
from collections import deque
from typing import TypeVar

from rddl import Entity, Operand, Operator
from rddl.core import Variable

EntityType = TypeVar("EntityType", bound=Entity)  # all subclasses of Entity
OperatorType = TypeVar("OperatorType", bound=Operator, covariant=True)  # all subclasses of Operator
PredicateType = TypeVar("PredicateType", bound=Operand, covariant=True)  # all subclasses of Predicate  #FIXME: maybe should be predicate?



class RDDLParser:
    """Parses textual predicate expressions into nested ``Operator``/``Predicate`` trees."""

    # Matches a predicate head up to its opening paren, an argument up to a ',' or ')',
    # and a 'name:Type' variable declaration, respectively.
    pred_ex = re.compile(r'(?P<predicate>\w\S*)\(')
    arg_ex = re.compile(r'(?P<args>\w\S*)(?P<end>\s*[,\)]\s*)')
    var_ex = re.compile(r'(?P<var>\w\S*)\:(?P<type>\w\S*)(?P<end>\s*[,\)]\s*)')

    def __init__(self, combinator_mapping: dict[str, type[OperatorType]], predicate_mapping: dict[str, type[PredicateType]], type_definitions: dict[str, type[EntityType]]):
        """Store the operator/predicate/type mappings and build the operator-matching regex."""
        self.combinator_mapping = combinator_mapping
        self.predicate_mapping = predicate_mapping
        self.type_definitions = type_definitions
        self.all_ops = '|'.join([f'({op})' for op in combinator_mapping.keys()])
        self.op_ex = re.compile(fr'\s*(?P<op>{self.all_ops})\s*')
        self._entity_bindings: dict[str, Variable] = {}  # name -> Variable, shared within one parsed expression

    def _get_operand(self, name) -> type[Operand]:
        """Look up a predicate class by name; raise if not in the predicate mapping."""
        if name not in self.predicate_mapping:
            raise ValueError(f"Unknown function {name}! The mapping provided does not define such function!")
        return self.predicate_mapping[name]

    def _get_operator(self, symbol) -> type[Operator]:
        """Look up an operator class by symbol; raise if not in the combinator mapping."""
        if symbol not in self.combinator_mapping:
            raise ValueError(f"Unknown operator {symbol}! The mapping provided does not define such operator!")
        return self.combinator_mapping[symbol]

    @staticmethod
    def match_and_trim(regex, text):
        """Match ``regex`` at the start of ``text``; on success return it and the consumed remainder."""
        match = regex.match(text)
        if match:
            text = text[match.end():]
        return match, text

    @staticmethod
    def match_predicate(text):
        """Parse a leading ``name(arg, arg, ...)`` predicate; return (name, args, remaining text)."""
        predicate, args = None, None
        match, text = RDDLParser.match_and_trim(RDDLParser.pred_ex, text)
        if match:
            predicate = match.group('predicate')
            args = []
            while True:
                match, text = RDDLParser.match_and_trim(RDDLParser.arg_ex, text)
                if match:
                    args.append(match.group('args'))
                    if ")" in match.group('end'):
                        break
            print(f"{predicate}: {args}")
        return predicate, args, text

    # def create_predicate(self, predicate, args):
    #     true_args = [self.entity_bindings[arg] for arg in args]
    #     return self._get_function(predicate)(*true_args)

    def _resolve_arg(self, arg: str) -> Variable:
        """Resolve a textual argument to a Variable. A first occurrence is declared as
        ``name:Type`` (creating and remembering the variable); later occurrences reuse the
        name to share the same binding (same global name => same entity, see Variable)."""
        if ":" in arg:
            name, type_name = (part.strip() for part in arg.split(":", 1))
            if type_name not in self.type_definitions:
                raise ValueError(f"Unknown type {type_name}! The provided type definitions do not define such type!")
            variable = Variable(self.type_definitions[type_name], arg_name=name, global_name=name)
            self._entity_bindings[name] = variable
            return variable
        if arg in self._entity_bindings:
            return self._entity_bindings[arg]
        raise ValueError(f"Argument '{arg}' is used before its type is declared; declare it once as 'name:Type'.")

    def create_predicate(self, predicate, args):
        """Instantiate the predicate class for ``predicate``, binding ``args`` to its variable slots."""
        operand_class = self._get_operand(predicate)
        true_args = [self._resolve_arg(arg) for arg in args]
        kwargs = dict(zip(operand_class._VARIABLES.keys(), true_args))
        return operand_class(**kwargs)

    def parse_action_predicate(self, text):
        """Parse an action's predicate expression. Thin alias for ``parse``."""
        return self.parse(text)

    def parse(self, text):
        """Parse a predicate expression into a nested ``Operator``/``Predicate`` tree.

        Alternately matches a predicate then an operator onto operand/operator stacks, then
        folds operators (in order) by consuming ``ARITY`` operands each into a single tree.
        """
        self._entity_bindings = {}  # each parsed expression gets a fresh binding scope
        expression_stack = deque()
        operator_stack = deque()

        while True:
            predicate, args, text = RDDLParser.match_predicate(text)
            if predicate:
                current_predicate = self.create_predicate(predicate, args)
                expression_stack.append(current_predicate)
            else:
                break
            match, text = RDDLParser.match_and_trim(self.op_ex, text)
            if match:
                operator: type[Operator] = self._get_operator(match.group('op'))
                operator_stack.append(operator)
            else:
                break

        while operator_stack:
            operator = operator_stack.popleft()
            operands = [expression_stack.popleft() for _ in range(operator.ARITY)]
            expression_stack.appendleft(operator(*operands))

        full_expression = expression_stack.pop()
        return full_expression
