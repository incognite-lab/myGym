"""Constraint registry keeping symbolically-sampled states logically consistent.

Rules index *predicate classes* (membership tested by ``__class__``). ``ExclusivityRule``
forbids more than one of its predicates being true; ``Consequent`` forward-chains forced
truths. ``RuleBook`` buckets rules and offers consistency checks and rule application over
the live symbolic cache. Rules ship empty — they are added externally.
"""
from typing import Any, Optional, Union
from sympy import Predicate

from rddl.core import LogicalOperand, Operand, Variable
from rddl.sampling_utils import StrictSymbolicCacheContainer


class Rule:
    """Base rule: a set of predicate classes, with membership tested by ``__class__``."""

    def __init__(self, predicate_classes) -> None:
        """Store the predicate classes this rule applies to."""
        self.__predicate_classes = predicate_classes

    def __contains__(self, predicate: Union[LogicalOperand, type[LogicalOperand]]) -> bool:
        """True if ``predicate`` (instance or class) belongs to this rule's predicate classes."""
        if isinstance(predicate, LogicalOperand):
            predicate = predicate.__class__
        return predicate in self.__predicate_classes


class ExclusivityRule(Rule):
    """
    Exclusive rule
    All predicates in this rule are mutually exclusive. When one is True, all other must be False.
    """

    def __init__(self, *predicates: type[LogicalOperand]):
        """Register the mutually exclusive predicate classes."""
        super().__init__(predicates)
        self._exclusive_predicates = predicates
        self._n_predicates = len(predicates)

    def check(self) -> bool:
        """True if no two of the rule's predicates currently ``decide()`` true together."""
        for i in range(self._n_predicates):
            for j in range(i + 1, self._n_predicates):
                if self._exclusive_predicates[i].decide() and self._exclusive_predicates[j].decide():
                    return False
        return True

    def list_breaking_predicates(self, exemplar: LogicalOperand, variables: Optional[list[Variable]] = None) -> list[type[LogicalOperand]]:
        """Return the predicate classes in this rule (other than the exemplar's) that are
        currently asserted true and therefore conflict with the exemplar under exclusivity."""
        exemplar_class = exemplar.__class__ if isinstance(exemplar, LogicalOperand) else exemplar
        if variables is None and isinstance(exemplar, LogicalOperand):
            variables = exemplar.gather_variables()
        cache = Operand.get_cache()
        if not isinstance(cache, StrictSymbolicCacheContainer):
            return []
        breaking = []
        for predicate_class, _predicate_variables, value in cache.get_predicates():
            if value and predicate_class is not exemplar_class and predicate_class in self._exclusive_predicates:
                breaking.append(predicate_class)
        return breaking


class Consequent(Rule):
    """
    Consequent rule
    When the exemplar predicate is true, the consequent must be true but not vice versa.
    """

    def __init__(self, exemplar: type[LogicalOperand], *consequences: type[LogicalOperand]):
        """Register the triggering ``exemplar`` and the consequences it forces true."""
        super().__init__(consequences + (exemplar,))
        self._exemplar = exemplar
        self._consequences = consequences

    def apply(self) -> None:
        """If the exemplar is true, force each consequence true in the symbolic cache."""
        if self._exemplar.decide():
            for predicate in self._consequences:
                predicate.set_symbolic_value(True)


class RuleBook:
    """Registry of exclusivity and consequent rules over the live symbolic cache."""

    def __init__(self) -> None:
        """Create empty rule buckets; rules must be added via ``add_rule``."""
        self._exclusivity_rules = []
        self._consequential_rules = []

    def add_rule(self, rule: Rule) -> None:
        """Bucket ``rule`` by type; raise on an unknown rule type."""
        if isinstance(rule, ExclusivityRule):
            self._exclusivity_rules.append(rule)
        elif isinstance(rule, Consequent):
            self._consequential_rules.append(rule)
        else:
            raise ValueError("Unknown rule type")

    def _get_current_predicate_set(self) -> list[tuple[Predicate, list[Variable], bool]]:
        """Return the live symbolic cache's predicates; raise unless it is a strict symbolic cache."""
        c = Operand.get_cache()
        if not isinstance(c, StrictSymbolicCacheContainer):
            raise ValueError("Current cache is not a 'StrictSymbolicCacheContainer'! Rule book cannot be used!")
        return c.get_predicates()

    def _construct_exclusivity_predicates(self) -> None:
        """WIP / unimplemented: intended to derive exclusivity predicates from the current state. Stub."""
        current_predicates = self._get_current_predicate_set()
        exclusivity_predicates = []
        for predicate, variables, value in current_predicates:
            if not value:
                continue
            for rule in self._exclusivity_rules:
                if predicate not in rule or not value:  # skip if predicate not in the rule or not true
                    continue


    def check_if_breaks_consistency(self, *predicates: LogicalOperand) -> bool:
        """Predictive consistency check. Returns True if asserting the given candidate
        predicates (or, when none are given, the current symbolic state as a whole) would
        violate an exclusivity rule, i.e. two predicates of different classes that share an
        exclusivity rule are both true. Non-mutating."""
        current_predicates = self._get_current_predicate_set()
        true_classes = [predicate_class for predicate_class, _variables, value in current_predicates if value]
        candidate_classes = [p.__class__ for p in predicates] if predicates else true_classes
        for candidate_class in candidate_classes:
            for rule in self._exclusivity_rules:
                if candidate_class not in rule:
                    continue
                for other_class in true_classes:
                    if other_class is not candidate_class and other_class in rule:
                        return True
        return False


    def check_consistency(self) -> bool:
        """True if every exclusivity rule currently holds."""
        for rule in self._exclusivity_rules:
            if not rule.check():
                return False
        return True

    def apply_rules(self) -> None:
        """Forward-chain every consequent rule, forcing their consequences true."""
        for rule in self._consequential_rules:
            rule.apply()
