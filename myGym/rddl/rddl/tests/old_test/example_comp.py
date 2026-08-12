import numpy as np
from typing import Iterable

""" FUNCTIONS (will be defined outside, depending on specific implementation usage)"""
def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(point_A - point_B)


def is_holding(gripper, obj) -> bool:
    return gripper.is_holding(obj)

""" MAPPING ALLOWING TO DEFINE FUNCTIONS THAT WILL BE USED AS PREDICATES """
functions = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance
}
# CONSTANTS
NEAR_THRESHOLD = 0.01


class Entity:
    pass


""" BASIC CLASSES """
class Location(Entity):

    def __init__(self) -> None:
        self._loc = np.random.randn(3)

    def location(self) -> Iterable[float]:
        return self._loc


class ObjectEntity(Location):

    def location(self) -> Iterable[float]:
        return np.random.randn(3)


class Gripper(ObjectEntity):

    def is_holding(self, obj: ObjectEntity) -> bool:
        return np.random.randn() < 0


""" OPERANDS AND OPERATORS """
class Operand:

    def decide(self) -> bool:
        raise NotImplementedError("decide not implemented for generic operand")

    def evaluate(self) -> float:
        raise NotImplementedError("evaluate not implemented for generic operand")


class Predicate(Operand):
    pass


class Action(Operand):

    def __init__(self) -> None:
        self._predicate: Predicate

    def decide(self):
        print(f"Checking action {str(self.__class__)}")
        return self._predicate.decide()

    def evaluate(self):
        print(f"Evaluating action {str(self.__class__)}")
        return self._predicate.evaluate()


class Operator(Operand):
    _ARITY = None
    _SYMBOL = None

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    @property
    def ARITY(cls) -> int:
        return cls._ARITY


class UnaryOperator(Operator):
    _ARITY = 1

    def __init__(self, operand: Operand) -> None:
        super().__init__()
        self._operand = operand

    def __repr__(self) -> str:
        return f"{self._SYMBOL}{self._operand}"


class BinaryOperator(Operator):
    _ARITY = 2

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__()
        self._left: Operand = left
        self._right: Operand = right

    def __repr__(self) -> str:
        return f"{self._left} {self._SYMBOL} {self._right}"


"""   CLASSES """
class IsHolding(Predicate):
    _HOLDING_PREDICATE = functions["is_holding"]

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        super().__init__()
        self._gripper = gripper
        self._obj = obj

    def decide(self):
        print("Checking holding predicate")
        return IsHolding._HOLDING_PREDICATE(self._gripper, self._obj)

    def evaluate(self):
        print("Evaluating holding predicate")
        return IsHolding._HOLDING_PREDICATE(self._gripper, self._obj)


class EuclideanDistance(Predicate):
    _EDISTANCE_PREDICATE = functions["euclidean_distance"]

    def __init__(self, object_A: Location, object_B: Location) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def decide(self):    # how to handle, if it makes no sense?
        print("Checking euclidean distance predicate")
        return EuclideanDistance._EDISTANCE_PREDICATE(self.object_A.location(), self.object_B.location()) > 0

    def evaluate(self):
        print("Evaluating euclidean distance predicate")
        return EuclideanDistance._EDISTANCE_PREDICATE(self.object_A.location(), self.object_B.location())


class Near(Predicate):
    _EDISTANCE_PREDICATE = functions["euclidean_distance"]

    def __init__(self, object_A: Location, object_B: Location) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def decide(self):
        print("Checking near predicate")
        return Near._EDISTANCE_PREDICATE(self.object_A.location(), self.object_B.location()) < NEAR_THRESHOLD

    def evaluate(self):
        print("Evaluating near predicate")
        return Near._EDISTANCE_PREDICATE(self.object_A.location(), self.object_B.location()) - NEAR_THRESHOLD


class Not(Predicate):

    def __init__(self, predicate: Predicate) -> None:
        self._predicate = predicate

    def decide(self):
        print("Checking not predicate")
        return not self._predicate.decide()

    def evaluate(self):
        print("Evaluating not predicate")
        return -self._predicate.evaluate()


""" ACTION CLASSES """
class Reach(Action):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._predicate = Near(self._gripper, self._obj)

    def __repr__(self) -> str:
        return f"Reach({self._gripper}, {self._obj})"


class Grasp(Action):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._predicate = IsHolding(self._gripper, self._obj)

    def __repr__(self) -> str:
        return f"Grasp({self._gripper}, {self._obj})"


class Move(Action):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._predicate = EuclideanDistance(self._gripper, self._obj)

    def __repr__(self) -> str:
        return f"Move({self._gripper}, {self._obj})"


class Release(Action):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._predicate = Not(IsHolding(self._gripper, self._obj))

    def __repr__(self) -> str:
        return f"Release({self._gripper}, {self._obj})"


class Leave(Action):

    def __init__(self, gripper: Gripper, obj: ObjectEntity) -> None:
        self._gripper = gripper
        self._obj = obj
        self._predicate = Not(Near(self._gripper, self._obj))

    def __repr__(self) -> str:
        return f"Leave({self._gripper}, {self._obj})"


""" OPERATOR CLASSES """
class AndOp(BinaryOperator):
    _SYMBOL = "&"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def decide(self):
        left_check = self._left.decide()
        right_check = self._right.decide()
        print(f"Checking and operator; left: {left_check}, right: {right_check}, result: {left_check and right_check}")
        result = left_check and right_check
        return result

    def evaluate(self):
        left_eval = self._left.evaluate()
        right_eval = self._right.evaluate()
        print(f"Evaluating and operator; left: {left_eval}, right: {right_eval}, result: {left_eval + right_eval}")
        result = left_eval + right_eval
        return result


class SequentialOp(BinaryOperator):
    _SYMBOL = "->"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def decide(self):
        first_result = self._left.decide()
        after_result = self._right.decide()
        print(f"Checking sequential operator; first: {first_result}, after: {after_result}")
        return after_result

    def evaluate(self):
        first_evaluation = self._left.evaluate()
        after_evaluation = self._right.evaluate()
        print(f"Evaluating sequential operator; first: {first_evaluation}, after: {after_evaluation}")
        return first_evaluation + after_evaluation if self._left.decide() else first_evaluation


if __name__ == "__main__":
    gripper = Gripper()
    apple = ObjectEntity()
    loc = Location()
    action_pick_n_place = SequentialOp(
        SequentialOp(
            SequentialOp(
                AndOp(
                    Reach(gripper, apple),
                    Grasp(gripper, apple)
                ),
                Move(apple, loc)
            ),
            Release(gripper, apple)
        ),
        Leave(gripper, apple)
    )
    action = "reach(gripper, apple) & grasp(gripper, apple) -> move(apple, loc) -> release(gripper, apple) -> leave(gripper, apple)"
    print(action_pick_n_place.decide())
    print(action_pick_n_place.evaluate())
