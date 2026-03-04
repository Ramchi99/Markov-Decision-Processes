from typing import Callable, Mapping
from frozendict import frozendict

from pdm4ar.exercises_def import *
from pdm4ar.exercises_def.structures import Exercise

available_exercises: Mapping[str, Callable[[], Exercise]] = frozendict(
    {
        "04": get_exercise4,
    }
)
