"""
Family-trees / kinship task (Hinton 1986).
"""

import numpy as np


ENGLISH_PEOPLE = [
    "Christopher", "Penelope", "Andrew", "Christine",
    "Margaret", "Arthur", "Victoria", "James",
    "Jennifer", "Charles", "Colin", "Charlotte",
]
ITALIAN_PEOPLE = [
    "Roberto", "Maria", "Pierro", "Francesca",
    "Gina", "Emilio", "Lucia", "Marco",
    "Angela", "Tomaso", "Alfonso", "Sophia",
]
RELATIONS = [
    "father", "mother", "husband", "wife",
    "son", "daughter", "uncle", "aunt",
    "brother", "sister", "nephew", "niece",
]


def build_triples():
    """Return all valid (person1, relation, person2) triples for both trees."""
    raise NotImplementedError


def split_train_test(triples, n_test: int = 4, seed: int = 0):
    raise NotImplementedError


def build_model():
    """24-input + 12-relation -> two 6-unit encoders -> 12 -> 6 -> 24 output."""
    raise NotImplementedError


def train(model, train_triples, n_sweeps: int = 1500, lr: float = 0.005):
    raise NotImplementedError


def inspect_person_encoding(model) -> np.ndarray:
    """Return the 6-D code for each of 24 people; expect interpretable axes."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
