import random


def sample_k(elements: list, k: int) -> list:
    if k > len(elements):
        return elements
    return random.sample(elements, k)
