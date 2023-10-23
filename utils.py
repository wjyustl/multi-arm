import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """

    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def distance_list(a: list, b: list) -> np.ndarray:
    """Compute the distance between two list's array. This function is vectorized.

    Args:
        a (np.ndarray): First list.
        b (np.ndarray): Second list.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert len(a) == len(b), "Check the distance_list's inputs' len;"
    result = np.zeros(3)
    for i in range(len(a)):
        result += np.linalg.norm(a[i] - b[i], axis=-1)
    result = result / len(a)
    return result


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values
        figure = plt.figure()
        figure.add_subplot().plot()  ##need a value
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True
