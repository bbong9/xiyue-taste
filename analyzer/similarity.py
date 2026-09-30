import numpy as np


def build_similarity(tracks):
    if not tracks:
        return np.empty((0, 53)), []
    matrix = np.asarray([track["features"]["vector"] for track in tracks], dtype=float)
    standard_deviation = matrix.std(axis=0)
    matrix = np.divide(
        matrix - matrix.mean(axis=0),
        standard_deviation,
        out=np.zeros_like(matrix),
        where=standard_deviation != 0,
    )
    weights = np.ones(53)
    weights[34:46] = 0.6
    weights[48:50] = 0.5
    weights[50:52] = 0.8
    weights[52] = 0.5
    matrix *= weights
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(
        matrix, lengths, out=np.zeros_like(matrix), where=lengths != 0
    )
    count = len(tracks)
    neighbor_count = min(20, count - 1)
    neighbors = []
    for start in range(0, count, 1000):
        stop = min(start + 1000, count)
        scores = matrix[start:stop] @ matrix.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        indices = np.argsort(-scores, axis=1)[:, :neighbor_count]
        for row, selected in zip(scores, indices):
            neighbors.append(
                [[int(index), round(float(row[index]), 4)] for index in selected]
            )
    return matrix, neighbors
