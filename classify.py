from sklearn.linear_model import LogisticRegression
from typing import Union
import numpy as np
from scipy.sparse import spmatrix
NDArray = Union[np.ndarray, spmatrix]

class Classifier:
    def __init__(self):
        self.clf = LogisticRegression(
            C=10.0,
            solver='liblinear',
            penalty='l2',
            class_weight='balanced',
            max_iter=1500
        )

    def train(self, features: NDArray, labels: NDArray) -> None:
        self.clf.fit(features, labels)

    fit = train

    def predict(self, features: NDArray) -> NDArray:
        return self.clf.predict(features)
