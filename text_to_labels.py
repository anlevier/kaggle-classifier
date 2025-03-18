from typing import Iterable, Text, Union
import numpy as np
from scipy.sparse import spmatrix
from sklearn.preprocessing import LabelEncoder
NDArray = Union[np.ndarray, spmatrix]

class TextToLabels:
  def __init__(self):
    self.lbl_encoder = LabelEncoder()

  def fit_transform(self, labels: Iterable[Text]) -> NDArray:
    self.lbl_encoder.fit(labels)
    return self.lbl_encoder.transform(labels)

  def index(self, label: Text) -> Union[None, int]:
    classes = self.lbl_encoder.classes_

    if label in classes:
      return self.lbl_encoder.transform([label])[0]
    else:
      return None

  def __contains__(self, label: Text) -> bool:
    return False if self.index(label) is None else True