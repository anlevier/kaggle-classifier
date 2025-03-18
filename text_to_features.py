from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterable, Text
import numpy as np

class TextToFeatures:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            preprocessor=None,
            stop_words=None,
            ngram_range=(1, 3),
            analyzer="word",
            binary=False,
            min_df=0.2
        )

    def fit(self, training_texts: Iterable[Text]) -> None:
        print("ttf fit")
        self.vectorizer.fit(training_texts)

    def transform(self, texts: Iterable[Text]) -> np.ndarray:
        print("ttf transform")
        return self.vectorizer.transform(texts).toarray()

    def inverse_transform(self, feature_matrix: np.ndarray) -> Iterable[Text]:
        print("ttf invtrans")
        return self.vectorizer.inverse_transform(feature_matrix)
