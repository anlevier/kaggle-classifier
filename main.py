import sys
import csv
import itertools
import random
from train_test_split import split_data
from text_to_features import TextToFeatures
from text_to_labels import TextToLabels
from classify import Classifier
from sklearn.metrics import accuracy_score

def main():
    train = "train.csv"
    test = "test_data.csv"
    label = "test.csv"

    train_texts, train_labels, test_texts, test_labels = split_data(train)

    ttf = TextToFeatures()
    ttf.fit(train_texts)

    ttl = TextToLabels()
    train_encoded_labels = ttl.fit_transform(train_labels)

    clf = Classifier()
    clf.train(ttf.transform(train_texts), train_encoded_labels)

    label_new_data(test, label, clf, ttf)


def label_new_data(input_file, output_file, classifier, text_to_features):
    with open(input_file, 'r') as input:
        reader = csv.DictReader(input)
        rows = list(reader)

    texts = [row['TEXT'] for row in rows]

    features = text_to_features.transform(texts)

    predicted_labels = classifier.predict(features)

    fieldnames = reader.fieldnames
    fieldnames.remove('TEXT')
    fieldnames.append('LABEL')

    labeled_data = []
    for row, label in zip(rows, predicted_labels):
        row['LABEL'] = label
        labeled_data.append({key: row[key] for key in fieldnames})

    with open(output_file, 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled_data)

if __name__ == '__main__':
    main()