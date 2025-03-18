import csv
from sklearn.model_selection import train_test_split

def split_data(input_file):
    with open(input_file, 'r') as input:
        reader = csv.DictReader(input)

        IDs = []
        X = []
        y = []
        for row in reader:
            IDs.append(row['ID'])
            X.append(row['TEXT'])
            y.append(row['LABEL'])

    X_train, X_test, y_train, y_test, IDs_train, IDs_test = train_test_split(X, y, IDs, test_size=0.2, random_state=42)

    with open('data_train.csv', 'w', newline='') as train_output:
        writer = csv.writer(train_output)
        writer.writerow(['ID', 'TEXT', 'LABEL'])
        writer.writerows(zip(IDs_train, X_train, y_train))

    with open('data_test.csv', 'w', newline='') as test_output:
        writer = csv.writer(test_output)
        writer.writerow(['ID', 'TEXT', 'LABEL'])
        writer.writerows(zip(IDs_test, X_test, y_test))

    train_texts = []
    train_labels = []

    with open('data_train.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            train_texts.append(row['TEXT'])
            train_labels.append(row['LABEL'])

    test_texts = []
    test_labels = []

    with open('data_test.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            test_texts.append(row['TEXT'])
            test_labels.append(row['LABEL'])

    return train_texts, train_labels, test_texts, test_labels
