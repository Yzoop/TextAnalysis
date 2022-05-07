import json  # allows reading json data into python dictionary

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split  # splits data into train and validational sets
from sklearn.feature_extraction.text import TfidfVectorizer  # converts texts to vectors
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier  # classifier model
from sklearn.metrics import classification_report, accuracy_score  # beautiful metrics report
from sklearn.preprocessing import MultiLabelBinarizer


def texts_labels_from_data(path, dataset_size=1):
    with open(path) as f:
        # read tagot labeled data
        data = json.load(f)
    # TODO: change dataset size from 0 (0 examples) to 1 (all examples). 0.6 - 60% of examples
    np.random.seed(17)
    np.random.shuffle(data)
    data = data[:int(len(data) * dataset_size)]
    # split (text, label) pairs into 2 seperate lists for train_test_split function
    texts, labels = [item["text"] for item in data], [item["annotation"] for item in data]
    return texts, labels


def convert_data_to_vectors(texts, labels, return_binarizer=False):
    # initialize a tfidf vectorizer object
    tfidf_vectorizer = TfidfVectorizer()
    # converts text into vectors
    text_vectors = tfidf_vectorizer.fit_transform(texts)
    # convert labels to 0/1 vectors, i.e. imagine we have a text and 2 labels: ["POLYTICS", "HISTORY"].
    # if we have 5 labels in classification problem, we will have y vector:
    # [0, 1, 0, 0, 1], i.e. first zero. see https://scikit-learn.org/stable/modules/multiclass.html
    label_binarizer = MultiLabelBinarizer()
    label_vectors = label_binarizer.fit_transform(labels)
    # split data into train and test(validational) sets. P.S. test_size means the size of the test set,
    # so if we set test_size=0.2, we take 20% of the data to test set. The model will not use it to train itself
    # random_state allows us to have the same output, something like freeze random to have the same results
    # everytime we run the script.
    if return_binarizer:
        return text_vectors, label_vectors, label_binarizer
    return text_vectors, label_vectors


def train_classifier(train_x, train_y):
    logreg_model = LogisticRegression(random_state=17, penalty="none")
    multiclass_classifier = MultiOutputClassifier(
        logreg_model)  # multiclass classifier https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    multiclass_classifier.fit(train_x, train_y)
    return multiclass_classifier


def calculate_metrics(model, label_binarizer, train_x, test_x, train_y, test_y):
    print("Train accuracy:", accuracy_score(train_y, model.predict(train_x)))
    print("Test accuracy:", accuracy_score(test_y, model.predict(test_x)))
    prediction_test_y = model.predict(test_x)
    report = classification_report(test_y, prediction_test_y, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    index_dict = {idx: class_name for idx, class_name in zip(report_df.index, label_binarizer.classes_)}
    report_df.rename(index=index_dict,
                     inplace=True)
    return report_df


def plot_validation_curve(texts, labels, dataset_sizes):
    micro_avgs = []
    for size in dataset_sizes:
        texts_samples, labels_samples = texts[:int(len(texts) * size)], labels[:int(len(labels) * size)]
        text_vectors, label_vectors, label_binarizer = convert_data_to_vectors(texts_samples,
                                                                               labels_samples,
                                                                               return_binarizer=True)
        train_x, test_x, train_y, test_y = train_test_split(text_vectors,
                                                            label_vectors,
                                                            test_size=0.2,
                                                            random_state=17)
        multiclass_classifier = train_classifier(train_x, train_y)
        metrics = calculate_metrics(multiclass_classifier, label_binarizer, train_x, test_x, train_y, test_y)
        micro_avg_f1 = metrics.loc[pd.Index(["micro avg"]), pd.Index(["f1-score"])].values[0][0]
        micro_avgs.append(micro_avg_f1)
    ax = sns.lineplot(x=dataset_sizes, y=micro_avgs)
    ax.set(xlabel='dataset size', ylabel='avg f1')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    texts, labels = texts_labels_from_data("../data/text_classification_dataset_2022-05-07.json")
    text_vectors, label_vectors, label_binarizer = convert_data_to_vectors(texts,
                                                                           labels,
                                                                           return_binarizer=True)
    train_x, test_x, train_y, test_y = train_test_split(text_vectors,
                                                        label_vectors,
                                                        test_size=0.2,
                                                        random_state=17)
    multiclass_classifier = train_classifier(train_x, train_y)
    metrics = calculate_metrics(multiclass_classifier, label_binarizer, train_x, test_x, train_y, test_y)
    print(metrics)
    plot_validation_curve(texts, labels, dataset_sizes=[0.2, 0.25, 0.5, 0.6, 0.7,  0.8, 0.9, 0.95])
