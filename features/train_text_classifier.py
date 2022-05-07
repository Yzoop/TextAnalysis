import json  # allows reading json data into python dictionary

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split  # splits data into train and validational sets
from sklearn.feature_extraction.text import TfidfVectorizer  # converts texts to vectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score  # beautiful metrics report
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer


def texts_labels_from_data(path, dataset_size=1):
    with open(path) as f:
        # read tagot labeled data
        data = json.load(f)
    # TODO: change dataset size from 0 (0 examples) to 1 (all examples). 0.6 - 60% of examples
    np.random.seed(17)
    np.random.shuffle(data)
    data = data[:int(len(data) * dataset_size)]
    # split (text, label) pairs into 2 seperate lists for train_test_split function
    texts, labels = [item["text"] for item in data], [item["annotation"][0] for item in data]
    return texts, labels


def convert_data_to_vectors(texts):
    # initialize a tfidf vectorizer object
    tfidf_vectorizer = TfidfVectorizer(ngram_range=[1, 3])
    # converts text into vectors
    text_vectors = tfidf_vectorizer.fit_transform(texts)
    return text_vectors


def train_classifier(train_x, train_y):
    logreg_model = LogisticRegression(random_state=17, penalty="none", n_jobs=3)
    logreg_model.fit(train_x, train_y)
    return logreg_model


def calculate_metrics(model, train_x, test_x, train_y, test_y):
    print("Train accuracy:", accuracy_score(train_y, model.predict(train_x)))
    print("Test accuracy:", accuracy_score(test_y, model.predict(test_x)))
    prediction_test_y = model.predict(test_x)
    report = classification_report(test_y, prediction_test_y, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df


def plot_validation_curve(texts, labels, dataset_sizes):
    micro_avgs = []
    text_vectors = convert_data_to_vectors(texts)
    train_x, test_x, train_y, test_y = train_test_split(text_vectors,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=17)
    for size in dataset_sizes:
        train_x_samples, train_y_samples = train_x[:int(train_x.shape[0] * size), :], \
                                           train_y[:int(len(train_y) * size)]
        classifier = train_classifier(train_x_samples, train_y_samples)
        metrics = calculate_metrics(classifier, train_x_samples, test_x, train_y_samples, test_y)
        micro_avg_f1 = metrics.loc[pd.Index(["macro avg"]), pd.Index(["f1-score"])].values[0][0]
        micro_avgs.append(micro_avg_f1)
    ax = sns.lineplot(x=dataset_sizes, y=micro_avgs)
    ax.set(xlabel='dataset size', ylabel='avg f1')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    texts, labels = texts_labels_from_data("../data/text_classification_dataset_2022-05-07.json")
    text_vectors = convert_data_to_vectors(texts)
    # split data into train and test(validational) sets. P.S. test_size means the size of the test set,
    # so if we set test_size=0.2, we take 20% of the data to test set. The model will not use it to train itself
    # random_state allows us to have the same output, something like freeze random to have the same results
    # everytime we run the script.
    train_x, test_x, train_y, test_y = train_test_split(text_vectors,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=17)
    multiclass_classifier = train_classifier(train_x, train_y)
    metrics = calculate_metrics(multiclass_classifier, train_x, test_x, train_y, test_y)
    print(metrics)
    plot_validation_curve(texts, labels, dataset_sizes=np.arange(0.2, 1, step=0.1)) # do not set 0, cause it means 0 examples
