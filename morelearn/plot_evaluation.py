import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_helpers import discrete_scatter
from .plot_classifiers import plot_classifiers
from .plot_2d_separator import plot_2d_classification, plot_2d_separator
from .tools import heatmap
from .datasets import make_forge, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

def plot_roc():
    X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svc = SVC(gamma=.05).fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    # find threshold closest to zero:
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
             label="threshold zero", fillstyle="none", c='k', mew=2)
    plt.legend(loc=4);


def plot_precision_recall_select():
    X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2],
                  random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svc = SVC(gamma=.05).fit(X_train, y_train)

    precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))
    close_zero = np.argmin(np.abs(thresholds))

    rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
    rf.fit(X_train, y_train)

    # RandomForestClassifier has predict_proba, but not decision_function
    # Only pass probabilities for the positive class
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
        y_test, rf.predict_proba(X_test)[:, 1])

    plt.plot(recall, precision, label="svc")

    plt.plot(recall[close_zero], precision[close_zero], 'o', markersize=10,
             label="threshold zero svc", fillstyle="none", c='k', mew=2)

    plt.plot(recall_rf, precision_rf, label="rf")

    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
    plt.plot( recall_rf[close_default_rf], precision_rf[close_default_rf], '^', c='k',
             markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc="best");

def plot_precision_recall_curve():
    X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svc = SVC(gamma=.05).fit(X_train, y_train)

    precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))
    # find threshold closest to zero:
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(recall[close_zero], precision[close_zero], 'o', markersize=10,
             label="threshold zero", fillstyle="none", c='k', mew=2)

    plt.plot(recall, precision, label="precision recall curve")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc="best")

def plot_kNN_overfitting_curve(k = range(1, 11)):
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66)

    # Build a list of the training and test scores for increasing k
    training_accuracy = []
    test_accuracy = []

    for n_neighbors in k:
        # build the model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
        # record training and test set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(k, training_accuracy, label="training accuracy")
    plt.plot(k, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    _ = plt.legend()

def plot_kNN_overfitting(k = [1, 3, 9]):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    X, y = make_forge()

    for n_neighbors, ax in zip(k, axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    _ = axes[0].legend(loc=3)

def plot_grid_results(c = [0.001, 0.01, 0.1, 1, 10, 100], gamma = [0.001, 0.01, 0.1, 1, 10, 100] ):
    param_grid = {'C': c,
              'gamma': gamma}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0)
    grid_search.fit(X_train, y_train)
    results = pd.DataFrame(grid_search.cv_results_)
    scores = np.array(results.mean_test_score).reshape(6, 6)
    heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
            ylabel='C', yticklabels=param_grid['C'], cmap="viridis");

def plot_holdout_predict_tree():
    names = ["Decision tree predictions"]

    classifiers = [
        DecisionTreeClassifier()
        ]

    plot_classifiers(names, classifiers, figuresize=(20,8))

def plot_holdout(nsamples=100):
    # create a synthetic dataset
    X, y = make_blobs(centers=2, random_state=0, n_samples = nsamples)
    # split data and labels into a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.legend(["Train class 0", "Train class 1", "Test class 0",
                    "Test class 1"], ncol=4,  loc=(-0.1, 1.1));

def plot_overfitting(degrees = [1, 4, 15]):
    np.random.seed(0)

    n_samples = 30

    true_fun = lambda X: np.cos(1.5 * np.pi * X)
    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()))
    plt.show()
