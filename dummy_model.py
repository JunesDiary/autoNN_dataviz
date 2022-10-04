from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
def dummy_model():
    # create dataset
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=15, random_state=1)
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(1, 51)]
    # evaluate a decision tree for each depth
    for i in values:
        # configure the model
        model = KNeighborsClassifier(n_neighbors=i)
        # fit model on the training dataset
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
