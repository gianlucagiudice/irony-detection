import time

from multiprocessing import Process, Manager
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree, svm, naive_bayes

from src.training.Logger import Logger


# Number of folds used for training
N_FOLDS = 10

# List of classifier to train
CLASSIFIER_LIST = [
    naive_bayes.MultinomialNB(),    # Multinomial Naive Bayes
    svm.SVC(kernel='linear'),       # Support Vector Machines
    tree.DecisionTreeClassifier()   # Decision Trees
    # TODO: Bayesian Networks
]


class TrainingManager:
    def __init__(self, X, y, classifier):
        # Dataset
        self.X = X
        self.y = y
        # Training
        self.classifier = classifier
        # Shared dict among process
        self.shared_dict = Manager().dict()
        # Folds
        self.folds = None
        # Process pool
        self.process_pool = []
        # Logger
        self.logger = Logger.getLogger()

    def train_classifier(self):
        # Generate N folds for training
        self.generate_folds()
        # Create process pool
        self.create_process_pool()
        # Start all process
        self.start_all_processes()
        # Join all process
        self.join_all_processes()
        # Return shared dict
        return dict(self.shared_dict)

    def generate_folds(self):
        self.folds = KFold(n_splits=N_FOLDS).split(self.X)

    def create_process_pool(self):
        for fold_number, (train_fold, test_fold) in enumerate(self.folds, start=1):
            X_train, X_test = self.X[train_fold], self.X[test_fold]
            y_train, y_test = self.y[train_fold], self.y[test_fold]
            self.process_pool += [Process(target=self.fit_classifier,
                                          args=(X_train, y_train, X_test, y_test, fold_number))]

    def fit_classifier(self, X_train, y_train, X_test, y_test, fold):
        start_time = time.time()
        # Start training
        clf = self.classifier.fit(X_train, y_train)
        # Test classifier
        y_pred = clf.predict(X_test)
        # Save fold report
        self.shared_dict[fold] = {
            'report': classification_report(y_true=y_test, y_pred=y_pred, output_dict=True),
            'confusion-matrix': confusion_matrix(y_true=y_test, y_pred=y_pred).tolist()
        }
        # Print report
        elapsed_time = time.time() - start_time
        h_readable_elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
        output = '{}\n{}'.format(
            'Training on fold {} completed! ({})'.format(fold, h_readable_elapsed),
            classification_report(y_true=y_test, y_pred=y_pred))
        self.logger.print(output)

    def start_all_processes(self):
        [process.start() for process in self.process_pool]

    def join_all_processes(self):
        [process.join() for process in self.process_pool]