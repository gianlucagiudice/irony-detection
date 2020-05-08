import time
from multiprocessing import Process, Manager, Lock

import psutil
from sklearn import tree, svm, naive_bayes
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from src.training.Logger import Logger

# Number of folds used for training
N_FOLDS = 10

# List of classifier to train
CLASSIFIER_LIST = [
    naive_bayes.MultinomialNB(),    # Multinomial Naive Bayes
    svm.SVC(kernel='linear'),       # Support Vector Machines
    tree.DecisionTreeClassifier()   # Decision Trees
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
        self.lock = Lock()
        # Logger
        self.logger = Logger.getLogger()

    def train_classifier(self):
        # Generate N folds for training
        self.generate_folds()
        # Create process pool
        self.create_process_pool()
        # Join all process
        self.join_all_processes()
        # Return shared dict
        return dict(self.shared_dict)

    def generate_folds(self):
        self.folds = KFold(n_splits=N_FOLDS).split(self.X)

    def create_process_pool(self):
        for fold_number, (train_fold, test_fold) in enumerate(self.folds, start=1):
            # Split data
            X_train, X_test = self.X[train_fold], self.X[test_fold]
            y_train, y_test = self.y[train_fold], self.y[test_fold]
            # Start new process
            self.start_new_process(X_train, y_train, X_test, y_test, fold_number)
        # Print info
        self.logger.print('All process started.')

    def fit_classifier(self, X_train, y_train, X_test, y_test, fold):
        start_time = time.time()
        # Start training
        try:
            clf = self.classifier.fit(X_train, y_train)
        except ValueError:
            return
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
        with self.lock:
            self.logger.print(output)

    def start_new_process(self, X_train, y_train, X_test, y_test, fold_number):
        # Create new process
        process = Process(target=self.fit_classifier,
                          args=(X_train, y_train, X_test, y_test, fold_number))
        # Start new process
        process.start()
        # Add process to process pool
        self.process_pool.append(process)
        # Print info
        self.logger.print(str(process))
        # Memory usage
        process_info = psutil.Process(process.pid)
        process_mem = process_info.memory_info()[0]
        # Wait for memory
        while True:
            time.sleep(5)
            available_mem = psutil.virtual_memory()[1]
            if available_mem - 1.5 * process_mem > 0:
                break
            else:
                time.sleep(10)
                pass

    def join_all_processes(self):
        [process.join() for process in self.process_pool]
