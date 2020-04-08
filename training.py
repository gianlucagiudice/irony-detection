from sklearn.model_selection import train_test_split
import json
from multiprocessing import Process, Manager
from sklearn import tree
from pathlib import Path
import pickle

from src.dataset.DataFrame import *
from src.Config import DATASET_PATH_OUT
from sklearn.metrics import classification_report, confusion_matrix
from src.Config import MODELS_PATH, REPORTS_PATH

from sklearn.model_selection import KFold

# Parameters
TARGET_DATASET = 'TwReyes2013'
TARGET_DATASET = 'Test'
N_FOLDS = 10


# add  dataframe, fold_number
def fit_classifier(X_train, y_train, X_test, y_test, classifier, fold, shared_dict, features):
    # Create classifier
    print('\n>>> Start training on fold {} . . .'.format(fold))
    clf = classifier()
    # Start training
    clf = clf.fit(X_train, y_train)
    # Test classifier
    y_pred = clf.predict(X_test)
    # Save report
    report_dict = dict()
    report_dict['report'] = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    report_dict['confusion_matrix'] = confusion_matrix(y_true=y_test, y_pred=y_pred).tolist()
    shared_dict[fold] = report_dict
    # Save model
    dump_classifier(features, clf, fold)
    # Print report
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print('Training on fold {} completed!'.format(fold))


def create_folder(dataset_name):
    Path(MODELS_PATH + dataset_name).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_PATH + dataset_name).mkdir(parents=True, exist_ok=True)


def extract_features_from_name(filename):
    return filename.split('.')[0].split('-')[1:]


def dump_classifier(features, classifier, fold):
    path = '{}{}/{}/'.format(MODELS_PATH, TARGET_DATASET, '-'.join(features), '/')
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = '{}${}_{}'.format('-'.join(features), type(classifier).__name__, fold)
    pickle.dump(classifier, open(path + filename + '.pickle', 'wb'))


def dump_report(dictionary):
    path = '{}{}/'.format(REPORTS_PATH, TARGET_DATASET)
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = '{}${}'.format(dictionary['features'], dictionary['classifier'])
    json.dump(dict(dictionary), open(path + filename + '.json', 'w'))


name = '/labeled_matrix-bow.csv'
path = DATASET_PATH_OUT + TARGET_DATASET + name

create_folder(TARGET_DATASET)
features = extract_features_from_name(name)
print(features)

print('Reading dataset . . .')
X, y = read_data_frame(path)
print('Read completed')


manager = Manager()
shared_dict = manager.dict()


process_pool = []
target_classifier = tree.DecisionTreeClassifier

shared_dict['features'] = '-'.join(features)
shared_dict['classifier'] = type(target_classifier()).__name__

for fold_number, (train_index, test_index) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True).split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    process_pool += [Process(target=fit_classifier,
                             args=(X_train, y_train, X_test, y_test,
                                   target_classifier, fold_number, shared_dict, features))]


# Start all processes
[process.start() for process in process_pool]
# Join all processes
[process.join() for process in process_pool]


# Save report
dump_report(shared_dict)
