from sklearn.model_selection import train_test_split
import json
from multiprocessing import Process, Manager

from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB

from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
import pickle
import time
import os

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
    start = time.time()
    # Create classifier
    clf = classifier()
    # Start training
    clf = clf.fit(X_train, y_train)
    # Test classifier
    y_pred = clf.predict(X_test)
    # Save report
    report_dict = dict()
    report_dict['report'] = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
    report_dict['confusion-matrix'] = confusion_matrix(y_true=y_test, y_pred=y_pred).tolist()
    shared_dict[fold] = report_dict
    # Save model
    dump_classifier(clf, features, fold)
    # Print report
    elapsed = round(time.time()-start, 3)/60
    output = '{}\n{}'.format('Training on fold {} completed! ({}m)'.format(fold, elapsed),
                             classification_report(y_true=y_test, y_pred=y_pred))
    print(output)


def create_folder(dataset_name):
    Path(MODELS_PATH + dataset_name).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_PATH + dataset_name).mkdir(parents=True, exist_ok=True)


def extract_features_from_name(filename):
    return filename.split('.')[0].split('-')[1:]


def dump_classifier(classifier, features_list, fold):
    path = '{}{}/{}/'.format(MODELS_PATH, TARGET_DATASET, '-'.join(features_list), '/')
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = '{}${}_{}'.format('-'.join(features_list), type(classifier).__name__, fold)
    pickle.dump(classifier, open(path + filename + '.pickle', 'wb'))


def dump_report(dictionary):
    path = '{}{}/'.format(REPORTS_PATH, TARGET_DATASET)
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = '{}${}'.format(dictionary['features'], dictionary['classifier'])
    json.dump(dictionary, open(path + filename + '.json', 'w'))


def evaluate_overall_folds(folds_dict):
    # Build overall dict
    report_type = ["False", "True", "macro avg", "weighted avg"]
    target_report = ["precision", "recall", "f1-score", "support"]
    overall_report = {key: {value: [] for value in target_report} for key in report_type}
    accuracy_list = []
    overall_confusion_matrix = np.zeros((2, 2))
    # Visit dict
    for fold in folds_dict.values():
        for key in report_type:
            for target in target_report:
                overall_report[key][target] += [fold["report"][key][target]]
        accuracy_list.append(fold["report"]["accuracy"])
        overall_confusion_matrix += np.array(fold["confusion-matrix"])
    # Build overall report
    report_dict = {key: {value: 0 for value in target_report} for key in report_type}
    for key in report_type:
        for target in target_report[:-1]:
            report_dict[key][target] = float(np.mean(overall_report[key][target]))
        report_dict[key]["support"] = float(np.sum(overall_report[key]["support"]))
    report_dict["accuracy"] = float(np.mean(accuracy_list))
    # Build out dict
    out_dict = dict()
    out_dict["report"] = report_dict
    out_dict["confusion-matrix"] = overall_confusion_matrix.tolist()
    # Return dict
    return out_dict


def matrix_filename():
    # TODO: Sistemare questa linea
    files = [file for file in os.listdir(DATASET_PATH_OUT + TARGET_DATASET) if 'labeled_matrix' in file]
    return files


matrix_file_list = matrix_filename()
classifier_list = [tree.DecisionTreeClassifier, GaussianNB, svm.SVC]


# TODO: Da sistemare sto obrobrio
for idx_file, target_file in enumerate(matrix_file_list, 1):
    print(">>> Target file {} - N. {}/{}".format(target_file, idx_file, len(matrix_file_list)))
    for idx_classifier, target_classifier in enumerate(classifier_list, 1):
        print(">> Classifier {}. {}/{} - N. {}/{}".format(type(target_classifier()).__name__,
                                                          idx_file, len(matrix_file_list),
                                                          idx_classifier, len(classifier_list)))

        path = DATASET_PATH_OUT + TARGET_DATASET + '/' + target_file
        create_folder(TARGET_DATASET)
        features = extract_features_from_name(target_file)
        print(features)

        print('Reading dataset . . .')
        X, y = shuffle(*read_data_frame(path))
        print('Read completed')

        manager = Manager()

        process_pool = []
        target_classifier = tree.DecisionTreeClassifier

        model_report = dict()
        model_report['features'] = '-'.join(features)
        model_report['classifier'] = type(target_classifier()).__name__

        shared_dict = manager.dict()
        folds = KFold(n_splits=N_FOLDS, shuffle=True).split(X)
        print('\n>>> Start training ')
        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            process_pool += [Process(target=fit_classifier,
                                     args=(X_train, y_train, X_test, y_test,
                                           target_classifier, fold_number, shared_dict, features))]

        # Start all processes
        [process.start() for process in process_pool]
        # Join all processes
        [process.join() for process in process_pool]

        model_report['folds'] = dict(shared_dict)
        model_report['overall'] = evaluate_overall_folds(model_report['folds'])

        # Save report
        dump_report(model_report)
