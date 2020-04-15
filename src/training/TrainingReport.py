import json
import numpy as np
from pathlib import Path

from src.config import REPORTS_PATH, TARGET_DATASET


class TrainingReport:
    def __init__(self, features, classifier, folds_report):
        # Model report
        self.model_report = dict()
        # Report data
        self.features = features
        self.classifier = classifier
        self.folds_report = folds_report

    def create_report(self):
        # Create report metadata
        self.create_report_metadata()
        # Save folds report
        self.create_report_data()
        # Dump report
        self.dump_report()
        
    def create_report_metadata(self):
        # List of features
        self.model_report['features'] = '-'.join(self.features)
        # Classifier
        self.model_report['classifier'] = type(self.classifier).__name__

    def create_report_data(self):
        # Folds report
        self.model_report['folds'] = self.folds_report
        # Overall report
        self.model_report['overall'] = self.evaluate_overall_folds()

    def evaluate_overall_folds(self):
        report_type = ["False", "True", "macro avg", "weighted avg"]
        target_report = ["precision", "recall", "f1-score", "support"]

        def evaluate_folds_values():
            overall_report = {key: {value: [] for value in target_report} for key in report_type}
            accuracy_list = []
            overall_confusion_matrix = np.zeros((2, 2))
            # Extract dict value
            for fold in self.folds_report.values():
                for key in report_type:
                    for field in target_report:
                        overall_report[key][field] += [fold["report"][key][field]]
                accuracy_list.append(fold["report"]["accuracy"])
                overall_confusion_matrix += np.array(fold["confusion-matrix"])
            return overall_report, accuracy_list, overall_confusion_matrix

        reports, accuracy, confusion_matrix = evaluate_folds_values()
        # Combine values
        report_dict = {key: {value: 0 for value in target_report} for key in report_type}
        for key in report_type:
            for target in target_report[:-1]:
                report_dict[key][target] = float(np.mean(reports[key][target]))
            report_dict[key]["support"] = float(np.sum(reports[key]["support"]))
        report_dict["accuracy"] = float(np.mean(accuracy))
        # Return overall dict
        return {
            'report': report_dict,
            'confusion-matrix': confusion_matrix.tolist()
        }

    def dump_report(self):
        PATH = '{}{}/'.format(REPORTS_PATH, TARGET_DATASET)
        Path(PATH).mkdir(parents=True, exist_ok=True)
        filename = '{}${}'.format(self.model_report['features'], self.model_report['classifier'])
        json.dump(self.model_report, open('{}{}.json'.format(PATH, filename), 'w'))

