import subprocess

from src.utils.config import REPORTS_PATH, WEKA_PATH, DATASET_PATH_OUT
from src.utils.parameters import TARGET_DATASET
from training import create_report_folder, extract_features_from_name, read_matrix_filename

COMMAND = ['java', '-classpath', WEKA_PATH, 'weka.classifiers.bayes.BayesNet',
           '-x', '10', '-t', '{}', '-D', '-Q', 'weka.classifiers.bayes.net.search.local.K2',
           '--', '-P', '1', '-S', 'BAYES', '-E','weka.classifiers.bayes.net.estimate.SimpleEstimator',
           '--', '-A', '0.5']


def dump_report(text, features):
    path = '{}{}.weka/'.format(REPORTS_PATH, TARGET_DATASET)
    filename = '{}${}'.format('-'.join(features), 'BayesNet')
    with open('{}{}.txt'.format(path, filename), 'w') as out:
        out.write(text)


def main():
    # Title
    print("\t\t\t{} TRAINING MODELS ON DATASET: {} {}\n".format("=" * 5, TARGET_DATASET, "=" * 5))
    # Create report folder
    create_report_folder(optional=".weka")
    matrix_file_list = read_matrix_filename()
    # Iterate over each features
    for idx_file, target_file in enumerate(matrix_file_list, 1):
        print("\t\t%%%%% Target file {} - N. {}/{} %%%%%\n".format(target_file, idx_file, len(matrix_file_list)))
        path = '{}{}/{}'.format(DATASET_PATH_OUT, TARGET_DATASET, target_file)
        # Train model
        command = ' '.join(COMMAND).format(path)
        output = subprocess.run(command,  stdout=subprocess.PIPE, text=True, shell=True).stdout
        # Dump report
        print(output)
        dump_report(output, extract_features_from_name(target_file))


main()
