# Irony Detection
More info about the project can be found here:
[Thesis](https://github.com/gianlucagiudice/irony-detection/blob/master/references/thesis/tesi.pdf)

### Requirements
- python3.7
- Java 6 or above
- Libraries listed in `requirements.txt`
    
    Run `pip3 install -r requirements.txt` to install all python libraries
    
### Input
Put the dataset into:
```
data/raw/DATASET_NAME/
```
Also write the associated labels in the `_lables.json` file within the same folder.
### Output
```
data/processed/DATASET_NAME/
```
### How to execute the program

#### Single script
Run `run.sh`. This involves:
1. Feature extraction using three different strategy for text representation:
    1. BOW
    2. BERT
    3. Sentence-BERT
2. Training of the models
3. Perform PCA
#### Manual version
- Feature extraction
    ```
    /main.py TARGET_DATASET TEXT_REPRESENTATION
    ```
  Parameters:
     - TARGET_DATASET = The name of the dataset
     
     - TEXT_REPRESENTATION = Strategy used for text representation.
        Valid values are:
        - bow
        - bert
        - sbert
- Training
    ```
    /training.py TARGET_DATASET
    ```
 - Weka experiment converter
 
    ```
    /notebooks/weka_experiment_converter.ipynb 
    ```
 
    This notebook is used to convert `.csv` format of the weka experiments to the `.json` format which is 
    consistent with the output produced by the scikit-learn reports output.
    
    In order to convert the reports just place `.csv` weka output into the folder
    `reports/TARGET_DATASET/weka_experiment.csv`.
    
    Be aware that the name must be `weka_experiment.csv`

- PCA
    ```
    /pca.py 
    ```
  
#### Analysis
In order to analyze the reports and PCA output, several notebooks have been created.
- Models report - Performance comparison
    ```
    /notebooks/report_analysis.ipynb
    ```
- PCA
    ```
    /notebooks/pca*.ipynb
    ```
 
