# Irony Detection

#### Input
Insert dataset file with the associated `_lables.json` into
```
data/raw/DATASET_NAME/
```

#### Output
```
data/processed/DATASET_NAME/
```

#### Entry point
- Preprocessing
    ```
    /main.py 
    ```
- Training
    ```
    /training.py 
    ```
- PCA
    ```
    /pca.py 
    ```
- Analysis
    - Models report
        ```
        /notebooks/report_analysis_unbiased.ipynb
        ```
     - PCA
        ```
        /notebooks/pca*.ipynb
        ```
 