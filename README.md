# scMFF: A Machine Learning Framework with Multiple Feature Fusion Strategies for Cell Type Identification 

## Project Overview
scMFF is a Python toolkit for cell type classification using clinical transcriptome data, implementing a model with Multiple Features Fusion Strategies. 

## Model Architecture
![The Model Architecture of scKSFD](https://github.com/BMBDM/scMFF/blob/main/Model.png)

## Installation Guide
- Clone the repository:
  ```
  git clone https://github.com/BMBDM/scMFF.git
  cd scMFF
  ```
- Install the dependencies:
  ```
  pip install -r requirements.txt
  ```
- Optionally, install the toolkit as an importable Python package:
  ```
  pip install .
  ```
  
## Usage Example
  ```
 from scMFF_pipeline import *
dir = 'Example_data/'
model_name = 'KNN'

scMFF_model(dir, model_name)
  ```
### Input
The input data should be in CSV format and placed in the Example_data/ directory:
- expr.csv
- cluster.csv

expr.csv format example:

|             | Cell_1 | Cell_2 | Cell_3 | Cell_4 |
|-------------|--------|--------|--------|--------|
| GeneA       |   10   |   0    |   3    |   8    |
| GeneB       |   5    |   2    |   0    |   4    |
| GeneC       |   0    |   7    |   1    |   0    |
| GeneD       |   3    |   3    |   2    |   5    |
| GeneE       |   6    |   1    |   0    |   2    |

cluster.csv format example:

| Cell type 1       |
|-------------      |
| Cell type 2       |
| Cell type 1       |
| Cell type 3       |


### Parameters
- model_name: KNN、SVM、RandomForest、XGBoost、LightGBM、NaiveBayes、NeuralNetwork

### Output
| Metric    | feature1 | feature2 | feature3 | feature4 | feature5 | feature6 | feature7 | feature8 | feature9 | feature10 |
| --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- |
| Accuracy  | 0.960309 | 0.979320 | 0.975985 | 0.974650 | 0.982990 | 0.758835 | 0.955302 | 0.980322 | 0.977319 | 0.979654  |
| Precision | 0.962588 | 0.979444 | 0.976576 | 0.974951 | 0.983186 | 0.761053 | 0.956969 | 0.980684 | 0.977808 | 0.979905  |
| Recall    | 0.960309 | 0.979320 | 0.975985 | 0.974650 | 0.982990 | 0.758835 | 0.955302 | 0.980322 | 0.977319 | 0.979654  |
| F1        | 0.960076 | 0.979313 | 0.975896 | 0.974613 | 0.982959 | 0.753840 | 0.955218 | 0.980296 | 0.977275 | 0.979609  |
| MCC       | 0.949851 | 0.973477 | 0.969405 | 0.967539 | 0.978250 | 0.689785 | 0.943005 | 0.974869 | 0.971040 | 0.973980  |

PS：Feature Embedding Files and Their Dimensions

| Feature ID | File Name                                                               | Dimension      |
|------------|-------------------------------------------------------------------------|----------------|
| Feature0   | expr.csv                                                                | (33694, 2998)  |
| Feature1   | expr_2000.csv                                                           | (2000, 2998)   |
| Feature2   | scPSD_embedding.txt.gz                                                  | (2998, 2000)   |
| Feature3   | PCA_embedding.txt.gz                                                    | (2998, 266)    |
| Feature4   | scGNN_embedding.csv                                                     | (2998, 129)    |
| Feature5   | Fusion_sum-expr-scpsd-scgnn-pca.txt.gz                                  | (2998, 2000)   |
| Feature6   | Fusion_hadamard-expr-scpsd-scgnn-pca.txt.gz                             | (2998, 2000)   |
| Feature7   | Fusion_attention-expr-scpsd-scgnn-pca.txt.gz                            | (2998, 2000)   |
| Feature8   | Fusion_moe-expr-scpsd-scgnn-pca.txt.gz                                  | (2998, 128)    |
| Feature9   | Fusion_residual-expr-scpsd-scgnn-pca.txt.gz                             | (2998, 128)    |
| Feature10  | Fusion_transformer-expr-scpsd-scgnn-pca.txt.gz                          | (2998, 128)    |

- 2998 cells

## Contact Us
If you have any questions or suggestions, please contact us via email: [sunn19@tsinghua.org.cn].
