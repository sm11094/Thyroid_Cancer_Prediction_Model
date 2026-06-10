# Thyroid Cancer Prediction System

# A robust, from-scratch implementation of a K-Dimensional Tree designed to classify patient medical data in 9-dimensional space.
# Takes patient details as input and returns a prediction of Thyroid Cancer Risk Level as well as Tumor diagnosis
# Boosted performance via KD-Tree - Promised O(log n) time complexity throughout


## Project Architecture
The project is modularized into distinct files:
* **`crud.py`**: Handles all KD-Tree structural operations (Create, Read/Search, Insert, Delete).
* **`kNN.py`**: Contains the core mathematical engines for spatial distance computation and retrieving the *k* nearest neighbors.
* **`predict.py`**: Contains the voting logic and the label decoder to translate raw outputs into readable medical diagnoses.


**Time complexity** 
# Currently, average case time complexities are as follows:
# build_kdtree --> O(n log n) . The tree will only be built once and thats that. It won't need to be rebuilt after every operation.
# search and insert --> O(log n)
# delete --> O(n^(1-1/k)) where k is the number of dimensions. So this is almost O(n). This is due to splitting axis discrepency between desired splitting axis and current splitting axis
# search_knn / get_knn --> O(log n). 
# generate_prediction --> O(k) where k is the number of nearest neighbors. We have used 5 in the demo, might increase or decrease depending on accuracy
# deduce_label --> O(1)


**Limitations**
# Since this is just a basic k nearest neighbors algorithm, it contains the following limitations:
# 1. Equal Feature Weighting --> features such as age, family history, have equal weights (influence) on the generated prediction even though a huge nodule size should have a greater influence.
# 2. Class Imbalance --> In real world scenarios, there are more benign cases then malignant cases and that is reflected on our dataset of choice. It contains a lot more benign cases and as a result the chances that a nearest neighbor is malignant which causes predictions to be more benign than malignant even though a case may look heavily malignant
# 3. For extreme cases, the project would rather let the doctor diagnose the issue rather than make a bad prediction

## How to Run
1. Ensure both `features.csv` and `labels.csv` are in the root directory.
2. Run the main file from your terminal:
   python gui.py


## Authors:
# Zaid Ahmed
# Ahsan Ali
# Syed Muhammad Mohsin