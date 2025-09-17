# Mass Case Description Analysis and SVM Classification

This notebook analyzes the `mass_case_description_train_set_augmented_10k.csv` dataset and builds a Support Vector Machine (SVM) model to classify breast mass pathology as either benign or malignant.

## Project Description

The goal of this project is to preprocess and analyze a dataset containing information about breast masses and train a classification model to predict the likelihood of a mass being malignant based on its characteristics.

## Dataset

The dataset used in this project is `mass_case_description_train_set_augmented_10k.csv`. It contains various features related to breast masses, including breast density, image view, mass shape, mass margins, assessment, and pathology.

## Analysis and Modeling Steps

The notebook performs the following steps:

1.  **Data Loading and Initial Inspection:**
    -   Loads the dataset using pandas.
    -   Displays the first few rows, data types, and checks for missing values.

2.  **Data Cleaning and Preprocessing:**
    -   Removes duplicate rows.
    -   Renames the 'breast_density' column.
    -   Fills missing values in 'mass shape' and 'mass margins' with "Unknown".
    -   Encodes the 'pathology' column, merging 'BENIGN_WITHOUT_CALLBACK' into 'BENIGN' and mapping 'MALIGNANT' to 1 and 'BENIGN' to 0.
    -   Drops columns not relevant for training the model (patient ID, abnormality ID, image file paths).

3.  **Feature Engineering and Scaling:**
    -   Separates features (X) and the target variable (y).
    -   Applies one-hot encoding to categorical features in X.
    -   Standardizes the features using StandardScaler.

4.  **Model Training and Evaluation (Cross-Validation):**
    -   Uses Stratified K-Fold cross-validation (with 5 splits) to evaluate the SVM model's performance.
    -   Trains an SVM model with a radial basis function (rbf) kernel.
    -   Reports the accuracy, true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for each fold.
    -   Calculates and prints the mean accuracy and standard deviation across all folds.

5.  **Final Model Evaluation (Train/Test Split):**
    -   Splits the data into a training set (80%) and a hold-out test set (20%) using `train_test_split` with stratification.
    -   Standardizes the training and test sets separately.
    -   Trains the final SVM model on the training data.
    -   Evaluates the model on the hold-out test set.
    -   Prints the accuracy and a detailed classification report (precision, recall, f1-score).

6.  **Confusion Matrix Visualization:**
    -   Generates and displays a confusion matrix for the final model's predictions on the test set.
    -   Visualizes the confusion matrix using a heatmap.

## Results

The cross-validation and final test set evaluation show that the SVM model achieves a mean accuracy of approximately 89.5% in classifying breast mass pathology. The confusion matrix provides a detailed breakdown of the model's performance in terms of correctly and incorrectly classified instances.

## How to Run the Notebook

1.  Upload the `mass_case_description_train_set_augmented_10k.csv` file to your Colab environment.
2.  Run each code cell sequentially.

## Dependencies

The following libraries are required to run this notebook:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   sklearn
