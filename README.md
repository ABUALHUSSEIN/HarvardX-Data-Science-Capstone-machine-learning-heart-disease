# HarvardX-Data-Science-Capstone-machine-learning-heart-disease

📊 Heart Disease Prediction Using Machine Learning
This project aims to predict the presence of heart disease using various machine learning classification models. It was completed as part of an edX course final project, using the tidymodels framework in R.

🧠 Models Used
Logistic Regression

k-Nearest Neighbors (kNN)

Support Vector Machine (SVM)

Naive Bayes

Random Forest

XGBoost

⚙️ Key Features
Custom preprocessing pipelines tailored to each model (e.g., normalization for SVM/kNN, dummy variables for tree-based models).

Feature engineering including:

Handling missing values

Transforming skewed features (log, Box-Cox)

One-hot encoding categorical variables

Correlation filtering

Hyperparameter tuning using cross-validation and grid_latin_hypercube().

Evaluation metrics including:

Accuracy

Confusion Matrix

ROC Curve and AUC

📁 Project Structure
bash
نسخ
تحرير
heart-disease-prediction/
├── heart_disease_project.Rmd   # RMarkdown analysis file
├── heart_disease_project.pdf   # Rendered report
├── data/                       # Dataset folder
│   └── heart.csv
├── README.md                   # Project documentation
📌 Dataset
The dataset was sourced from this GitHub repository. It contains features related to patient medical history and cardiovascular metrics.

🧪 Getting Started
To run the project:

Open heart_disease_project.Rmd in RStudio.

Knit the document to generate a PDF report.

Ensure all required packages (e.g., tidymodels, dplyr, ggplot2, etc.) are installed.

📈 Results
Each model was evaluated and compared using cross-validation. Final results showed that Logistic Regression performed the best based on AUC and overall accuracy.
