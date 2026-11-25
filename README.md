Hospital Readmission Prediction Using Machine Learning
1. Project Overview
This project is based on Machine Learning to predict whether a diabetic patient is likely to be readmitted to a hospital within 30 days after discharge.

The system analyzes patient information and uses a trained Logistic Regression model to generate a readmission risk score.

This project is developed as an academic project for Brainware University.

2. Project Information
Project Title: Machine Learning-Based Prediction of Hospital Readmission Risk
Developed By: Aishik Mondal
Student ID: BWU/BCA/24/131
University: Brainware University
Course: BCA
3. Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Joblib
Jupyter Notebook
4. Dataset
This project uses the Diabetes 130-US Hospitals for Years 1999–2008 dataset from the UCI Machine Learning Repository.

The dataset contains hospital encounter information about diabetic patients.

The target variable is created from the readmitted column:

Class 1: Readmitted within 30 days (<30)
Class 0: Not readmitted or readmitted after 30 days
5. Machine Learning Algorithms
Three machine learning algorithms were compared:

Logistic Regression
Decision Tree
Random Forest
Logistic Regression was selected as the final model based on its overall validation performance.

6. Model Performance
The following results are from the validation dataset:

Evaluation Metric	Logistic Regression
Accuracy	69.76%
Precision	17.49%
Recall	49.33%
F1-Score	27.05%
ROC-AUC	0.6615
PR-AUC	0.2001
The classification threshold was selected using the validation dataset.

7. Main Features
Patient information input
Machine learning-based prediction
Readmission risk score
Risk classification using a selected threshold
Interactive Streamlit web application
Model comparison and evaluation
Model testing and validation
8. Project Structure
Hospital_Readmission_Prediction/
│
├── data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
│
├── app.py
├── demo.py
├── hospital_readmission.ipynb
├── compare_models.py
├── independent_validation.py
├── test_model.py
├── test_final_models.py
│
├── logistic_regression_model.pkl
├── logistic_regression_threshold.pkl
├── improved_model_comparison_results.csv
├── independent_validation_results.csv
├── model_comparison_results.csv
├── readmission_threshold.pkl
├── validation_roc_curve.png
│
├── .gitignore
└── README.md
Note: The large hospital_readmission_model.pkl file is excluded from GitHub because it exceeds GitHub's 100 MB file limit. It remains on the local computer.

9. Installation and Setup
Step 1: Clone the repository
git clone https://github.com/aishik-messi/Hospital_Readmission_prediction.git
Step 2: Open the project folder
cd Hospital_Readmission_prediction
Step 3: Install the required libraries
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter joblib
Step 4: Run the Streamlit application
streamlit run app.py
The application will open in your browser:

http://localhost:8501
10. Limitations
The model uses historical hospital data from 1999–2008.
The evaluation uses an internal patient-level holdout test set.
External validation has not been performed.
The model's prediction score is not a clinically calibrated probability.
The application is developed for academic and educational purposes only.
It should not be used for real medical decision-making.
11. Future Improvements
Improve prediction performance.
Use more recent hospital datasets.
Perform external validation.
Improve the user interface.
Explore additional machine learning algorithms.
Add more patient-related features.
12. Author
Aishik Mondal

Brainware University

GitHub: aishik-messi

13. Acknowledgement
Dataset Source: UCI Machine Learning Repository

This project was developed for academic learning and demonstration purposes.
