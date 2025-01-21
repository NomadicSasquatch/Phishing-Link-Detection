# Machine Learning-Based Phishing Link Detection System

This project consists of a Machine Learning-Based Phishing Link Detection System that leverages various classification algorithms to accurately detect phishing websites. By utilizing techniques such as feature importance analysis, data preprocessing, SMOTE (for handling class imbalances), and hyperparameter tuning (GridSearchCV), this system demonstrates strong performance in classifying URLs (or feature sets) as phishing or non-phishing.

## Table of Contents
1. Key Features
2. Technologies Used
3. Prerequisites
4. Installation and Running
##

### 1. Key Features
- Phishing Detection Using Machine Learning
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Model Training
- SMOTE for Class Imbalance
- Pipeline Automation
- Model Evaluation
- Hyperparameter Tuning
- Visualization of Results

### 2. Technologies Used
- Python 3.7+

### 3. Prerequisites
- NumPy
- Pandas
- Seaborn
- Matplotlib
- scikit-learn
- imbalanced-learn (for SMOTE)
- Jupyter Notebook (optional, for interactive EDA and development)

### 4. Installation and Running
1. Clone the repository
```
git clone https://github.com/NomadicSasquatch/Phishing-Link-Detection.git
cd Phishing-Link-Detection
```
2. Create and activate a virtual environemnt (recommended)
- Recommended step to ensure that there is no conflict with libraries installed for other projects
```
python -m venv venv
venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the Project
```
python main.py
```
5. Before closing your IDE, run
```
deactivate
```
to deactivate the virtual environment


