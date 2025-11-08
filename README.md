# Credit-card-fraud-detection
# Project Overview

Credit card fraud has become one of the most critical challenges faced by banks and financial institutions today. This project aims to detect fraudulent credit card transactions using an Artificial Neural Network (ANN) model.
The model is trained on the Kaggle Credit Card Fraud Detection dataset, which contains anonymized transaction data labeled as fraudulent or legitimate.

Our objective is to build a robust classification model that minimizes false negatives (i.e., frauds classified as legitimate) while maintaining a high overall accuracy.

# Objectives

Analyze and preprocess the dataset to handle missing values and class imbalance.

Train an ANN to classify transactions as fraudulent or non-fraudulent.

Optimize model performance using feature scaling, dropout, and tuning of hyperparameters.

Evaluate results using metrics like Precision, Recall, F1-Score, and ROC-AUC.

# Dataset

Source: Kaggle Credit Card Fraud Detection Dataset

Records: 284,807 transactions

Fraudulent Cases: 492 (0.17%)

Features:

V1 to V28 — anonymized numerical features (PCA-transformed)

Amount — transaction amount

Time — seconds elapsed between transactions

Class — target variable (1 = Fraud, 0 = Legitimate)

# Workflow

- Data Preprocessing

Load dataset and handle missing data (if any).

Perform data normalization and train-test split.

Address class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

- Model Building

Create an Artificial Neural Network (ANN) using TensorFlow / Keras.

Layers:

Input layer with 30 neurons

Hidden layers with ReLU activation and Dropout for regularization

Output layer with Sigmoid activation (binary classification)

- Model Training

Optimizer: Adam

Loss: Binary Crossentropy

Epochs: 50–100 (tuned for performance)

Batch size: 64

- Model Evaluation

Evaluate using metrics:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC Curve

# Results
Metric	Value
Accuracy	~99.8%
Precision	~92%
Recall	~85%
F1-Score	~88%
ROC-AUC	~0.99

(Note: Actual values may vary depending on training parameters.)

# Technologies Used

Python 3.10+

TensorFlow / Keras

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn (for SMOTE)

# How to Run

Clone the repository:

[git clone https://github.com/<your-username>/credit-card-fraud-detection.git]


Navigate to the project folder:

[cd credit-card-fraud-detection]


Install dependencies:

[pip install -r requirements.txt]


Run the Jupyter notebook:

[jupyter notebook credit_card_fraud.ipynb]

# Future Enhancements

Integrate deep learning models (LSTM / Autoencoders) for anomaly detection.

Build a real-time fraud detection API using Flask or FastAPI.

Deploy the model on AWS / Streamlit for live inference.

# Author

Aayushi soni.
📧 [aayushisoni6295@gmail.com
]
