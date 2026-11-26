# Kestrel Home – AI-Powered Service Request Routing

## Project Overview

This project uses Machine Learning and Natural Language Processing to classify incoming home service requests and route them to the appropriate service team.

The goal is to improve initial request routing, reduce unnecessary transfers between teams, and support faster customer service.

## Problem Statement

Customer service requests can be assigned to the wrong team, resulting in unnecessary transfers, additional customer contacts, and increased operational costs.

This project builds a machine learning model to predict the correct initial routing team based on the customer's request details.

## Teams Supported

The model predicts one of seven teams:

- Repairs
- Billing
- Filters & Consumables
- Installs & Demo
- Product Advice
- Returns & Replacement
- Warranty Claims

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (TF-IDF)
- Linear Support Vector Machine (Linear SVM)
- Jupyter Notebook
- Gradio
- Joblib

## Project Workflow

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Text cleaning and feature engineering
4. Model training and comparison
5. Hyperparameter tuning
6. Model evaluation
7. Confidence-based human review
8. Model saving and application development

## Model Performance

The tuned Linear SVM model achieved the following results on the chronological validation dataset:

| Metric | Score |
|---|---:|
| Accuracy | 95.29% |
| Macro F1-score | 95.12% |

The model was trained on 10,822 labelled service requests and evaluated on 2,165 validation records.

## Intelligent Routing

The application uses a confidence-margin threshold to identify uncertain predictions.

- Automatic Routing: Higher-confidence predictions.
- Human Review Required: Predictions with a confidence margin below 0.20.

The confidence margin is the difference between the highest and second-highest decision scores. It is not a calibrated probability.

## Application

A Gradio application allows users to enter service request details and receive a predicted team and routing recommendation.

## Project Structure

```text
kestrel-home-service-routing-ml/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── kestrel_routing.ipynb
└── kestrel_routing_model.pkl
```

## How to Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Key Outcomes

- Built a seven-class service request classification model.
- Achieved 95.29% validation accuracy.
- Implemented confidence-based human review.
- Generated automatic-routing and human-review queues.
- Saved the trained model for application use.

## Limitations

- The model predicts the initial routing team, not the final resolution team.
- Confidence scores are not calibrated probabilities.
- The application is not connected to a live customer service system.
- Production performance must be monitored using real operational feedback.

## Data Confidentiality

This project was developed using internal Kestrel Home operational data. The dataset, customer records, model artifacts, and derived operational files are confidential and must not be published or distributed publicly. This documentation does not authorize external sharing.

---

**Developed as a Machine Learning project using Python and Scikit-learn.**
