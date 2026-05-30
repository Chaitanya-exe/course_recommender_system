from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utility import load_training_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ohe_column_encoding(hot: list, df: pd.DataFrame):
    
    if hot is not None:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")

        for col in hot:
            temp = ohe.fit_transform(df[[col]])
            df = pd.concat([df, temp], axis=1).drop(columns=[col])
    
        return df
    
    return df
    

# Load the dataset
dataset = load_training_dataset()
features = [

    "similarity_score",
    "domain_match",
    "subject_required",
    "subject_overlap",
    "marks_required",
    "marks_margin",
    "skill_relevance_percentage",
    "career_align_percentage",
   

]
y = dataset.pop("label")
X = dataset[features].copy()

# encoding for categorical features
# X = ohe_column_encoding(hot=['preferred_domain', 'domain'], df=X)
# level_order = ['PreUG', 'UG', 'PG', 'PhD']
# ord = OrdinalEncoder(categories=[level_order])
# degree_levels = pd.Series(ord.fit_transform(dataset[['degree_level']]).flatten(), name='degree_levels')
# program_levels = pd.Series(ord.fit_transform(dataset[['program_level']]).flatten(), name='program_levels')
# X = pd.concat([X, degree_levels, program_levels], axis=1).drop(columns=['degree_level', 'program_level'])


# Performing train, test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# Initialising and training model
# model = RandomForestClassifier(random_state=42,n_estimators=200)

model = LogisticRegression(random_state=42, max_iter=100)
model.fit(X_train, y_train)

# Predections on test data
predictions = model.predict(X_test)

# Metrics

print("\nAccuracy: \n")
print(accuracy_score(y_test, predictions))

print("\nF1 scores:\n")
print(f1_score(y_test, predictions, average="macro"))

print("\nClassification Report: \n")
print(classification_report(y_test, predictions))

# Confustion Matrix

cm = confusion_matrix(y_test, predictions)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()