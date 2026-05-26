from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_regression
from utility import load_training_dataset
import matplotlib.pyplot as plt
import pandas as pd


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

    "career_align_percentage"

]
y = dataset.pop("label")
X = dataset[features].copy()

# Performing train, test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# Initialising and training model
model = RandomForestClassifier(random_state=42,n_estimators=200, max_depth=6)
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

# Metrics on features

importance = pd.DataFrame({
    "feature" : features,

    "score": model.feature_importances_ 
})

importance = importance.sort_values("score")

plt.barh(
    importance['feature'],
    importance['score'],
)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# Confustion Matrix

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()



