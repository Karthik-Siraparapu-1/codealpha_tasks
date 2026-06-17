import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Loading Heart Disease Dataset...")

df = pd.read_csv("data/heart_data.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "SVM": SVC(probability=True)
}

results = {}

print("\nTraining models...\n")

for name, model in models.items():
    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    ).mean()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "roc": roc,
        "cv": cv_score
    }

    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Cross-validation:", round(cv_score * 100, 2), "%")
    print("ROC-AUC:", round(roc, 4))
    print()

metrics_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[x]["accuracy"] for x in results],
    "ROC_AUC": [results[x]["roc"] for x in results],
    "Cross_Validation": [results[x]["cv"] for x in results]
})

metrics_df.to_csv(
    "outputs/model_metrics.csv",
    index=False
)

best_model_name = max(
    results,
    key=lambda x: results[x]["roc"]
)

best_model = results[best_model_name]["model"]

print("\nBest Model:", best_model_name)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(roc, 4))

report_text = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report_text)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report_text)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[x]["accuracy"] for x in results],
    "ROC-AUC": [results[x]["roc"] for x in results]
})

comparison_df.plot(
    x="Model",
    y=["Accuracy", "ROC-AUC"],
    kind="bar",
    figsize=(8, 5)
)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=300)
plt.close()

if hasattr(best_model, "feature_importances_"):
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300)
    plt.close()

joblib.dump(
    best_model,
    "models/disease_model.pkl"
)

test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred
test_predictions["Probability"] = y_prob

test_predictions["Risk Level"] = test_predictions["Probability"].apply(
    lambda x:
    "Critical" if x >= 0.80 else
    "High" if x >= 0.60 else
    "Moderate" if x >= 0.40 else
    "Low"
)

test_predictions.to_csv(
    "outputs/test_predictions.csv",
    index=False
)

risk_summary = test_predictions["Risk Level"].value_counts()

plt.figure(figsize=(8, 5))
risk_summary.plot(kind="bar")
plt.title("Risk Level Distribution")
plt.tight_layout()
plt.savefig("outputs/risk_distribution.png", dpi=300)
plt.close()

print("\nModel saved successfully!")
print("Predictions exported successfully!")
print("Reports generated successfully!")
print("Project completed.")