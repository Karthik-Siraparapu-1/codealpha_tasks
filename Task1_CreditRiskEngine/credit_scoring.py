import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")

df = pd.read_csv("data/loan_data.csv")

print("\nDataset Preview:")
print(df.head())

# Features and target
X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# Detect categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

# Evaluate models
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    results[name] = scores.mean()

# Save model comparison
comparison_df = pd.DataFrame(
    results.items(),
    columns=["Model", "Accuracy"]
)

comparison_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nModel Comparison:")
print(comparison_df)

# Select best model
best_model_name = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

# Final pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_model)
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
final_pipeline.fit(X_train, y_train)

# Predict
y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Accuracy: {round(accuracy * 100, 2)}%")
print(f"ROC-AUC Score: {round(roc_auc, 4)}")

report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)

# Save report
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Feature importance (if Random Forest)
if best_model_name == "Random Forest":
    importances = final_pipeline.named_steps["classifier"].feature_importances_

    encoded_columns = final_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    feature_importance = pd.DataFrame({
        "Feature": encoded_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(12, 7))
    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300)
    plt.close()

# Save model
joblib.dump(final_pipeline, "models/best_credit_model.pkl")

print("\nBest model saved successfully!")

# Real-world sample prediction
sample = pd.DataFrame([{
    "Age": 25,
    "Gender": "male",
    "Education": "Bachelor",
    "Person Income": 60000,
    "Employee Experience": 3,
    "Home Onwership": "RENT",
    "Loan Amount": 12000,
    "Loan Intent": "PERSONAL",
    "Loan interest Rate": 11.5,
    "Loan percentage": 0.25,
    "Credit History": 4,
    "Credit Score": 720,
    "Previous Loan": "No"
}])

probability = final_pipeline.predict_proba(sample)[0][1]
prediction = final_pipeline.predict(sample)[0]

print(f"\nApproval Probability: {round(probability * 100, 2)}%")

# Risk engine
if probability >= 0.80:
    risk = "Low Risk"
elif probability >= 0.60:
    risk = "Medium Risk"
elif probability >= 0.40:
    risk = "High Risk"
else:
    risk = "Very High Risk"

print(f"Risk Category: {risk}")

if probability >= 0.75:
    decision = "APPROVED"
elif probability >= 0.55:
    decision = "MANUAL REVIEW"
else:
    decision = "REJECTED"

print(f"Loan Decision: {decision}")


with open("outputs/applicant_report.txt", "w") as report:
    report.write("Applicant Loan Risk Report\n")
    report.write("===========================\n")
    report.write(f"Age: 25\n")
    report.write(f"Income: 60000\n")
    report.write(f"Loan Amount: 12000\n")
    report.write(f"Credit Score: 720\n")
    report.write(f"Approval Probability: {round(probability * 100, 2)}%\n")
    report.write(f"Risk Category: {risk}\n")
    report.write(f"Final Decision: {decision}\n")

print("\nApplicant report generated successfully!")