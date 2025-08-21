import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load the dataset
df = pd.read_csv("customer-Churn.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# 2️⃣ Basic cleaning of data
# Remove customerID since it's not useful for prediction
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (some may have spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values fill if absent 
df.fillna(0, inplace=True)

# 3️⃣ Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 4️⃣ Split into features (X) and target (y)
X = df_encoded.drop("Churn_Yes", axis=1)
y = df_encoded["Churn_Yes"]

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7️⃣ Predictions (predict)
y_pred = model.predict(X_test)

# 8️⃣ Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9️⃣ Simple EDA visualization
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn Status")
plt.show()
