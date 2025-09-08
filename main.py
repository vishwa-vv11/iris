# ===== IRIS FLOWER CLASSIFICATION PROJECT =====

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset using Copy Path
# 👉 Right-click your iris.csv file -> Copy as path -> paste it below inside r"..."
data = pd.read_csv(r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\IRIS\iris.csv")

print("First 5 rows of dataset:")
print(data.head())

# 2. Separate features (X) and target (y)
X = data.drop("species", axis=1)   # features (all columns except species)
y = data["species"]                # target (species column)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model Training (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=data['sepal_length'], y=data['sepal_width'], hue=data['species'], palette="Set2")
plt.title("Iris Sepal Length vs Width")
plt.show()
