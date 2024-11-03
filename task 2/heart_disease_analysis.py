import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data = pd.read_csv('heart.csv')

print("First few rows of the dataset:")
print(data.head())

print("\nDataFrame Info:")
print(data.info())

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f"\nLogistic Regression Accuracy: {log_reg_accuracy}")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

k_values = range(1, 21)
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_knn)
    knn_accuracies.append(accuracy)
    print(f"KNN Accuracy with K={k}: {accuracy}")

plt.figure(figsize=(10, 5))
plt.plot(k_values, knn_accuracies, marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.xticks(k_values)
plt.grid()
plt.show()

best_k = knn_accuracies.index(max(knn_accuracies)) + 1
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn_best = knn_best.predict(X_test_scaled)

knn_best_accuracy = accuracy_score(y_test, y_pred_knn_best)
print(f"\nKNN (K={best_k}) Accuracy: {knn_best_accuracy}")

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn_best))

importance = log_reg.coef_[0]
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='blue')
plt.xlabel('Importance')
plt.title('Feature Importance from Logistic Regression')
plt.show()

conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn_best)

print("\nConfusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)

print("\nConfusion Matrix for KNN:")
print(conf_matrix_knn)
