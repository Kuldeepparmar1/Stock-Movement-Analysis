import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

input_file = "Stock Data"  # Replace with your file
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df['Tomorrow_Close'] = df['Close'].shift(-1)
df['Movement'] = np.where(df['Tomorrow_Close'] > df['Close'], 1, 0)
df = df.dropna(subset=['Tomorrow_Close'])
features = ['Open', 'High', 'Close', 'Volume']
X = df[features]
y = df['Movement']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance)
import matplotlib.pyplot as plt

# Adding predictions to the original test set
X_test['Prediction'] = y_pred
X_test['Actual'] = y_test.values

X_test['Prediction'] = y_pred
X_test['Actual'] = y_test.values
# Plotting predicted vs. actual movements
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, X_test['Prediction'], label='Predicted Movement', color='blue', marker='o')
plt.plot(X_test.index, X_test['Actual'], label='Actual Movement', color='orange', linestyle='dashed', marker='x')
plt.xlabel('Index')
plt.ylabel('Movement')
plt.title('Predicted vs Actual Stock Movements')
plt.legend()
plt.show()
