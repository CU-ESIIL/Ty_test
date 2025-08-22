# Data Analysis

This page describes the analytical methods applied to the project's datasets. It provides sample code and explanations of statistical or visualization techniques.

```python
# Example analysis snippet
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Sample Plot')
plt.show()
```


## Random Forest Example

Random forests are ensemble-based machine learning models that combine multiple decision trees to improve predictive performance.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

This example demonstrates how to train and evaluate a simple random forest classifier using scikit-learn.

