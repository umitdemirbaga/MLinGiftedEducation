import pandas as pd
from sklearn.tree import DecisionTreeRegressor 

df = pd.read_csv("moralSensitivity.csv")

X = data.drop(columns=["moralSensitivity"])
y = data["moralSensitivity"]

# Create a decision tree model
model = DecisionTreeRegressor()  # or DecisionTreeClassifier for classification

# Fit the model to the data
model.fit(X, y)

# Get the feature importances
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

# Sort the feature importances
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importance)
