import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("moralSensitivity.csv")

X = data.drop(columns=["moralSensitivity"])
y = data["moralSensitivity"]

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Get the feature coefficients
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})

# Sort the coefficients by absolute value to see feature importance
feature_importance['Importance'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance)
