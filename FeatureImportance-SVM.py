import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

df = pd.read_csv("moralSensitivity.csv")

X = data.drop(columns=["moralSensitivity"])
y = data["moralSensitivity"]

# Create an SVM classifier
model = SVC(kernel='linear')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an RFE model with SVM
selector = RFE(estimator=model, n_features_to_select=1, step=1)

# Fit the RFE model to the training data
selector = selector.fit(X_train, y_train)

# Get the feature ranking from RFE
feature_ranking = selector.ranking_

# Create a DataFrame with the feature names and their ranking
feature_importance = pd.DataFrame({'Feature': X.columns, 'Ranking': feature_ranking})

# Sort the feature importance
feature_importance = feature_importance.sort_values(by='Ranking')

# Display the feature importance
print(feature_importance)