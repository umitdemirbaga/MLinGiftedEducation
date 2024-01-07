from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# loading your preprocessed data
df = pd.read_csv("moralSensitivity.csv")
X = df.drop("moralSensitivity", axis=1)
y = df["moralSensitivity"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Regression": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

# Create an empty dataframe to store results
results_df = pd.DataFrame()

# Calculate mutual information for each class within the target variable
for name, classifier in classifiers.items():
    
    # Calculate mutual information for each class
    for class_value in sorted(y.unique()):
        y_binary = (y_train == class_value).astype(int)
        classifier.fit(X_train, y_binary)
        
        feature_importance = mutual_info_classif(X_train, y_binary)
        result = pd.DataFrame(data=feature_importance, index=X_train.columns, columns=[f'{name}_MI_{class_value}'])
        result = result.sort_values(by=f'{name}_MI_{class_value}', ascending=False)
        
        results_df = pd.concat([results_df, result], axis=1)

# Access results like this:
# results_df["Random Forest_MI_2"] for Random Forest MI for moralSensitivity 2
# results_df["SVM_MI_3"] for SVM MI for moralSensitivity 3
# and so on



# Mutual Information VIsualisation

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="whitegrid")

# Define a mapping of original column names to modified labels
column_name_mapping = {
    "Random Forest_MI_2": "A few",
    "Random Forest_MI_3": "Some",
    "Random Forest_MI_4": "Most",
    "Random Forest_MI_5": "All",
    
    "SVM_MI_2": "A few",
    "SVM_MI_3": "Some",
    "SVM_MI_4": "Most",
    "SVM_MI_5": "All",
    
    "Logistic Regression_MI_2": "A few",
    "Logistic Regression_MI_3": "Some",
    "Logistic Regression_MI_4": "Most",
    "Logistic Regression_MI_5": "All",
    
    "Decision Tree_MI_2": "A few",
    "Decision Tree_MI_3": "Some",
    "Decision Tree_MI_4": "Most",
    "Decision Tree_MI_5": "All",
    
    "XGBoost_MI_2": "A few",
    "XGBoost_MI_3": "Some",
    "XGBoost_MI_4": "Most",
    "XGBoost_MI_5": "All",
}

# Plot bar plots for each model
for model_name in classifiers.keys():
    model_columns = [col for col in results_df.columns if model_name in col]
    model_df = results_df[model_columns]

    plt.figure(figsize=(6, 3))
    for column in model_df.columns:
        # Use the modified label from the mapping
        modified_label = column_name_mapping.get(column, column)
        plt.bar(model_df.index, model_df[column], label=modified_label)

    plt.title(f"Mutual Information Scores for {model_name}")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information Score")
    plt.legend(title=f"Moral sensitivity classes", fontsize=11, bbox_to_anchor=(1.02, 1.0), loc='upper left')
    plt.xticks(rotation=45, ha="right")
    
    # Save the figure as PNG with 300 dpi
    plt.savefig(f"{model_name}_mutual_information.png", dpi=300, bbox_inches="tight")
    
    plt.show()
