import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt

# Set the directory for saving plots
plots_dir = "image"
os.makedirs(plots_dir, exist_ok=True)

# Load data
data = pd.read_csv("../data/Churn_Modelling.csv")
data = data.dropna()

# Preprocess the data
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)
X = data.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# SHAP explanation and feature importance plot
def generate_shap_plots(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature importance bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    feature_importance_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    summary_plot_path = os.path.join(plots_dir, "shap_summary.png")
    plt.savefig(summary_plot_path)
    plt.close()

    return feature_importance_path, summary_plot_path

# Retention strategy visualization
def generate_retention_strategies_plot(X_test_df):
    plt.figure()
    X_test_df['Churn_Risk_Score'].hist(bins=10)
    churn_risk_hist_path = os.path.join(plots_dir, "churn_risk_histogram.png")
    plt.title("Churn Risk Score Distribution")
    plt.xlabel("Churn Risk Score")
    plt.ylabel("Number of Customers")
    plt.savefig(churn_risk_hist_path)
    plt.close()

    return churn_risk_hist_path

# Run functions and save plot paths
shap_feature_importance_path, shap_summary_plot_path = generate_shap_plots(model, X_test)
print(f"Feature Importance Plot saved at: {shap_feature_importance_path}")
print(f"SHAP Summary Plot saved at: {shap_summary_plot_path}")


# Early Warning System with Retention Strategy
def early_warning_and_retention(X_test, y_pred_proba):
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_test_df['Churn_Risk_Score'] = y_pred_proba

    # Retention strategies based on risk factors
    def recommend_retention_strategy(row):
        if row['Churn_Risk_Score'] > 0.7:
            if row['IsActiveMember'] == 0:
                return "Offer incentives to increase engagement."
            elif row['NumOfProducts'] < 2:
                return "Offer a discount on additional products."
            elif row['Balance'] < 50000:
                return "Provide financial counseling or bonus interest rate."
        return "Standard follow-up and customer satisfaction survey."

    X_test_df['Retention_Strategy'] = X_test_df.apply(recommend_retention_strategy, axis=1)
    return X_test_df

# Create data with retention strategies
X_test_df = early_warning_and_retention(X_test, y_pred_proba)

# Filter high-risk customers
high_risk_customers = X_test_df[X_test_df['Churn_Risk_Score'] > 0.7]

# Generate churn risk distribution plot
churn_risk_plot_path = generate_retention_strategies_plot(X_test_df)
print(f"Churn Risk Distribution Plot saved at: {churn_risk_plot_path}")

# Define a function to save the DataFrame as a PNG image
def save_table_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.3))  # Adjust height based on the number of rows
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale for better readability
    # Add title
    plt.title(title, fontsize=14, weight='bold', pad=20)

    # Set a specific column width for the first column (0-indexed)
    #strat_column_width = 0.5  # Adjust the width as necessary
    for i in range(len(df) + 1):  # +1 to include the header 
        table[(i, 5)].set_width(0.5)

    
    # Save the table as a PNG
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table saved as {filepath}")

# Save high-risk customers table
save_table_as_image(
    high_risk_customers[['RowNumber','Churn_Risk_Score', 'IsActiveMember', 'NumOfProducts', 'Balance', 'Retention_Strategy']],
    "predict_customer_churn_high_risk_customers.png",
    "High-Risk Customers and Retention Strategies"
)