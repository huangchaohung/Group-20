import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import pickle

def real_time_segmentation_evaluation(df):
    gbc_df = df.copy()
    X = gbc_df.drop(columns='cluster')
    y = gbc_df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)

    importances = gbc.feature_importances_
    encoded_feature_names = np.hstack([
        ['TransactionFrequency', 'Recency', 'AverageTransactionAmount', 'housing', 'loan', 'cd_account', 'securities'],
    ])

    feature_importance_df = pd.DataFrame({
        'feature': encoded_feature_names,
        'importance': importances
    })

    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Plot the top 5 most important features using a bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(5))
    plt.title('Top 5 Important Features')
    plt.savefig('../../image/real_time_segmentation_importance.png', bbox_inches='tight')
    plt.close()

    y_pred = gbc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('../../image/real_time_segmentation_confusion.png', bbox_inches='tight')

    with open('../model/real_time_segmentation.pkl', 'wb') as file:
        pickle.dump(gbc, file)
