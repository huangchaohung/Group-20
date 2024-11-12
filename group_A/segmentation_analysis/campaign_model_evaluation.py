import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def evaluate_graph(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for '+name)
    plt.savefig('../../image/'+name+'_confusion.png', bbox_inches='tight')
    plt.close()

    # Probability predictions for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f'ROC-AUC Score: {roc_auc:.2f}')

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve '+name)
    plt.legend()
    plt.savefig('../../image/'+'name'+'_ROC.png', bbox_inches='tight')
    plt.close()

def campaign_model_evaluation(X, X_train_resampled, X_test, y_train_resampled, y_test):
    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    evaluate_graph(rf, X_test, y_test, 'Random Forest')

    # Scalar transformation
    scaler = StandardScaler()
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_resampled_scaled, y_train_resampled)
    evaluate_graph(log_reg, X_test_scaled, y_test, 'Logistic Regression')

    # SVM
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train_resampled_scaled, y_train_resampled)
    evaluate_graph(svm, X_test_scaled, y_test, 'SVM')

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_resampled_scaled, y_train_resampled)
    evaluate_graph(knn, X_test_scaled, y_test, 'KNN')

    # Gradient Boosting
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train_resampled_scaled, y_train_resampled)
    evaluate_graph(gbc, X_test_scaled, y_test, 'Gradient Boosting')

    gbc_importance = gbc.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': gbc_importance
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Important Features')
    plt.savefig('../../image/campaign_feature_importance.png', bbox_inches='tight')
    plt.close()

def campaign_feature_visualization(df):
    plt.figure()
    sns.boxplot(x='y', y='duration', data=df)
    plt.title('duration Distribution')
    plt.savefig('../../image/duration_against_subscription.png', bbox_inches='tight')
    plt.close()

    poutcome_counts = df.groupby(['poutcome', 'y']).size().unstack(fill_value=0)
    poutcome_counts['success_rate'] = poutcome_counts['yes'] / (poutcome_counts['yes'] + poutcome_counts['no'])
    fig, ax = plt.subplots(figsize=(8, 4)) 
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=poutcome_counts.values,
                    colLabels=poutcome_counts.columns,
                    rowLabels=poutcome_counts.index,
                    cellLoc='center',
                    loc='center')
    
    plt.savefig('../../image/poutcome_summary_table.png', bbox_inches='tight')
    plt.close()