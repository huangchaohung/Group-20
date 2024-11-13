import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def apply_SMOTE(df, categorical_cols):
    rf_df = df.copy()
    rf_df = pd.get_dummies(rf_df, columns=categorical_cols)

    X = rf_df.drop(columns='y')
    y = rf_df['y'].map({'no': 0, 'yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X, y, X_train_resampled, X_test, y_train_resampled, y_test