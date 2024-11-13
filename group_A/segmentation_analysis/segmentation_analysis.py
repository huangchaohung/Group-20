from data_prep import load_data
from data_prep import clean_data
from segment_corr import segmentation_corr_plt
from clustering_modeling import preprocess_seg
from clustering_modeling import elbow
from clustering_modeling import kmeans_pca
from clustering_modeling import kmeans_pattern
from clustering_modeling import model_comparison
from characteristic_visualization import visualization
from campaign_data_analysis import numerical_analysis
from balance import apply_SMOTE
from campaign_model_evaluation import campaign_model_evaluation
from campaign_model_evaluation import campaign_feature_visualization
from real_time_evalution import real_time_segmentation_evaluation

def customer_segmentation():
    # Load Data
    df_full = load_data(r"../../data/Combined_dataset.csv")

    # Define columns for segmentation analysis
    seg_cols = ['housing', 'loan', 'cd_account', 'securities', 'TransactionFrequency', 'Recency', 'AverageTransactionAmount']
    df = clean_data(df_full, seg_cols)

    # Compute and plot the correlation heatmap
    segmentation_corr_plt(df)

    processed_df = preprocess_seg(df)
    elbow(processed_df, 40)
    elbow(processed_df, 15)
    clusters, silhouette_avg = kmeans_pca(processed_df, 7)
    df['cluster'] = clusters
    df_full['cluster'] = clusters
    kmeans_pattern(processed_df)
    model_comparison(processed_df, silhouette_avg)
    visualization(df)
    real_time_segmentation_evaluation(df)

    categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'poutcome']
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'TransactionFrequency', 'Recency', 'AverageTransactionAmount', 'housing', 'loan', 'cd_account', 'securities']

    numerical_analysis(df_full[numerical_cols])
    numerical_cols.append('cluster')
    df_full = df_full.drop(columns=['month','day'])
    X, y, X_train_resampled, X_test, y_train_resampled, y_test = apply_SMOTE(df_full, categorical_cols)
    campaign_model_evaluation(X, X_train_resampled, X_test, y_train_resampled, y_test)
    campaign_feature_visualization(df_full)


if __name__ == "__main__":
    import os 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    customer_segmentation()