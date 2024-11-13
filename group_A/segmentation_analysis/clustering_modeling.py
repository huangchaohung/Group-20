from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

def preprocess_seg(df):
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['TransactionFrequency', 'Recency', 'AverageTransactionAmount', 'housing', 'loan', 'cd_account', 'securities']),  # Scale specified numerical features
        ('cat', OneHotEncoder(), [])
    ])
    processed_df = preprocessor.fit_transform(df)
    return processed_df

def elbow(processed_df, n):
    inertia = []
    K = range(2, n)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 
        kmeans.fit(processed_df) 
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.savefig('../../image/elbow'+str(n)+'.png', bbox_inches='tight')

def kmeans_pca(processed_df, n):
    # Optimal number of clusters we choose
    optimal_clusters = n

    # Initialize the KMeans model with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')

    # Apply KMeans to the processed data and assign the predicted clusters to each data point
    clusters = kmeans.fit_predict(processed_df)

    # Add the cluster labels as a new column in the original DataFrame
    clusters = clusters

    # Initialize PCA to reduce the data to 2 principal components for visualization
    pca = PCA(n_components=2)

    # Apply PCA to the processed df data and reduce it to 2 dimensions
    X_pca = pca.fit_transform(processed_df)

    # Create a scatter plot to visualize the customer segments
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1')
    plt.title('Customer Segments based on PCA')
    plt.savefig('../../image/Kmeans_PCA.png', bbox_inches='tight')

    silhouette_avg = silhouette_score(processed_df, clusters)
    return clusters, silhouette_avg

def kmeans_pattern(processed_df):
    score = -1
    n_comp = -1
    ns_keams = []
    cluster_range = range(2, 21)
    for i in range(2,21):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        temp = kmeans.fit_predict(processed_df)
        silhouette_avg = silhouette_score(processed_df, temp)
        if score < silhouette_avg:
            score = silhouette_avg
            n_comp = i
        ns_keams.append(silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, ns_keams, marker='o', linestyle='-', color='b')
    plt.xticks(cluster_range)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Pattern in Number of Clusters')
    plt.grid(True)
    plt.savefig('../../image/Kmeans_Silhouette_Score_Pattern.png', bbox_inches='tight')

def model_comparison(processed_df, score_kmeans):
    #GaussianMixture
    gmm = GaussianMixture(n_components=7, random_state=0)
    temp = gmm.fit_predict(processed_df)
    score_GMM = silhouette_score(processed_df, temp)

    # Mean_shift
    mean_shift = MeanShift()
    temp = mean_shift.fit_predict(processed_df)
    score_meanshift = silhouette_score(processed_df, temp)

    #DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=800)
    temp = dbscan.fit_predict(processed_df)
    score_DBSCAN = silhouette_score(processed_df, temp)

    model_names = ['KMeans', 'MeanShift', 'DBSCAN', 'GaussianMixture']
    scores = [score_kmeans, score_meanshift, score_DBSCAN, score_GMM]
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, scores, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Clustering Model')
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of Clustering Models')
    plt.ylim(0, 1)  # Silhouette scores range from -1 to 1, adjust the range as needed
    plt.grid(True, axis='y')
    plt.savefig('../../image/Cluster_Model_Comparison.png', bbox_inches='tight')
