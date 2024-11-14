import matplotlib.pyplot as plt
import seaborn as sns

def numerical_analysis(df):
    df.hist(bins=100, figsize=(14, 10), color='blue')
    plt.suptitle('Distribution of Numeric Features')
    plt.savefig('../../image/num_distribution.png', bbox_inches='tight')

    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('../../image/num_corr.png', bbox_inches='tight')
