import seaborn as sns
import matplotlib.pyplot as plt

def segmentation_corr_plt(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Customer Characteristic Correlation Heatmap')
    plt.savefig('../../image/characteristic_corr.png', bbox_inches='tight')
