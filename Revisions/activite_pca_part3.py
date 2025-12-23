import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_correlation_matrix(data):
    plt.figure(figsize=(12,8))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=True, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')


### Step 0: OK, let's get some real data!
#TODO: Load the heart dataset with Pandas


### Step 1: PCA with sklearn
# See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#TODO: Apply PCA to the dataset


# Look at the explained variance ratio, is it the same as you computed in part 2 ?


### Step 3: Visualize data, what do you observe ?
pca_data = pd.DataFrame(data = pca_result, columns = ['PCA 1', 'PCA 2'])
plt.scatter(pca_data['PCA 1'], pca_data['PCA 2'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()