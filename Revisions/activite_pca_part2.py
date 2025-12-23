import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(data):
    plt.figure(figsize=(12,8))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=True, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')


### Step 0: OK, let's get some real data!
#TODO: Load the heart dataset with Pandas

# Your code here

# Let's visualize the correlation matrix
plot_correlation_matrix(data)




### Step 1: Compute the covariance matrix!

# Your code here






### Step 2: Compute the new PCA axes and sort them in decrease variance order!
new_variances, new_features = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(new_variances)[::-1]

sorted_eigenvalues = new_variances[sorted_indices]
sorted_eigenvectors = new_features[:, sorted_indices]




### Step 3: Explained variance
#TODO: Compute the percentage of the total variance
total_variance = new_variances.sum()

# Your code here

plt.figure(figsize=(12,8))
plt.plot(variance_percent, '.-', markersize=10)
plt.title('Variance percentage of PCA axes')
plt.xlabel('PCA axis')
plt.ylabel('Variance (%)')
plt.grid()




### Step 4: Feature selection
#TODO: Compute how many PCA axes do you need to keep in order to have > 85% of the total variance ?
#TODO: And for > 95 % of the total variance ?

# Your code here



#TODO: Call apply_pca with your chosen number of axes
#TODO: Show the correlation matrix of the transformed data, what do you observe ?

def apply_pca(data, num_axes):
    return data.dot(sorted_eigenvectors[:,:num_axes])

# Your code here


