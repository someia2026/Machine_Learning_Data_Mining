import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Create a matrix with stripes of color so it's easy to visualize
n_samples, n_features = 100, 80
n_colors = 5
data_matrix = np.repeat(np.arange(n_colors),n_samples*n_features//n_colors).reshape(n_samples,n_features)

plt.imshow(data_matrix,vmin=0,vmax=n_colors-1)


# TODO: Implement the following cross-validation function!
def cross_validation(data, n_splits=5, shuffle=True, random_state=None):
    """
    Generator function that performs k-fold cross-validation.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Training data
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
    random_state : int, optional
        Random seed for reproducibility
    
    Yields:
    -------
    train_data : array
        Training data for this fold
    valid_data : array
        Test data for this fold

    Resources:
    ----------
    Python Generators: https://www.w3schools.com/python/python_generators.asp
    NumPy Indexing: https://numpy.org/doc/stable/user/basics.indexing.html
    NumPy Random: https://www.w3schools.com/python/numpy/numpy_random.asp
    """
    
    # Your code here
            
    for k in range(n_splits):
        
        # Your code here
        
        yield train_data, valid_data


# Visualize your results
n_splits = 5
fig, axes = plt.subplots(n_splits, 2)

for i, (train, valid) in enumerate(cross_validation(data_matrix, n_splits, shuffle=False)):
    print(i, train.shape, valid.shape)
    axes[i,0].imshow(train,vmin=0,vmax=n_colors-1)
    axes[i,1].imshow(valid,vmin=0,vmax=n_colors-1)
    
    
    
    
    
    
    
    
    
    




    
