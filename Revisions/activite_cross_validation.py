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
    """
    
    """Yields:
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

    
    """"Je délimite la taille de mon set en entier"""
    nb_lignes = data.shape[0]
    
    """Ma liste de numéro"""
    indices = np.arange(nb_lignes) """n_samples = len(data)"""
    
    # je mélange si on me demande
    if shuffle:
        if random_state:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # maintenant pour chaque fold
    for k in range(n_splits):
        
        # je calcule où ça commence et où ça finit
        taille = nb_lignes // n_splits
        debut = k * taille
        fin = debut + taille
        
        # dernier fold prend tout ce qui reste
        if k == n_splits - 1:
            fin = nb_lignes
        
        # les indices pour validation
        valid_idx = indices[debut:fin]
        
        # les indices pour train (tout sauf valid)
        train_idx = np.concatenate([indices[:debut], indices[fin:]])
        
        # je récupère les données
        train_data = data[train_idx]
        valid_data = data[valid_idx]
        
        yield train_data, valid_data
            
    """for k in range(n_splits):
        
        # Your code here
        
        yield train_data, valid_data"""
    # Your code here
    n_samples = data.shape[0]
    fold_size = n_samples // n_splits # Dans notre cas, fold_size = 20

    for k in range(n_splits):
        
        valid_data = data[k * fold_size : (k+1) * fold_size]
        
        # Valid set:
        # k = 0: data[0 * fold_size : 1 * fold_size] : taille de la tranche: fold_size
        # k = 1: data[1 * fold_size : 2 * fold_size] : taille de la tranche: fold_size
        # k = 2: data[2 * fold_size : 3 * fold_size] : taille de la tranche: fold_size
        # k = 3: data[3 * fold_size : 4 * fold_size] : taille de la tranche: fold_size
        # k = 4: data[4 * fold_size : 5 * fold_size] : taille de la tranche: fold_size
        
        # Train set:
        # k = 0: data[fold_size:]
        
        yield train_data, valid_data

# Visualize your results
n_splits = 5
fig, axes = plt.subplots(n_splits, 2)

for i, (train, valid) in enumerate(cross_validation(data_matrix, n_splits, shuffle=False)):
    print(i, train.shape, valid.shape)
    axes[i,0].imshow(train,vmin=0,vmax=n_colors-1)
    axes[i,1].imshow(valid,vmin=0,vmax=n_colors-1)
    
    
    
    
    
    
    
    
    
    




    
