import numpy as np
import numpy.random as rng

# Define some random Gaussian test data
x = rng.normal(5., 2., size=10)
y = rng.normal(3., 1.5, size=10)
size = 10

### Step 1: Mean 
#TODO: Implement your own mean function!
def my_mean(x):
    
    total = 0
    for value in x:
        total += value
    mean = total / len(x)
    return mean    
    
print(np.mean(x), my_mean(x))



### Step 2: Variance
#TODO: Implement your own variance function!
def my_var(x):
    
    mean = my_mean(x)
    
    # Then calculate sum of squared differences from mean
    sum_squared_diff = 0
    for value in x:
        diff = value - mean
        sum_squared_diff += diff ** 2
    
    # Variance is average of squared differences
    variance = sum_squared_diff / len(x)
    return variance
    
    return

print(np.var(x), my_var(x))



### Step 3: Covariance
#TODO: Implement your own covariance function!

def my_cov(x, y):
    mean_x = my_mean(x)
    mean_y = my_mean(y)

    total_product = 0
    count = 0

    for i in range(len(x)):
        dev_x = x[i] - mean_x
        dev_y = y[i] - mean_y
        product = dev_x * dev_y

        total_product = total_product + product
        count = count + 1

    covariance = total_product / count
    return covariance
print(np.cov(x,y), my_cov(x,y))
