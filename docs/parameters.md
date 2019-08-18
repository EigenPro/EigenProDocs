# Parameters to the model
Each implementation of the EigenPro model shares many common parameters. This page will give an overview of what each parameter means. More information can be found on the documentation page dedicated specifically to a given implementation.


## Initialization parameters

### batch_size (int)
The mini-batch size to use for gradient descent. If not provided, the model will automatically estimate the best batch size

### n_epoch (int)
The total number of passes to be done over the training data.

### n_components : (int)
The maximum number of eigendirections used in modifying the kernel operator. The convergence rate speedup over normal gradient descent is approximately the largest eigenvalue over the n_componentth eigenvalue.

!> n_components must be smaller than subsample_size. Some versions will automatically set it to subsample_size-1 if it is too large.

### subsample_size : (int)
The number of subsamples used for estimating the largest n_component eigenvalues and eigenvectors.

?> Larger values lead to better estimation of more eigenvectors, but will add significant overhead.

### mem_gb : (int)
The physical device memory of your device in GB. Setting this higher may increase the batch size, allowing for more parallelization, but setting it too high may lead to a memory error.

### kernel : (string)
The kernel mapping used by EigenPro. All methods support the kernels "gaussian", "laplacian", and "cauchy".

### bandwidth : (float)
The scale parameter for the kernel

!> The sklearn implementation uses gamma rather than bandwidth. You can calculate between the values using the formula gamma=.5/(bandwidth^2)


## Fit
### Parameters
#### X (float, matrix)
The features of the training data on which you want to perform regression. It will be a float matrix of shape [number of training samples, number of features].

#### Y (float, matrix)
The target values of the training data on which you want to perform regression. It will be a float matrix of shape [number of training samples, number of outputs to predict]

## Predict
### Parameters
#### X (float, matrix)
The features of the training data which you want to predict. It will be a float matrix of shape [number of training samples, number of features].
### Returns
#### Y (float, matrix)
The predicted values for each of the samples. It will be a float matrix of shape [number of training samples, number of outputs to predict]
