# Setting up EigenPro2 with MATLAB
### Preprocessed MNIST data
The data is preprocessed by mapping the feature into [0, 1].
```
cd data
unzip mnist.zip
```

### Setting CPU/GPU flag
User can change the first line in 'run_expr.m' to switch between GPU and CPU. When set
```
use_gpu = true;
```
the script will use MATLAB gpuarray to store data and weights.

### Selecting the kernel
User can select the kernel function by change the second line in 'run_expr.m'.
```
ktype = 'Gaussian';
```
Current options involve 'Gaussian', 'Laplace', and 'Cauchy'.

### Running experiments (in MATLAB console)
The experiments will compare Pegasos, Kernel EigenPro, Random Fourier Feature with linear SGD, and Random Fourier Feature with EigenPro on MNIST.
```
run('run_expr.m')
```

## Training with SGD/EigenPro iteration
### SGD iteration
The following function call will update 'initial_weights' to 'new_weights'
using 'n_epoch' SGD epochs. 
```
[new_weights, time] = ...
    sgd_iterate(random_seed, train_x, train_y, initial_weights,
                phi, eta, batch_size, n_epoch, method_name);
```
Note that 'phi' is a given feature function that maps the original data features
to kernel features or random Fourier features.
Besides, there are two methods available now: 'Pegasos' and 'Linear'.

### EigenPro iteration
EigenPro iteration has interface similar to that of SGD iteration.
```
[new_weights, time] = ...
    eigenpro_iterate(random_seed, train_x, train_y, initial_weights, phi,
                     eta, batch_size, n_epoch, method_name, k, M, tau);
```
Noticeably, it has extra parameters specified for EigenPro.
Here we will compute the top-'k' eigensystem of
a subsample covariance involving 'M' data samples
to form the EigenPro preconditioner.
The available methods are 'Kernel EigenPro' and 'EigenPro'.
