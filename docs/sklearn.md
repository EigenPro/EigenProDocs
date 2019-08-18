# Using EigenPro2 with Sklearn

## Setup
If you do not already have scikit-learn, you must first install it. Instructions are on [this page](https://scikit-learn.org/stable/install.html)

## Example Usage in Python
After importing sklearn, we can easily fit to some example data. First, we should do the necessary imports.


```python
from sklearn.fast_kernel import FKC_EigenPro
import numpy as np
```

!> Here we are running classification, you can also import FKR_EigenPro to run regression.

Next, we need to load the data. Here we choose 4000 samples with 20 features each, and we want to classify them as 0, 1, or 2.

```python
n_samples, n_features, n_targets = 4000, 20, 3
rng = np.random.RandomState(1)
x_train = rng.randn(n_samples, n_features)
y_train = rng.randint(n_targets, size=n_samples)
```

Finally, we are done with setup and can create the classifier. We choose a low subsample size since there isn't much data, and use a Gaussian kernel (default) with bandwidth 1.

```python
classifier = FKC_EigenPro(n_epoch=3, bandwidth=1, subsample_size=50)
```

We use the fit function to fit the data with the classifier. Then we use the predict function to predict new labels.

```python
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_train)
```
We can check what percentage of labels are correct, and see that our error is .775%.

```python
loss = np.mean(y_train != y_pred)
print(loss)
```
