# SigClust

Python version of SigClust. Currently only implements the sample covariance method.

### Quick start

Use the `.fit` method, where `data=` is a pandas DataFrame or numpy matrix. Rows should be observations and columns should be features. Pass a list (or array) of cluster labels to `labels=`. Access the p-value from the procedure with `sc.p_value`.

```python
from sigclust import SigClust
sc = SigClust()
sc.fit(data=dataframe, labels=[1, 1, 2, 2])

print(sc.p_value)
print(sc.z_score)
```

### To run tests:
```
python -m unittest test_sigclust.py
```

### Acknowledgements
I borrow from Arthur Tilley's [Python version](https://github.com/aetilley/sigclust) of SigClust, which is nicely written. My version is more aligned with the `scikit-learn` ecosystem.
