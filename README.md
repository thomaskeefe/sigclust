# SigClust

Python implementation of SigClust, as described by [Huang et al](https://arxiv.org/abs/1305.5879).

To install in development mode, use `conda develop .` after setting up the conda environment. If you accidentally run this more than once, conda will recognize that and not do anything. You can "uninstall" it project (i.e. un-link it) using `conda develop -u .`. 

To run tests, use `python -m unittest discover -s tests/`

### Quick start

Use the `.fit` method, where `data=` is a pandas DataFrame or numpy matrix. Rows should be observations and columns should be features. Pass a list (or array) of cluster labels to `labels=`. Access the p-value from the procedure with `sc.p_value`.

```python
from sigclust import SigClust
import matplotlib.pyplot as plt

sc = SigClust()
sc.fit(data=dataframe, labels=[1, 1, 2, 2])

print(sc.p_value)
print(sc.z_score)

# Plot null distribution and test statistic
plt.hist(sc.simulated_cluster_indices)
plt.axvline(sc.sample_cluster_index)
```

### To run tests:
```
cd tests/
python -m unittest discover
```

### Acknowledgements
I borrow from Arthur Tilley's [Python version](https://github.com/aetilley/sigclust) of SigClust.
