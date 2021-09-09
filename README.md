# bayesfunc

## Installation
Run:
```python
python setup.py develop
```
which copies symlinks to your package directory.

## Tutorial
Look at examples/simple.ipynb in the repo.

## Documentation
https://bayesfunc.readthedocs.io/en/latest/

## Make documentation locally
Navigate to `docs` and run `make html`.

## TODOs
- Check full covariance for all "fully-connected" units
- fix L property for gp
- allow the sampled function to be "frozen"
- copy nets; draw a sample from one net and compute log-prob under the other
- regression tests
- finalise docs
- guide to extending kernels
- philosophy of having no learned prior parameters
- parameterize "Choleskys" as lower triangular not full
