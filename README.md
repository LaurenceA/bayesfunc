# bayesfunc

## Installation
Run:
```python
python setup.py devel
```
which copies symlinks to your package directory.

## Tests

### Python's unittest

The easiest way is to run them using Python's own test library. Assuming you're
in the repository root:

```sh
python -m unittest
```
To run a single test, you have to use module path loading syntax:

```sh
# All tests in file
python -m unittest testing.test_models
# Run all tests in a class
python -m unittest testing.test_models.TestRaoBDenseNet
# Run a single test
python -m unittest testing.test_models.TestRaoBDenseNet.test_likelihood
```
which requires that `testing` be a valid module, so it must have an `__init__.py` file.

