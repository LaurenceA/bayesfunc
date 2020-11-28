from setuptools import setup

setup(
    name="bayesfunc",
    version="0.1.0",
    description="Variational inference for BNN, DGP and DKP priors over functions",
    long_description="",
    author="Laurence Aitchison, Sebastian Ober",
    author_email="laurence.aitchison@gmail.com",
    url="https://github.com/laurencea/bayesfunc/",
    license="MIT license",
    packages=["bayesfunc", "uci"],
    test_suite="testing",
)
