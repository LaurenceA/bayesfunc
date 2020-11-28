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
    install_requires=[
        "torch>=1.5.0",
        # Coarse dependencies
        "numpy>=1.0<2.0",
        "scipy>=1.5.2<1.6",
    ],
    test_suite="testing",
)
