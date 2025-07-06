from setuptools import setup, find_packages

setup(
    name="pyWOMBAT-lite",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
    ],
    description="A 1D biogeochemical model package of WOMBAT-lite",
    author="Pearse James Buchanan",
    license="CSIRO",
)