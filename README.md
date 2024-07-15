# Trying out Mojo 24.4 with data science libs
This project was just for learning the basics of executing mojo code within a conda environment,
and in this case how common data science libs such as numpy, pandas, polars and scikit-learn
can be imported and used in mojo.

TODO: replace python libs when the community adds native counterparts for managing DataFrames etc.

## Install and run
Mojo does not include a python binary, but it will look for one on the top of the PATH. This lets us simply activate a conda environment where python is installed and mojo will use that python version.

1. Install mojo, see https://docs.modular.com/mojo/manual/get-started

2. Install Conda, https://github.com/conda-forge/miniforge

3. Install the conda environment
```bash
conda env create -f environment.yaml -n mojo-rul
conda activate mojo-rul
```

4. Run the code
```bash
mojo src/run.mojo
```