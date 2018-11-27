#!/bin/bash

# Load python
module load python/3.6.5-5
#module --ignore-cache load python/3.7.0

# Activate virtual environment
rm -rf test_env/*
#python3 -m venv test_env
virtualenv -p python3 test_env
source test_env/bin/activate

# Install python packages (some are unused in this demo)
pip install logmatic-python

# Deactivate virtual environment
deactivate

# Compress virtual environment
# tar -cvzf test_env.tar.gz test_env

# Remove directory
# rm -R test_env
