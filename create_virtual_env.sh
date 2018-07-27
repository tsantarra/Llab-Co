
#!/bin/bash

# Load python
# module load python/3.5.2

# Activate virtual environment
python -m venv test_env
source test_env/Scripts/activate

# Install python packages (some are unused in this demo)
pip install logmatic-python

# Deactivate virtual environment
deactivate

# Compress virtual environment
tar -cvzf test_env.tar.gz test_env

# Remove directory
rm -R test_env