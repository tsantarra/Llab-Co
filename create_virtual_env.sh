
#!/bin/bash

# Load python - not needed?
# module load python

# Activate virtual environment
python -m venv test_env
source test_env/bin/activate

# Install python packages (some are unused in this demo)
pip install logmatic-python

# Deactivate virtual environment
deactivate

# Compress virtual environment
tar -cvzf test_env.tar.gz test_env

# Remove directory
rm -R test_env