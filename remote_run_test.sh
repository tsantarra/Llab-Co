#!/bin/bash
# load necessary modules
module load python/3.5
#module load stashcp

# transfer data from stashcache
#stashcp /user/tsantarra/input_data/$2.npy data.npy

# untar and activate virtual environment
#tar -xzf python_virtenv_demo.tar.gz
#source ./python_virtenv_demo/bin/activate

# untar local library
#tar -xzf misshapen.tar.gz

# Make directory for output files
mkdir out

# Run python script
#./python_virtenv_demo/bin/python2.7 find_PsTs.py
python3.5 test_run.py $1 $2

# tar output file
tar -czf out.$1.$2.tar.gz out

# Remove loaded data so not copied back
#rm data.npy

# deactivate virtual environment
#deactivate