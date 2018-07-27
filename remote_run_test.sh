#!/bin/bash
# load necessary modules
module load python/3.6
#module load stashcp

# transfer data from stashcache
# stashcp /user/tsantarra/input_data/$2.npy data.npy

# untar and activate virtual environment
tar -xzf test_env.tar.gz
source ./test_env/Scripts/activate

# untar local library
#tar -xzf misshapen.tar.gz

# Make directory for output files
mkdir out

# Run python script
./test_env/Scripts/python test_run.py $1 $2
# python3.5 test_run.py $1 $2

# tar output file
tar -czf out.$1.$2.tar.gz out

# Remove loaded data so not copied back
#rm data.npy

# deactivate virtual environment
deactivate