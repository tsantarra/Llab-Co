#!/bin/bash

# transfer data from stashcache
# module load stashcp
# stashcp /user/tsantarra/input_data/$2.npy data.npy

# load necessary modules
#module --ignore-cache load python/3.6.5-5
module --ignore-cache load python/3.7.0

# untar local library
tar -xzf Llab-Co.tar.gz

cd Llab-Co

# untar and activate virtual environment
# tar -xzf test_env.tar.gz
source ./test_env/bin/activate

# Make directory for output files
mkdir -p out

# Run python script
./test_env/bin/python3 experiments.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}

cd ..

# tar output file
tar -czf out.$1-$2-$3-$4-$5-$6-$7-$8-$9-${10}-${11}-${12}.tar.gz Llab-Co/out

# Remove loaded files so not copied back
rm -rf Llab-Co
# rm -rf test_env

# deactivate virtual environment
deactivate