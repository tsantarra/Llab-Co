#!/bin/bash

# load necessary modules
module load python/3.5.2

# transfer data from stashcache
# module load stashcp
# stashcp /user/tsantarra/input_data/$2.npy data.npy

# untar and activate virtual environment
tar -xzf test_env.tar.gz
source ./test_env/bin/activate

# untar local library
tar -xzf Llab-Co.tar.gz

# Make directory for output files  - already have one!
rm Llab-Co/out/*
# mkdir out

# Run python script
./test_env/bin/python3 Llab-Co/test_run.py $1 $2 $3 $4 $5 $6 $7 $8 $9

# tar output file
tar -czf out.$1-$2-$3-$4-$5-$6-$7-$8-$9.tar.gz Llab-Co/out

# Remove loaded files so not copied back
rm -rf Llab-Co
rm -rf test_env

# deactivate virtual environment
deactivate