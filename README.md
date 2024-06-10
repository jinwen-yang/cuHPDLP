# cuHPDLP

This is the code accompanying the paper "Restarted Halpern PDHG for Linear Programming". This repository contains experimental code for solving linear programming using restarted Halpern PDHG on NVIDIA GPUs. 

## Setup

A one-time step is required to set up the necessary packages on the local machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running 

All commands below assume that the current directory is the working directory.

```shell
$ julia --project scripts/solve.jl \
--instance_path=INSTANCE_PATH \
--output_directory=OUTPUT_DIRECTORY \ 
--tolerance=TOLERANCE \
--time_sec_limit=TIME_SEC_LIMIT
```