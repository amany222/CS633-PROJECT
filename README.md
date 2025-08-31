# Parallel 3D Data Analysis using MPI

This project was developed as part of **CS433: Parallel Computing**.  
It implements a **parallel program with MPI** to efficiently process large 3D datasets by distributing computations across multiple processes in a high-performance computing (HPC) environment.



## Overview
- The program partitions a **3D dataset** across processes using **domain decomposition**.
- Each process computes local **minima and maxima** on its subdomain.
- Uses **non-blocking MPI communication** with custom datatypes to exchange boundary data with neighbors.
- Final global extrema are computed using **MPI collective reductions**.

-

#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --time=00:10:00
#SBATCH --partition=standard

echo `date`
mpirun -np 32 ./src data_64_64_96_7.bin.txt 2 2 2 64 64 96 7 output.txt
echo `date`

