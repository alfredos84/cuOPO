# cuOPO package

cuOPO is a C/CUDA toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes.

## Setup and execution
To run this project, download the folder `/src` containing the files and open a terminal. Then execute the following command lines:

```
$ chmod 777 cuOPO.sh # enable permissions for execution
$ ./cuOPO.sh         # execute the files
```

You will notice that the simulation will start, and when finished a new folder will have been created containing the output files.

In the `cuOPO.sh` file you will find the command line for the compilation:
```
$ nvcc cuOPO.cu -D<REGIME> --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO
```
where the preprocessor variable `<REGIME>` could be either `CW_OPO` or `NS_OPO`. This compiles the package using two coupled equations. If the user wants to use three coupled equations, use the additional preprocessor variable `-DTHREE_EQS`. The compilation flag `--gpu-architecture=sm_75` is related to the GPU architecture, and the user should check the proper number (here we use 75). The flags `-lcufftw` and `-lcufft`.

Finally, the execution is done using the command line in the `cuOPO.sh` file is
```
$ ./cuOPO $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $N $U $MODDEP $FREQMOD | tee -a $FILE
```
where `$ARGx` and others are passed values for several variables needed for simulations.

For any questions or queries, do not hesitate to contact the developer by writing to alfredo.daniel.sanchez@gmail.com
