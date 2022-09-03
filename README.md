# cuOPO package

cuOPO is a C/CUDA toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes.

## Setup and execution

To run simulations using the package clone this project typing in a terminal
```
git clone https://github.com/alfredos84/cuOPO.git
```
Once the project was cloned, the user will find a parent folder `cuOPO` containing two other
- `src`: contains the main file `cuOPO.cu`, the header files `<header>.h`, and the bash file `cuOPO.sh` used to compile and execute the package by passing several simulations parameters.
- `cw_3eqs_PPLN_delta_0_POWER_3.5`: this folder contains the output files for a given set of parameters and should be taken as an example. After cloning the project, this folder should be **renamed** before running the first test simulation. If not, the new files will replace the older ones.

### Bash file `src/cuOPO.sh`

The bash file is mainly used to massively perform simulations by passing the main file different parameters such as pump power, cavity detuning, etc. Before starting the user has to allow the system execute the bash file. To do that type in the terminal
```
chmod 777 cuOPO.sh # enable permissions for execution
```

Finally, to execute the file execute the following command line
```
./cuOPO.sh         # execute the files
```

When finished a new folder named `cw_3eqs_PPLN_delta_0_POWER_3.5` will have been created containing the output files.

In the `cuOPO.sh` file you will find the command line for the compilation:
```
nvcc cuOPO.cu -D<REGIME> --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO
```
where the preprocessor variable `<REGIME>` could be either `CW_OPO` or `NS_OPO`. This compiles the package using two coupled-wave equations. If the user wants to use three coupled-wave equations, add the additional preprocessor variable `-DTHREE_EQS` in the compilation line. The flag `--gpu-architecture=sm_75` is related to the GPU architecture, and the user should check the proper number (instead of 75). The flags `-lcufftw` and `-lcufft` tell the compiler to use the `CUFFT library` that performs the Fourier transform on GPU .

Finally, the execution is done using the command line in the `cuOPO.sh` file is
```
./cuOPO $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $N $U $MODDEP $FREQMOD | tee -a $FILE
```
where `$ARGx` and others variables externaly passed to the main file `cuOPO.cu`. It was written this way to make easy to massively perform simulations.

For any questions or queries, do not hesitate to contact the developer by writing to alfredo.daniel.sanchez@gmail.com
