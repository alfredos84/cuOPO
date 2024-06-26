# cuOPO package

is a CUDA-based toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media. CUDA programming allows you to implement parallel computing in order to speed up calculations that typically require a considerable computational demand.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes. However, the user is free to incorporate picosecond or femtosecond regimes by making the proper corrections.

This code is useful for simulations based on three-wave mixing proccesses such as optical parametric oscillators (OPOs).
It solves the coupled-wave equations (CWEs) for signal, idler and pump using a parallel computing scheme based on Split-Step Fourier Method (SSFM).

If this code is used for further publications, please cite the following article (https://doi.org/10.1016/j.cpc.2023.108910):
Sanchez, A. D., S. Chaitanya Kumar, and M. Ebrahim-Zadeh. "CUDA-based optical parametric oscillator simulator." Computer Physics Communications 294 (2024): 108910.

For running this code is necessary to have a GPU in your computer and installed the CUDA drivers and the CUDA-TOOLKIT as well. 
To install the CUDA driver on a Linux system please visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html


## Setup and execution

To run simulations using the package clone this project typing in a terminal
```
git clone https://github.com/alfredos84/cuOPO.git
```
Once the project was cloned, the user will find a parent folder `cuOPO` containing two other
- `src`: contains the main file `cuOPO.cu`, the headers folder, and the bash file `cuOPO.sh` used to compile and execute the package by passing several simulations parameters.
- `src/headers`: this folder contains the headers files needed to execute the package.

### Bash file `src/cuOPO.sh`

The bash file is mainly used to massively perform simulations by passing the main file different parameters such as pump power, cavity detuning, etc. Before starting the user has to allow the system execute the bash file. To do that type in the terminal
```
chmod 777 cuOPO.sh # enable permissions for execution
```

Finally, to execute the file execute the following command line
```
./cuOPO.sh         # execute the files
```

When finished a new folder named `cw_2eqs_PPLN_beta_0_N_4_GDD_0_LP_532nm` will have been created containing the output files.

In the `cuOPO.sh` file you will find the command line for the compilation:
```
nvcc cuOPO.cu -D<REGIME> -D<CRYSTAL> -DTHREE_EQS --gpu-architecture=sm_XX -lcufftw -lcufft -o cuOPO
```
where the preprocessor variable `<REGIME>` could be either `CW_OPO` or `NS_OPO`. `<CRYSTAL>` set the nonlinear crystal used in simulations. Currently, there are available two crystals, namely, MgO:PPLN and Mgo:sPPLT. This compiles the package using two coupled-wave equations. If the user wants to use three coupled-wave equations, add the additional preprocessor variable `-DTHREE_EQS` in the compilation line. The flag `--gpu-architecture=sm_XX` is related to the GPU architecture (see below more information about this point). The flags `-lcufftw` and `-lcufft` tell the compiler to use the `CUFFT library` that performs the Fourier transform on GPU .

Finally, the execution is done using the command line in the `cuOPO.sh` file is
```
./cuOPO <ARGUMENTS_TO_PASS>
```
where `$ARGx` and others are variables externaly passed to the main file `cuOPO.cu`. It was written in this way to make easier performing simulations massively.

### Outputs

This package returns a set of `.dat` files with the signal, idler and pump electric fields, separated into real and imaginary parts. It also returns time and frequency vectors

### GPU architecture
Make sure you know your GPU architecture before compiling and running simulations. For example, pay special attention to the sm_75 flag defined in the provided `cuOPO.sh` file. That flag might not be the same for your GPU since it corresponds to a specific architecture. For instance, I tested this package using two different GPUs:
1. Nvidia Geforce MX250: architecture -> Pascal -> flag: sm_60
2. Nvidia Geforce GTX1650: architecture -> Turing -> flag: sm_75

Please check the NVIDIA documentation in https://docs.nvidia.com/cuda/


### Contact me
For any questions or queries, do not hesitate to contact the developer by writing to alfredo.daniel.sanchez@gmail.com
