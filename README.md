# cuOPO package

cuOPO is a C/CUDA toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes.

## Setup and execution
To run this project, download the folder containing the files and open a terminal. Then execute the following command lines:

```
$ chmod 777 cuOPO.sh # enable permissions for execution
$ ./cuOPO.sh         # execute the files
```

You will notice that the simulation will start, and when finished a new folder will have been created containing the output files.
