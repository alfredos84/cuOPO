# cuOPO

cuOPO is a C/CUDA toolkit for simulating optical parametric oscillators using the coupled-wave equations (CWEs) that well-describe the three wave mixing (TWM) processes in a second-order nonlinear media.

The provided software implements a solver for the CWEs including dispersion terms, linear absorption and intracavity element if they are required. It also includes flags to solve nanosecond or continuous wave time regimes.


#Execution

When you download it, test the cuOPO.sh file and run it. Before doing this, first enable the appropriate permissions to run in a terminal. For example by typing the command line
chmod 777 cuOPO.sh

Once the permissions are enabled, run the file in the terminal by typing
./cuOPO.sh
You will notice that the simulation will start, and when finished a new folder will have been created containing the output files.
