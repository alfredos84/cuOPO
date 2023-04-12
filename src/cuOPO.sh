#!/bin/bash

# This file contains a set of instructions to run the main file cuOPO.cu.
# Use this file to perform simulations in bulk. This means, for example, 
# systematically varying the input power, the cavity reflectivity, etc.


clear                                # Clear screen
rm *.dat                             # This removes all .dat files in the current folder. Comment this line for safe.
rm *.txt                             # This removes all .txt files in the current folder. Comment this line for safe. 
rm cuOPO                             # This removes a previuos executable file (if it exist)


########################################################################################################################################
# -----Compilation-----
# Notice there are 3 preprocessor variables in this compilation that are useful to set
# the regime and the number of equations used in the simulations:
# a) For set the regime use: -DCW_OPO (for cw) or -DNS_OPO (for nanosecond). You must to define it!!. 
# b) For set three coupled-wave equations use: -DTHREE_EQS (two eqs. is set by default).
# c) For set the used nonlinear crystal: -DPPLN (for MgO:PPLN) or -DSPPLT.

# nvcc cuOPO.cu -DCW_OPO -DPPLN --gpu-architecture=sm_60 -lcufftw -lcufft -o cuOPO
nvcc cuOPO.cu -DCW_OPO -DTHREE_EQS -DPPLN --gpu-architecture=sm_60 -lcufftw -lcufft -o cuOPO
FOLDERSIM="Simulations"

# There are three flags specific for CUDA compiler:
# --gpu-architecture=sm_75: please check your GPU card architecture (Ampere, Fermi, Tesla, Pascal, Turing, Kepler, etc) 
#                           to set the correct number sm_XX. This code was tested using a Nvidia GeForce GTX 1650 card (Turing
#                           architecture). 
# 				    Please visit https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
#                           to set the proper flag according to your GPU card.
# -lcufftw -lcufft        : flags needed to perform cuFFT (the CUDA Fourier transform)
########################################################################################################################################


########################################################################################################################################
# This is only for the folder name 
REG="cw"                             # Set regime: ns or cw. 
EQS="3"                              # Set number of equations to solve (2 or 3)
########################################################################################################################################


# The variables defined below (ARGX) will be passed as arguments to the main file 
# cuOPO.cu on each execution via the argv[X] instruction.

LP=(532)                             # Pump wavelength                                            (ARG1)
TCR=(37.4305)                        # Phase-matching temperature                                 (ARG2)
GP=(6.95339)                         # Grating period                                             (ARG3)
NN=(4)                               # N = Power/Pth                                              (ARG9)
RE=(98)                              # Reflectivity at signal wl (in percent %)                   (ARG5)
D=(0)                                # Net cavity detuning (in rad)                               (ARG6)
GD=(0)                               # GDD compensation (in percent %)                            (ARG7)
TD=(0)                               # TOD compensation (in percent %)                            (ARG8)
UPM=(0)                              # Using phase modulator: OFF/ON = 0/1                        (ARG9)
MD=(0)                               # EOM: β (modulation depth in π rads)                        (ARG10)
FM=(0)                               # δf = FSR - fpm [MHz] Frequency detuning for EOM            (ARG11)


# Each for-loop span over one or more values defined in the previous arguments. 
for (( l=0; l<${#LP[@]}; l++ ))
do  
	for (( u=0; u<${#UPM[@]}; u++ ))
	do  
		for (( n=0; n<${#NN[@]}; n++ ))
		do  
			for (( m=0; m<${#MD[@]}; m++ ))
			do  
				for (( f=0; f<${#FM[@]}; f++ ))
				do
					for (( i=0; i<${#D[@]}; i++ ))
					do  
						for (( r=0; r<${#RE[@]}; r++ ))
						do	
							for (( g=0; g<${#GD[@]}; g++ ))
							do
								for (( t=0; t<${#TD[@]}; t++ ))
								do
									L=${LP[$l]}
									printf "\nPump wavelength  = ${L} nm\n" 
									
									TEMP=${TCR[$l]}
									printf "\nPhase-matching T = ${TEMP} ºC\n"
									
									GRPER=${GP[$l]}
									printf "\nGrating period   = ${GRPER} um\n"
									
									N=${NN[$n]} 
									printf "\nPower/Pth        = ${N} \n" 
									
									R=${RE[$r]} 
									printf "\nR                = ${R} %% \n"
									
									DELTAS=${D[$i]}
									printf "\ndelta            = ${DELTAS}\n"
									
									GDD=${GD[$g]} 
									printf "\nGDD compensation = ${GDD}%%\n"
									
									TOD=${TD[$t]}
									printf "\nTOD compensation = ${TOD}%%\n"
									
									U=${UPM[$u]} 
									printf "\nPhase mod ON/OFF = ${UPM}\n"
									
									MODDEP=${MD[$m]}
									printf "\nModulation depth = ${MODDEP}\n"
									
									FREQMOD=${FM[$f]}
									printf "\nFrequency mod    = ${FREQMOD} \n"
									
									printf "\nMaking directory...\n"
									FOLDER="${REG}_${EQS}eqs_PPLN_beta_${MODDEP}_N_${N}_GDD_${GDD}_LP_${L}nm"
									FILE="${REG}_${EQS}eqs_PPLN_beta_${MODDEP}_N_${N}_GDD_${GDD}_LP_${L}nm.txt"
									
									printf "Bash execution and writing output file...\n\n"
									./cuOPO $L $TEMP $GRPER $N $R $DELTAS $GDD $TOD $U $MODDEP $FREQMOD | tee -a $FILE
									
									printf "Bash finished!!\n\n" 
									mkdir $FOLDER
									mv *.dat $FOLDER"/"
									mv *.txt $FOLDER"/"
								done
							done
						done
					done
				done
			done
		done
	done
done


if [ -d "$FOLDERSIM" ]; then
	echo "Moving simulations in ${FOLDERSIM}..."
	mv cw_* $FOLDERSIM"/" 
else

	mkdir $FOLDERSIM
	echo "Creating and moving simulations in ${FOLDERSIM}..."
	mv cw_* $FOLDERSIM"/" 
fi

