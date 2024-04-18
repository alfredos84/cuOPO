#!/bin/bash

# This file contains a set of instructions to run the main file cuOPO.cu.
# Use this file to perform simulations in bulk. This means, for example, 
# systematically varying the input power, the cavity reflectivity, etc.
# For such propose, please insert for-loops for any variable parameter.


clear            # Clear screen
rm *.dat         # This removes all .dat files in the current folder. Comment this line for safe.
rm *.txt         # This removes all .txt files in the current folder. Comment this line for safe. 
rm cuOPO         # This removes a previuos executable file (if it exist)


########################################################################################################################################

## Compilation

# To compile CUDA files NVCC compiler is used here (https://developer.nvidia.com/cuda-downloads). 
# The main file cuOPO.cu is compiled by

## for three coupled wave equations
# nvcc cuOPO.cu -diag-suppress 177 -DTHREE_EQS -DCW_OPO -DPPLN  --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO # for two equations



## for two coupled wave equations

CRYSTAL=("sPPLT" "PPLN")  # Set different crystals
for CR in "${CRYSTAL[@]}"; do

	if [ "$CR" = "PPLN" ]; then
		echo "Chosen crystal: MgO:PPLN."
		nvcc cuOPO.cu -diag-suppress 177 -DCW_OPO -DPPLN  --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO # for two equations
	elif [ "$CR" = "sPPLT" ]; then
		echo "Chosen crystal: MgO:sPPLT."
		nvcc cuOPO.cu -diag-suppress 177 -DCW_OPO -DSPPLT --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO # for two equations
	else
		echo "The CR variable has no value specified."
	fi

	# There are 3 preprocessor variables in this compilation used in the simulations:
	# a) Set the pumping regime: -DCW_OPO (for continuous wave) or -DNS_OPO (for nanosecond). Users must to define it!!. 
	# b) Set three coupled-wave equations use: -DTHREE_EQS (two eqs. is set by default).
	# c) Set the used nonlinear crystal: -DPPLN or -DSPPLT (for MgO:PPLN or MgO:sPPLT, respectively). Users must to define one of them!!
	#    Users are also be able to create analogous files containing the informacion for the specific nonlinear crystal to be used. 


	# There are three flags specific for CUDA compiler:
	# --gpu-architecture=sm_75: please check your GPU card architecture (Ampere, Fermi, Tesla, Pascal, Turing, Kepler, etc) 
	#                           to set the correct number sm_XX. This code was tested using a Nvidia GeForce GTX 1650 card (Turing
	#                           architecture). 
	# 				    		Please visit https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
	#                           to set the proper flag according to your GPU card.
	# -lcufftw -lcufft        : flags needed to perform cuFFT (the CUDA Fourier transform)
	########################################################################################################################################


	########################################################################################################################################

	## This is only for the folder name 
	REG="cw"                             # Set regime: ns or cw. 
	EQS="2"                              # Set number of equations to solve (2 or 3)

	# The variables defined below (ARGX) will be passed as arguments to the main file 
	# cuOPO.cu on each execution via the argv[X] instruction.

	if [ "$CR" = "PPLN" ]; then
		LP=(1550)		# Pump wavelength                                            (ARG1)
		TCR=(32.88)		# Phase-matching temperature                                 (ARG2)
		GP=(35.01)		# Grating period                                             (ARG3)
	elif [ "$CR" = "sPPLT" ]; then
		LP=(532)		# Pump wavelength                                            (ARG1)
		TCR=(37.4305)	# Phase-matching temperature                                 (ARG2)
		GP=(7.97)		# Grating period                                             (ARG3)
	else
		echo "Longitud de onda, Temperatura y grating period no especificados."
	fi

	NN=(2 3 4 5)			# N = Power/Pth, pumping level                         		 (ARG4)
	RE=(98)					# Reflectivity at signal wl (in percent %)                   (ARG5)
	D=(0)					# Net cavity detuning (in rad)                               (ARG6)
	GD=(100)				# GDD compensation (in percent %)                            (ARG7)
	TD=(0)  				# TOD compensation (in percent %)                            (ARG8)
	UPM=(1)					# Using phase modulator: OFF/ON = 0/1                        (ARG9)
	MD=(0.5 0.6 0.7 0.8)	# EOM: β (modulation depth in π rads)            			 (ARG10)
	FM=(4)					# δf = FSR - fpm [MHz] Frequency detuning for EOM            (ARG11)
	LCR=(40)				# Lcr Crystal length [mm]									 (ARG12)

	# Each for-loop span over one or more values defined in the previous arguments. 
	for (( lc=0; lc<${#LCR[@]}; lc++ ))
	do  
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

											LCRY=${LCR[$lc]}
											printf "\nCrystal length   = ${LCRY} \n"
											
											printf "\nMaking directory...\n"
											if [ "$CR" = "PPLN" ]; then
												FOLDER="MgO_${CR}_beta_${MODDEP}_N_${N}_GDDcomp_${GDD}_TOD_comp${TOD}_LP_${L}nm_Lcr_${LCRY}"
												FILE="MgO_${CR}_beta_${MODDEP}_N_${N}_GDDcomp_${GDD}_TOD_comp${TOD}_LP_${L}nm_Lcr_${LCRY}.txt"
											elif [ "$CR" = "sPPLT" ]; then
												FOLDER="MgO_${CR}_beta_${MODDEP}_N_${N}_GDDcomp_${GDD}_TOD_comp${TOD}_LP_${L}nm_Lcr_${LCRY}"
												FILE="MgO_${CR}_beta_${MODDEP}_N_${N}_GDDcomp_${GDD}_TOD_comp${TOD}_LP_${L}nm_Lcr_${LCRY}.txt"
											else
												echo "Nunguna salida."
											fi
																					
											printf "Bash execution and writing output file...\n\n"
											./cuOPO $L $TEMP $GRPER $N $R $DELTAS $GDD $TOD $U $MODDEP $FREQMOD $LCRY | tee -a $FILE
											
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
	done

	
done

# if [ -d "$FOLDERSIM" ]; then
# 	echo "Moving simulations in ${FOLDERSIM}..."
# 	mv cw_* $FOLDERSIM"/" 
# else

# 	mkdir $FOLDERSIM
# 	echo "Creating and moving simulations in ${FOLDERSIM}..."
# 	mv cw_* $FOLDERSIM"/" 
# fi

# mv -v $FOLDERSIM"/" ..
