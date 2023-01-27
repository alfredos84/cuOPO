/* Author Alfredo Daniel Sanchez: alfredo.daniel.sanchez@gmail.com */

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#include <sys/time.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "common.h"
#include "refindex.h"
#include "functions.h"

// Define 2 or 3 coupled-wave equations
#ifdef THREE_EQS
#include "cwes3.h"
#else
#include "cwes2.h"
#endif

#include "savefiles.h"

// Complex data type
using complex_t = cufftComplex;
using real_t = float;


int main(int argc, char *argv[]){
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Set GPU and timing
	
	std::cout << "\n\n\n#######---Welcome to OPO calculator---#######\n\n\n" << std::endl;
	
	time_t current_time; // timing the code
	time(&current_time);
	std::cout << "Date: " << ctime(&current_time) << std::endl;
	real_t iStart = seconds();


	#ifdef CW_OPO
	std::cout << "Regime: continuous wave" << std::endl;
	#endif
	#ifdef NS_OPO
	std::cout << "Regime: nanosecond" << std::endl;
	#endif
	#ifdef THREE_EQS
	std::cout << "Three equations" << std::endl;
	#else
	std::cout << "Two equations" << std::endl;
	#endif
	
	// Set up device (GPU) //
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	std::cout << "\n\nUsing Device " << dev << ": GPU " << deviceProp.name << std::endl;
	CHECK(cudaSetDevice(dev));
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Define relevant parameters //	
	const real_t PI   = 3.14159265358979323846;          //pi
	const real_t C    = 299792458*1E6/1E12;              // speed of ligth in vacuum [um/ps]
	const real_t EPS0 = 8.8541878128E-12*1E12/1E6;       // vacuum pertivity [W.ps/V²μm] 
	
	// Set parameters and constants //
	int N_rt         = atoi(argv[8]); // number of simulation round trips

	// Grids, crystal and cavity parameters //
	real_t lp        = atof(argv[15])*1e-3;  // pump wavelength   [μm]
	real_t ls        = 2*lp;           // signal wavelength [μm]
	real_t li        = lp*ls/(ls-lp);  // idler wavelength  [μm]

	real_t deff      = 14.77e-6;       // effective second-order susceptibility [um/V]
	real_t Temp      = atof(argv[16]); // crystal temperature [ºC]
	real_t Lambda    = atof(argv[17]); // grating period for QPM [μm]  
	real_t Lcr       = 5e3;            // crystal length [um]
		
	real_t np        = n_PPLN(lp, Temp);         // pump ref. index
	real_t vp        = group_vel_PPLN(lp, Temp); // pump group velocity [μm/ps]
	real_t b2p       = gvd_PPLN(lp, Temp);       // pump GVD [ps²/μm] 
	real_t b3p       = 0.*TOD_PPLN(lp, Temp);    // pump TOD [ps³/μm]	
	real_t kp        = 2*PI*deff/(np*lp);        // pump kappa [1/V]
	real_t alpha_crp = 0.002e-4;                 // pump linear absorption [1/μm]
	
	real_t ns        = n_PPLN(ls, Temp);         // signal ref. index
	real_t vs        = group_vel_PPLN(ls, Temp); // signal group velocity [μm/ps]
	real_t b2s       = gvd_PPLN(ls, Temp);       // signal GVD [ps²/μm] 
	real_t b3s       = 0.*TOD_PPLN(ls, Temp);    // signal TOD [ps³/μm]		
	real_t ks        = 2*PI*deff/(ns*ls);        // signal kappa [1/V]
	real_t alpha_crs = 0.025e-4;                 // signal linear absorption [1/μm]

	real_t ni        = n_PPLN(li, Temp);         // idler ref. index
	real_t vi        = group_vel_PPLN(li, Temp); // idler group velocity [μm/ps]
	real_t b2i       = gvd_PPLN(li, Temp);       // idler GVD [ps²/μm]
	real_t b3i       = 0.*TOD_PPLN(li, Temp);    // idler TOD [ps³/μm]	
	real_t ki        = 2*PI*deff/(ni*li);        // idler kappa [1/V]
	real_t alpha_cri = 0.025e-4;                 // idler linear absorption [1/μm]
	
	real_t dk        = 2*PI*( np/lp-ns/ls-ni/li-1/Lambda ); // mismatch factor
	real_t dkp       = 1/vp-1/vs;                           // group velocity mismatch	
	real_t Lcav      = atoi(argv[4]) * Lcr;                 // cavity length [um]
	real_t Rs        = atof(argv[5])*0.01;                  // Reflectivity at signal wavelength 
	real_t alphas    = 0.5*((1-Rs)+alpha_crs*Lcr);          // Total losses for threshold condition signal
	#ifdef THREE_EQS
	real_t Ri        = 0.8;                                 // Reflectivity at idler wavelength 
	real_t alphai    = 0.5*((1-Ri)+alpha_cri*Lcr);          // Total losses for threshold condition idler
	#endif
	real_t t_rt      = (Lcav+Lcr*(ns-1))/C;                 // round-trip time [ps]
	real_t FSR       = 1/t_rt;	                          // free-spectral range
	real_t finesse   = 2*PI/(1-Rs);                         // cavity finesse
	real_t lw        = FSR/finesse*1e6;                     // cavity Linewidth [MHz]
 	real_t delta     = atof(argv[6]);                       // cavity detuning [rad] 
	real_t epsilon   = atof(argv[7])*0.01;                  // dispersion compensation index
	real_t GDD       = -epsilon*b2s*Lcr;                    // GDD [ps²]
	real_t TODscomp  = -0.01*atof(argv[13])*b3s*Lcr;        // TOD compensation [ps³]
	real_t TODicomp  = -0.01*atof(argv[13])*b3i*Lcr;        // TOD compensation [ps³]
	
	// z discretization
	int steps_z      = atoi(argv[3]); // number of crystal divisions
	real_t dz        = Lcr/steps_z;   // number of z-steps in the crystal
	
	
	// Time and frequency discretization
	unsigned int ex    = atoi(argv[2]);
	unsigned int SIZE  = 1 << ex;   // points per time slice
	int extra_win      = 0;         // extra pts for short-time slices
	real_t dT          = t_rt/SIZE; // time step in [ps]
	real_t dF          = 1/t_rt;    // frequency step in [THz]
	
	bool video         = false;
	unsigned int Nrts;        // number of last round trips to save (only for cw)
	if(video){Nrts = 100;}
	else{Nrts = 16;}	
	
	#ifdef CW_OPO
	unsigned int SIZEL = SIZE*Nrts; // size of large vectors for full simulation
	real_t T_width     = t_rt*Nrts; // total time for the saved simulation
	#endif
	#ifdef NS_OPO
	unsigned int SIZEL = SIZE*N_rt; // size of large vectors for full simulation
	real_t T_width     = t_rt*N_rt; // total time for the saved simulation
	#endif	
	
	// Difine which fields are resonant (SRO, DRO or TRO)
	bool is_Ap_resonant = false;
	bool is_As_resonant = true;
	#ifdef THREE_EQS
	bool is_Ai_resonant = true;
	#endif
	
	// Time vector T for one round trip
	real_t *T = (real_t*) malloc(sizeof(real_t) * SIZE);
	linspace( T, SIZE, -0.5*t_rt, 0.5*t_rt);

	// Time vector Tp for full simulation
	real_t *Tp = (real_t*) malloc(sizeof(real_t) * SIZEL);
	linspace( Tp, SIZEL, -0.5*T_width, 0.5*T_width);
	
	// Time vector Fp for full simulation
	real_t dFp  = 1/T_width;
	real_t *Fp = (real_t*) malloc(sizeof(real_t) * SIZEL);
	linspace( Fp, SIZEL, -0.5*SIZEL*dFp, +0.5*SIZEL*dFp);
// 	inic_vector_F(Fp, SIZEL, dFp);

	
	// Frequency and angular frequency vectors f and Ω
	real_t *F = (real_t*) malloc(sizeof(real_t) * SIZE);
	linspace( F, SIZE, -0.5*SIZE*dF, +0.5*SIZE*dF);
	real_t *w = (real_t*) malloc(sizeof(real_t) * SIZE);
	fftshift(w,F, SIZE);
	for (int i=0; i<SIZE; i++)
		w[i] = 2*PI*w[i]; // angular frequency for one round trip [2*pi*THz]
		
			
	// Define memory size for complex host vectors
	int nBytes   = sizeof(complex_t)*SIZE;
	// Define input pump parameters
	real_t waist = 55;             // beam waist radius [um]
	real_t spot  = PI*waist*waist; // spot area [μm²]
	real_t Ith, Pth;               // Power and intensity threshold 
	#ifdef THREE_EQS
	// Power and intensity threshold non-degenerate OPO 
	if (!is_Ai_resonant){
		std::cout << "SRO: As is resonant" << std::endl;
		Ith   = EPS0*C*np*ns*ni*ls*li*pow((1/deff/Lcr/PI),2)*alphas/2;
	}
	if (!is_As_resonant){
		std::cout << "SRO, Ai is resonant" << std::endl;
		Ith   = EPS0*C*np*ns*ni*ls*li*pow((1/deff/Lcr/PI),2)*alphai/2;
	}
	if (is_As_resonant and is_Ai_resonant){
		std::cout << "DRO, As and Ai are resonant" << std::endl;
		Ith   = EPS0*C*np*ns*ni*ls*li*pow((1/deff/Lcr/PI),2)*alphas*alphai/8;
	}	
	Pth   = Ith*spot;
	#else
	// Power and intensity threshold degenerate DRO 
	Ith   = EPS0*C*np*powf((ns*ls*alphas/deff/Lcr/PI), 2)/8;
	Pth   = Ith*spot;
	#endif
	
	real_t Nth   = atof(argv[9]);             // Times over the threshold
	real_t Inten = atof(argv[9])*Ith;         // Pump intensity in [W/um²]
	real_t Power = Inten*spot;                // Pump power in [W]
	real_t Ap0   = sqrt(2*Inten/(np*EPS0*C)); // Input pump field strength [V/μm]
	
	// Define input pump vector
	#ifdef CW_OPO
	complex_t *Ap_in = (complex_t*)malloc(nBytes); // input pump vector
	input_field_T(Ap_in, Ap0, SIZE );              // set input pump vector (cw)
	#endif
	
	#ifdef NS_OPO
	real_t FWHM      = 10000;                              // intensity FWHM for input [ps]
	real_t sigmap    = FWHM*sqrtf(2)/(2*sqrtf(2*logf(2))); // σ of electric field gaussian pulse [ps]
	complex_t *Ap_in = (complex_t*)malloc(sizeof(complex_t)*SIZEL); // input pump vector
	input_field_T(Ap_in, Ap0, Tp, sigmap, SIZEL); // set input pump vector (gaussian pulse)
	#endif
	
	
	// Define input signal vector (NOISE)
	complex_t *As = (complex_t*)malloc(nBytes);
	NoiseGeneratorCPU ( As, SIZE );
	
	#ifdef THREE_EQS	
	// Define input idler vector (NOISE)
	complex_t *Ai = (complex_t*)malloc(nBytes);
	NoiseGeneratorCPU ( Ai, SIZE );
	#endif


	// Intracavy phase modulator
	bool using_phase_modulator = atoi(argv[10]);
	real_t mod_depth, fpm, df;
	if(using_phase_modulator){
		mod_depth       = atof(argv[11])*PI;
		df              = atof(argv[12])*sqrtf(Nth-1)*alphas/(PI*mod_depth)*FSR;
		fpm             = FSR - df;
	}
	
	
	// Intracavy third-order material to produce self-phase modulation (SPM)
	bool using_SPM = atoi(argv[14]);
	if(using_SPM){
		std::cout << "Using third-order material." << std::endl;
	}
	if(!using_SPM){
		std::cout << "Not using third-order material." << std::endl;
	}
	// Define these variables for SPM
	real_t gamma      = 1e10; // nonlinear coefficient
	real_t lfo        = 1.0;  // material length
	real_t alphafo    = 0.01; // absorption
	
	
	// Define string variables for saving files
	std::string Filename, SAux, Extension = ".dat";
	bool save_input_fields = true;	// Save input fields files
	if (save_input_fields){
		#ifdef CW_OPO
		Filename = "pump_input";	SaveFileVectorComplex (Ap_in, SIZE, Filename);
		#endif
		#ifdef NS_OPO
		Filename = "pump_input";	SaveFileVectorComplex (Ap_in, SIZEL, Filename);
		#endif
		Filename = "signal_input";	SaveFileVectorComplex (As, SIZE, Filename);
		#ifdef THREE_EQS	
		Filename = "idler_input";	SaveFileVectorComplex (Ai, SIZE, Filename);	
		#endif
	}
	
	
	
	bool print_param_on_screen = true;	// Print parameters on screen
	if ( print_param_on_screen ){
		std::cout << "\n\nSimulation parameters:\n\n " << std::endl;
		std::cout << "Number of round trips   = " << N_rt  << std::endl;
		std::cout << "Pump wavelength         = " << lp*1e3 << " nm" << std::endl;
		std::cout << "Signal wavelength       = " << ls*1e3 << " nm" << std::endl;
		#ifdef THREE_EQS
		std::cout << "Idler wavelength        = " << li*1e3 << " nm" << std::endl;
		#endif
		std::cout << "Temp                    = " << Temp << " ºC" << std::endl;
		std::cout << "np                      = " << np << std::endl;
		std::cout << "ns                      = " << ns << std::endl;
		std::cout << "ni                      = " << ni << std::endl;
		std::cout << "\u03BD⁻¹ pump                = " << 1.0/vp << " ps/\u03BCm" << std::endl;
		std::cout << "\u03BD⁻¹ signal              = " << 1.0/vs << " ps/\u03BCm" << std::endl;
		#ifdef THREE_EQS
		std::cout << "\u03BD⁻¹ idler               = " << 1.0/vi << " ps/\u03BCm" << std::endl;		
		#endif
		std::cout << "\u0394k                      = " << dk << " \u03BCm⁻¹" << std::endl;
		std::cout << "\u0394k'                     = " << dkp << " ps/\u03BCm" << std::endl;	
		std::cout << "GVD pump                = " << b2p << " ps²/\u03BCm" << std::endl;
		std::cout << "GVD signal              = " << b2s << " ps²/\u03BCm" << std::endl;
		#ifdef THREE_EQS
		std::cout << "GVD idler               = " << b2i << " ps²/\u03BCm" << std::endl;		
		#endif
		std::cout << "TOD pump                = " << b3p << " ps³/\u03BCm" << std::endl;
		std::cout << "TOD signal              = " << b3s << " ps³/\u03BCm" << std::endl;		
		std::cout << "Net GVD                 = " << (1-epsilon)*b2s << " ps²/\u03BCm" << std::endl;
		std::cout << "GVD compensation        = " << atoi(argv[7]) << " %"  << std::endl;
		std::cout << "Net TOD                 = " << (1-0.01*atoi(argv[13]))*b3s*Lcr*1e3 << " fs³"  << std::endl;
		std::cout << "TOD compensation        = " << atof(argv[13]) << " %"  << std::endl;		
		std::cout << "deff                    = " << deff*1e6 << " pm/V"  << std::endl;
		std::cout << "\u039B                       = " << Lambda << " \u03BCm"  << std::endl;
		std::cout << "\u03B1cp                     = " << alpha_crp << " \u03BCm⁻¹"  << std::endl;
		std::cout << "\u03B1cs                     = " << alpha_crs << " \u03BCm⁻¹" << std::endl;
		std::cout << "\u03B1s                      = " << alphas << std::endl;
		#ifdef THREE_EQS
		std::cout << "\u03B1i                      = " << alpha_cri << " \u03BCm⁻¹" << std::endl;
		std::cout << "\u03B1i                      = " << alphas << std::endl;
		#endif
		
		std::cout << "Crystal length          = " << Lcr*1e-3 << " mm"  << std::endl;
		std::cout << "Cavity  length          = " << Lcav*1e-3 << " mm"  << std::endl;
		std::cout << "\u0394z                      = " << dz << " \u03BCm"  << std::endl;
		std::cout << "Reflectivity (signal)   = " << Rs*100 << " %"  << std::endl;
		#ifdef THREE_EQS
		std::cout << "Reflectivity (idler)    = " << Ri*100 << " %"  << std::endl;	
		#endif
		std::cout << "Cavity Finesse          = " << finesse << std::endl;	
		std::cout << "Cavity lw (FWHM)        = " << lw << " MHz"  << std::endl;	
		std::cout << "Round-trip time         = " << std::setprecision(15) << t_rt << " ps"  << std::endl;	
		std::cout << "FSR                     = " << std::setprecision(15) << FSR*1e3 << " GHz"  << std::endl;
		std::cout << "Cavity detuning (\u03B4)     = " << delta << "\u03C0"  << std::endl;	
		std::cout << "Using N                 = 2^" << ex << " = " << SIZE << " points" << std::endl;
		std::cout << "dT                      = " << dT << " ps" << std::endl;
		std::cout << "SIZEL                   = " << SIZEL << std::endl;
		std::cout << "dFp                     = " << dFp << " THz" << std::endl;
		std::cout << "Max frequency           = " << Fp[SIZEL-1] << " THz" << std::endl;
		std::cout << "Ap0                     = " << Ap0 << " V/um" << std::endl; 
		std::cout << "waist                   = " << waist << " \u03BCm" << std::endl;
		std::cout << "spot                    = " << spot << " \u03BCm²" << std::endl;
		std::cout << "Power threshold         = " << Pth << " W" << std::endl;
		std::cout << "Power                   = " << Power << " W" << std::endl;
		std::cout << "Times above the thres.  = " << Nth << std::endl;
		if(using_phase_modulator){
		std::cout << "Using a phase modulator:" << std::endl;
		std::cout << "Mod. depth (\u03B2)          = " << atof(argv[11]) << "\u03C0 rad = " << mod_depth << " rad" << std::endl;
		std::cout << "Freq. detuning (\u03B4f)     = " << df*1e6 << " MHz" << std::endl;
		std::cout << "Mod. frequency(fm)      = " << fpm*1e3 << " GHz" << std::endl;
		std::cout << "\n\nPoint in the space of parameters:\n" << std::endl;
		std::cout << "(N,\u03B2,\u03B4f,\u03B5) = ( " << Nth << ", " << atof(argv[11]) << ", "  << std::setprecision(4) << df*1e6 << ", " << epsilon << " )\n\n" << std::endl;			
		}
		else{std::cout << "No phase modulator" << std::endl;
		std::cout << "\n\nPoint in the space of parameters:\n" << std::endl;
		std::cout << "( N, \u03B2, \u03B4f, \u03B5 ) = ( " << Nth << ", 0, 0, " << std::setprecision(2) << epsilon << " )\n\n" << std::endl;
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Define GPU vectors //
	
	// Parameters for kernels
	int dimx = 1 << 7;
	dim3 block(dimx); dim3 grid((SIZE + block.x - 1) / block.x);
	

	std::cout << "Setting constants and vectors in device..." << std::endl;
	real_t *w_gpu; // angular frequency 
	CHECK(cudaMalloc((void **)&w_gpu, sizeof(real_t) * SIZE ));

	real_t *T_gpu; // time for a single round trip
	CHECK(cudaMalloc((void **)&T_gpu, sizeof(real_t) * SIZE ));    
	CHECK(cudaMemcpy(T_gpu, T, sizeof(real_t)*SIZE, cudaMemcpyHostToDevice));    
    
	
	complex_t *Ap_gpu, *Ap_in_gpu, *Ap_total_gpu, *Apw_gpu, *As_gpu, *As_total_gpu, *Asw_gpu;
	CHECK(cudaMalloc((void **)&As_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Ap_gpu, nBytes ));
	CHECK(cudaMemset(Ap_gpu, 0, nBytes));
	
	#ifdef CW_OPO
	CHECK(cudaMalloc((void **)&Ap_in_gpu, nBytes ));
	#endif
	#ifdef NS_OPO
	CHECK(cudaMalloc((void **)&Ap_in_gpu, sizeof(complex_t)*SIZEL ));
	#endif
	CHECK(cudaMemset(Ap_in_gpu, 0, nBytes));
	CHECK(cudaMalloc((void **)&As_total_gpu, sizeof(complex_t) * SIZEL ));
	CHECK(cudaMalloc((void **)&Ap_total_gpu, sizeof(complex_t) * SIZEL ));
	CHECK(cudaMalloc((void **)&Asw_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Apw_gpu, nBytes ));

	CHECK(cudaMemcpy(As_gpu, As, nBytes, cudaMemcpyHostToDevice));
	#ifdef CW_OPO
	CHECK(cudaMemcpy(Ap_in_gpu, Ap_in, nBytes, cudaMemcpyHostToDevice));	
	#endif
	#ifdef NS_OPO
	CHECK(cudaMemcpy(Ap_in_gpu, Ap_in, sizeof(complex_t) * SIZEL, cudaMemcpyHostToDevice));
	#endif
	
	CHECK(cudaMemcpy(w_gpu, w, sizeof(real_t) * SIZE , cudaMemcpyHostToDevice));
	
    
	// RK4 (kx) and auxiliary (aux) GPU vectors 
	complex_t *k1p_gpu, *k2p_gpu, *k3p_gpu, *k4p_gpu, *k1s_gpu, *k2s_gpu, *k3s_gpu, *k4s_gpu;
	CHECK(cudaMalloc((void **)&k1p_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k2p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k3p_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k4p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k1s_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k2s_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k3s_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k4s_gpu, nBytes ));
	CHECK(cudaMemset(k1p_gpu, 0, nBytes));		CHECK(cudaMemset(k2p_gpu, 0, nBytes));
	CHECK(cudaMemset(k3p_gpu, 0, nBytes));		CHECK(cudaMemset(k4p_gpu, 0, nBytes));
	CHECK(cudaMemset(k1s_gpu, 0, nBytes));		CHECK(cudaMemset(k2s_gpu, 0, nBytes));
	CHECK(cudaMemset(k3s_gpu, 0, nBytes));		CHECK(cudaMemset(k4s_gpu, 0, nBytes));
	
	complex_t *auxp_gpu, *auxs_gpu;
	CHECK(cudaMalloc((void **)&auxp_gpu, nBytes ));	CHECK(cudaMalloc((void **)&auxs_gpu, nBytes ));
	CHECK(cudaMemset(auxp_gpu, 0, nBytes));		CHECK(cudaMemset(auxs_gpu, 0, nBytes));
	
	
	#ifdef THREE_EQS	
	complex_t *Ai_gpu, *Ai_total_gpu, *Aiw_gpu;
	CHECK(cudaMalloc((void **)&Ai_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Ai_total_gpu, sizeof(complex_t) * SIZEL ));
	CHECK(cudaMalloc((void **)&Aiw_gpu, nBytes ));

	CHECK(cudaMemcpy(Ai_gpu, Ai, nBytes, cudaMemcpyHostToDevice));	

	complex_t *k1i_gpu, *k2i_gpu, *k3i_gpu, *k4i_gpu, *auxi_gpu;
	CHECK(cudaMalloc((void **)&k1i_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k2i_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k3i_gpu, nBytes ));	CHECK(cudaMalloc((void **)&k4i_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&auxi_gpu, nBytes ));
	CHECK(cudaMemset(k1i_gpu, 0, nBytes));		CHECK(cudaMemset(k2i_gpu, 0, nBytes));
	CHECK(cudaMemset(k3i_gpu, 0, nBytes));		CHECK(cudaMemset(k4i_gpu, 0, nBytes));
	CHECK(cudaMemset(auxi_gpu, 0, nBytes));
	#endif
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Evolution in the crystal and multiple round-trips loop
	
	
	// Set plan for cuFFT 1D and 2D//
	cufftHandle plan1D; 
	cufftPlan1d(&plan1D, SIZE, CUFFT_C2C, 1);
	
	
	// Main loop (fields in the cavity)
	std::cout << "Starting main loop on CPU & GPU...\n" << std::endl;
	unsigned int mm = 0; // counts for cw saved round trips
	for (int nn = 0; nn < N_rt; nn++){
		if( nn%250 == 0 or nn == N_rt-1 )
			std::cout << "#round trip: " << nn << std::endl;

		#ifdef CW_OPO
		// update the input pump in each round trip
		CHECK(cudaMemcpy( Ap_gpu, Ap_in_gpu, nBytes, cudaMemcpyDeviceToDevice) );
		#endif
		#ifdef NS_OPO
		// read the input pump in nn-th round trip
		ReadPump<<<grid,block>>>( Ap_gpu, Ap_in_gpu, N_rt, nn, extra_win, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		#endif
		
		#ifdef THREE_EQS
		if (!is_Ai_resonant){	// For non-resonant field, it resets Ai in every round trip
			NoiseGeneratorCPU ( Ai, SIZE );
			CHECK(cudaMemcpy(Ai_gpu, Ai, nBytes, cudaMemcpyHostToDevice));			
		}
		#endif
		
		if (!is_As_resonant){	// For non-resonant field, it resets As in every round trip
			NoiseGeneratorCPU ( As, SIZE );
			CHECK(cudaMemcpy(As_gpu, As, nBytes, cudaMemcpyHostToDevice));			
		}
				
		#ifdef THREE_EQS // Single pass for coupled wave equations (2 or 3)
		EvolutionInCrystal( w_gpu, grid, block, Ap_gpu, As_gpu, Ai_gpu, Apw_gpu, Asw_gpu, Aiw_gpu, k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu, k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu, auxp_gpu, auxs_gpu, auxi_gpu, lp, ls, li, vp, vs, vi, b2p, b2s, b2i, b3p, b3s, b3i, dk, alpha_crp, alpha_crs, alpha_cri, kp, ks, ki, dz, steps_z, SIZE );
		#else
		EvolutionInCrystal( w_gpu, grid, block, Ap_gpu, As_gpu, Apw_gpu, Asw_gpu, k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu, k3p_gpu, k3s_gpu, k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu, lp, ls, vp, vs, b2p, b2s, b3p, b3s, dk, alpha_crp, alpha_crs, kp, ks, dz, steps_z, SIZE );
		#endif
		
		
		if(GDD!=0){ // adds dispersion compensation
			AddGDD<<<grid,block>>>(Asw_gpu, auxs_gpu, w_gpu, GDD, SIZE);
			CHECK(cudaDeviceSynchronize());
			cufftExecC2C(plan1D, (complex_t *)Asw_gpu, (complex_t *)As_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
			#ifdef THREE_EQS
			AddGDD<<<grid,block>>>(Aiw_gpu, auxi_gpu, w_gpu, GDD, SIZE);
			CHECK(cudaDeviceSynchronize());
			cufftExecC2C(plan1D, (complex_t *)Aiw_gpu, (complex_t *)Ai_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
			#endif
		}		
		
		if( using_phase_modulator ){ // use an intracavy phase modulator of one o more fields
			PhaseModulatorIntraCavity<<<grid,block>>>(As_gpu, auxs_gpu, mod_depth, fpm, T_gpu, SIZE);
			CHECK(cudaDeviceSynchronize());
		}
		
		if( using_SPM ){ // use an intracavy third-order material
			SelfPhaseModulation<<<grid,block>>>(As_gpu, auxs_gpu, gamma, lfo, alphafo, SIZE);
			CHECK(cudaDeviceSynchronize());
		}
		
		if (is_As_resonant){ // if As is resonant, adds phase and losses
			AddPhase<<<grid,block>>>(As_gpu, auxs_gpu, Rs, delta, nn, SIZE);
			CHECK(cudaDeviceSynchronize());
		}
		
		#ifdef THREE_EQS
		if (is_Ai_resonant){  // if Ai is resonant, adds phase and losses
			AddPhase<<<grid,block>>>(Ai_gpu, auxi_gpu, Ri, delta, nn, SIZE);
			CHECK(cudaDeviceSynchronize());
		}
		#endif
		
		#ifdef CW_OPO	// saves systematically every round trip
		if (video){  
			if (nn % 100 == 0){ // this branch is useful if the user want to save the round trips every 100 ones
				std::cout << "Saving the " << nn << "-th round trip" << std::endl;
				SaveRoundTrip<<<grid,block>>>(As_total_gpu, As_gpu, mm, extra_win, Nrts, SIZE ); // saves signal
				CHECK(cudaDeviceSynchronize());
				SaveRoundTrip<<<grid,block>>>(Ap_total_gpu, Ap_gpu, mm, extra_win, Nrts, SIZE ); // saves pump
				CHECK(cudaDeviceSynchronize());
				#ifdef THREE_EQS
				SaveRoundTrip<<<grid,block>>>(Ai_total_gpu, Ai_gpu, mm, extra_win, Nrts, SIZE ); // saves idler
				CHECK(cudaDeviceSynchronize());
				#endif
				mm += 1;
			}			
		}
		else{  // this branch is useful if the user want to save the last N_rt-Nrts round trips
			if (nn >= N_rt -Nrts){                
				SaveRoundTrip<<<grid,block>>>(As_total_gpu, As_gpu, mm, extra_win, Nrts, SIZE ); // saves signal
				CHECK(cudaDeviceSynchronize());
				SaveRoundTrip<<<grid,block>>>(Ap_total_gpu, Ap_gpu, mm, extra_win, Nrts, SIZE ); // saves pump
				CHECK(cudaDeviceSynchronize());
				#ifdef THREE_EQS
				SaveRoundTrip<<<grid,block>>>(Ai_total_gpu, Ai_gpu, mm, extra_win, Nrts, SIZE ); // saves idler
				CHECK(cudaDeviceSynchronize());
				#endif
				mm += 1;
			}
		}
		#endif
		#ifdef NS_OPO	// save the simulation in the NS regime
			SaveRoundTrip<<<grid,block>>>(Ap_total_gpu, Ap_gpu, nn, extra_win, N_rt, SIZE ); // saves signal
			CHECK(cudaDeviceSynchronize());
			SaveRoundTrip<<<grid,block>>>(As_total_gpu, As_gpu, nn, extra_win, N_rt, SIZE ); // saves pump
			CHECK(cudaDeviceSynchronize());
			#ifdef THREE_EQS
			SaveRoundTrip<<<grid,block>>>(Ai_total_gpu, Ai_gpu, nn, extra_win, N_rt, SIZE ); // saves idler
			CHECK(cudaDeviceSynchronize());
			#endif
		#endif
	}
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	// Saving results in .dat files using the function SaveFileVectorComplex() //
	
	
	// Decide whether or not save these vectors in the cuOPO.sh file
	short unsigned int save_vectors = atoi(argv[1]);
	if (save_vectors == 1){
		std::cout << "\nSaving time and frequency vectors...\n" << std::endl;
		Filename = "Tp"; SaveFileVectorReal (Tp, SIZEL, Filename+Extension);
 		Filename = "freq"; SaveFileVectorReal (Fp, SIZEL, Filename+Extension);
		Filename = "T"; SaveFileVectorReal (T, SIZE, Filename+Extension);
	}
	else{ std::cout << "\nTime and frequency were previuosly save...\n" << std::endl;
	}
	
	// Define CPU vectors to save the full simulation
	complex_t *As_total   = (complex_t*)malloc(sizeof(complex_t) * SIZEL);
	complex_t *Ap_total   = (complex_t*)malloc(sizeof(complex_t) * SIZEL);
	
	// Copy GPU -> CPU and save outputs
	CHECK(cudaMemcpy(As_total, As_total_gpu, sizeof(complex_t) * SIZEL, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Ap_total, Ap_total_gpu, sizeof(complex_t) * SIZEL, cudaMemcpyDeviceToHost));

	// Save data
	Filename = "signal_output";	SaveFileVectorComplex ( As_total, SIZEL, Filename );
	Filename = "pump_output";	SaveFileVectorComplex ( Ap_total, SIZEL, Filename );

	#ifdef THREE_EQS
	complex_t *Ai_total   = (complex_t*)malloc(sizeof(complex_t) * SIZEL);
	CHECK(cudaMemcpy(Ai_total, Ai_total_gpu, sizeof(complex_t) * SIZEL, cudaMemcpyDeviceToHost));
	Filename = "idler_output";	SaveFileVectorComplex ( Ai_total, SIZEL, Filename );		
	#endif
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////
	// Deallocating memory from CPU and GPU and destroying plans
	
	free(Tp); free(T); free(Fp); free(w); free(F); free(As);
	free(As_total); free(Ap_in); free(Ap_total);

	CHECK(cudaFree(As_gpu)); 		CHECK(cudaFree(Ap_gpu));
	CHECK(cudaFree(As_total_gpu));	CHECK(cudaFree(Ap_total_gpu));
	CHECK(cudaFree(Ap_in_gpu));	
	CHECK(cudaFree(T_gpu)); 		CHECK(cudaFree(w_gpu));
	CHECK(cudaFree(k1p_gpu));		CHECK(cudaFree(k2p_gpu));
	CHECK(cudaFree(k3p_gpu));        	CHECK(cudaFree(k4p_gpu));
	CHECK(cudaFree(k1s_gpu));        	CHECK(cudaFree(k2s_gpu));
	CHECK(cudaFree(k3s_gpu));        	CHECK(cudaFree(k4s_gpu));	
	CHECK(cudaFree(auxs_gpu));       	CHECK(cudaFree(auxp_gpu));

	#ifdef THREE_EQS
		free(Ai); free(Ai_total);
		
		CHECK(cudaFree(Ai_gpu));	CHECK(cudaFree(Ai_total_gpu));
		CHECK(cudaFree(k1i_gpu));     CHECK(cudaFree(k2i_gpu));
		CHECK(cudaFree(k3i_gpu));     CHECK(cudaFree(k4i_gpu));
		CHECK(cudaFree(auxi_gpu));
	#endif

	// Destroy CUFFT context and reset the GPU
	cufftDestroy(plan1D); 	cudaDeviceReset();    
	
	////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////	
	// Finish timing: returns the runtime simulation
	
	real_t iElaps = seconds() - iStart;
	if(iElaps>60){std::cout << "\n\n...time elapsed " <<  iElaps/60.0 << " min\n\n " << std::endl;}
	else{std::cout << "\n\n...time elapsed " <<  iElaps << " seconds\n\n " << std::endl;}

	time(&current_time);
	std::cout << ctime(&current_time) << std::endl;
	
	return 0;
}


/**
 Letter   Description  Escape-Sequence
-------------------------------------
A        Alpha        \u0391
B        Beta         \u0392
Γ        Gamma        \u0393
Δ        Delta        \u0394
Ε        Epsilon      \u0395
Ζ        Zeta         \u0396
Η        Eta          \u0397
Θ        Theta        \u0398
Ι        Iota         \u0399
Κ        Kappa        \u039A
Λ        Lambda       \u039B
Μ        Mu           \u039C
Ν        Nu           \u039D
Ξ        Xi           \u039E
Ο        Omicron      \u039F
Π        Pi           \u03A0
Ρ        Rho          \u03A1
Σ        Sigma        \u03A3
Τ        Tau          \u03A4
Υ        Upsilon      \u03A5
Φ        Phi          \u03A6
Χ        Chi          \u03A7
Ψ        Psi          \u03A8
Ω        Omega        \u03A9 
-------------------------------------
Letter   Description  Escape-Sequence
-------------------------------------
α        Alpha        \u03B1
β        Beta         \u03B2
γ        Gamma        \u03B3
δ        Delta        \u03B4
ε        Epsilon      \u03B5
ζ        Zeta         \u03B6
η        Eta          \u03B7
θ        Theta        \u03B8
ι        Iota         \u03B9
κ        Kappa        \u03BA
λ        Lambda       \u03BB
μ        Mu           \u03BC
ν        Nu           \u03BD
ξ        Xi           \u03BE
ο        Omicron      \u03BF
π        Pi           \u03C0
ρ        Rho          \u03C1
σ        Sigma        \u03C3
τ        Tau          \u03C4
υ        Upsilon      \u03C5
φ        Phi          \u03C6
χ        Chi          \u03C7
ψ        Psi          \u03C8
ω        Omega        \u03C9
-------------------------------------
*/
