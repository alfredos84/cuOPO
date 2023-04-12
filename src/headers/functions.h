/*---------------------------------------------------------------------------*/
// * This file contains a set of functions useful for the execution of the main file.
/*---------------------------------------------------------------------------*/


#ifndef _FUNCTIONSCUH
#define _FUNCTIONSCUH

#pragma once


/** Noise generator for initial signal/idler vectors  */
void NoiseGeneratorCPU ( complex_t *A,  uint SIZE )
{
	uint seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<real_t> distribution(0.0,1.0e-15);
	
	real_t nsx, nsy;    
	for (int i=0; i<SIZE; ++i) {
		nsx = distribution(generator); A[i].x = nsx;
		nsy = distribution(generator); A[i].y = nsy;
	}
	
	return ;
}


/** Define an input field (usually the pump field) in the time
 * domain. This function is overloaded and its use depends on
 * the employed regime (cw, nanosecond or user-defined)*/
void input_field_T(complex_t *A, real_t A0, real_t *T, real_t T0, int SIZE)
{
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0 * exp(-(T[i]*T[i])/(2*T0*T0)); // Gaussian pulse
		A[i].y = 0;
	}
	
	return ;
}


/** Define an input field (usually the pump field) in the time
 * domain. This function is overloaded and its use depends on
 * the employed regime (cw, nanosecond or user-defined)*/
void input_field_T(complex_t *A, real_t A0, int SIZE )
{
	
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0; // cw field
		A[i].y = 0;
	}
	
	return ;
}


/** Linear spacing for time vectors */
void linspace( real_t *T, int SIZE, real_t xmin, real_t xmax)
{
	for (int i = 0; i < SIZE; i++)
		T[i] = xmin + i * (xmax - xmin)/(SIZE-1);
	
	return ;
}


/** Initializes the frequency vectors */
void inic_vector_F(real_t *F, int SIZE, real_t DF)
{
	for (int i = 0; i < SIZE; i++){
		F[i] = i * DF - SIZE* DF/2.0;
	}
	
	return ;
}

/** Flips a vector for Fourier transforms */
void fftshift( real_t *W_flip, real_t *W, int SIZE )
{
	int i, c = SIZE/2;
	for ( i = 0; i < SIZE/2; i++ ){
		W_flip[i+c] = W[i];
		W_flip[i]   = W[i+c];
	}
	
	return ;
}


/** Scales a vector after Fourier transforms (CUFFT_INVERSE mode) */
__global__ void CUFFTscale(complex_t *A, uint SIZE)
{
	uint idx = blockIdx.x*blockDim.x+threadIdx.x;
	real_t s = SIZE;
	if ( idx < SIZE){
		A[idx] = A[idx] / s;
	}
	
	return ;
}


/** Add phase (delta) and losses (R) after a single-pass */
__global__ void AddPhase(complex_t *A, complex_t *aux, real_t R, real_t delta, uint nn)
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		aux[idx] = sqrtf(R) * A[idx] * CpxExp(PI*(nn+delta));
	}
	if (idx < SIZE){
		A[idx] = aux[idx];
	}
	
	return ;
}


/** This function compensates the GVD after a single-pass */
__global__ void AddGDD(complex_t *A, complex_t *aux, real_t *w, real_t GDD)
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp( 0.5*w[idx]*w[idx]*GDD );
	}
	if (idx < SIZE){
		A[idx] = aux[idx];
	}
	
	return ;
}


/** This function compensates the TOD after a single-pass */
__global__ void AddGDDTOD(complex_t *A, complex_t *aux, real_t *w, real_t GDD,  real_t TOD)
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp(w[idx]*(0.5*w[idx]*GDD + w[idx]*w[idx]*TOD/6));
	}
	if (idx < SIZE){
		A[idx] = aux[idx];
	}	
	
	return ;
}


/** Reads a large vector where the input pump is stored.
 * For nanosecond regime the input pulse is divided into hundres of round trips. */
__global__ void ReadPump( complex_t *Ap, complex_t *Ap_total, int N_rt, uint nn )
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < SIZE){
			Ap[idx] = Ap_total[idx];
		}
	}
	else if(nn > 0 && nn < (N_rt-1)){
		if (idx < SIZE){
			Ap[idx] = Ap_total[idx + nn*SIZE];
		}
	}
	else{
		if (idx < (SIZE)){
			Ap[idx] = Ap_total[idx + (SIZE*N_rt-1)-(SIZE)];
		}
	}
	
	return ;
}


/** Save the electric field after one round trip. */
__global__ void SaveRoundTrip( complex_t *A_total, complex_t *A, uint nn, uint N_rt ){
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < SIZE){
			A_total[idx] = A[idx];
		}
	}
	else if(nn > 0 && nn<(N_rt-1)){
		if (idx < SIZE){
			A_total[idx + nn*SIZE] = A[idx];
		}
	}
	else{
		if (idx < SIZE){
			A_total[idx + nn*SIZE] = A[idx];
		}
	}
	
	return ;
}


/** Applies an electro optical modulator to an electric field after one round trip. */
__global__ void PhaseModulatorIntraCavity(complex_t *A, complex_t *aux, real_t m, real_t fpm, real_t *T)
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp(m*sinf(2*PI*fpm*T[idx]));
	}
	if (idx < SIZE){
		A[idx] = aux[idx];}
		
	return ;
}


#endif // -> #ifdef _FUNCTIONSCUH
