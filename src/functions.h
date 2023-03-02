/*---------------------------------------------------------------------------*/
// * This file contains a set of functions useful for the execution of the main file.
/*---------------------------------------------------------------------------*/


#ifndef _FUNCTIONSCUH
#define _FUNCTIONSCUH

#pragma once

#include <iostream>
#include <string>
#include <random>
#include <chrono>
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


// Complex data type
using complex_t = cufftComplex;
using real_t = float;




/** Sinc funcion: sin(x)/x */
__host__ __device__ real_t sinc(  real_t x  )
{
	// SINC function
	if (x == 0){return 1.0;} else{ return sinf(x)/x;}
}


//* Overloaded operators for complex numbers */
__host__ __device__ inline complex_t  operator+(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &b, const real_t &a)
{
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &a, const complex_t &b) 
{
	
	complex_t c;    
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const real_t &a, const complex_t &b)
{
	
	complex_t c;    
	c.x = a   - b.x;
	c.y =     - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &b, const real_t &a) 
{
	
	complex_t c;    
	c.x = a   - b.x;
	c.y =     - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &a, const complex_t &b) 
{
	
	complex_t c;    
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const real_t &a, const complex_t &b)
{
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &b, const real_t &a)
{
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &a, const complex_t &b)
{
	
	complex_t c;    
	c.x = a.x * b.x - a.y * b.y ;
	c.y = a.x * b.y + a.y * b.x ;
	
	return c;
}


__host__ __device__ inline complex_t  operator/(const complex_t &b, const real_t &a) 
{
	
	complex_t c;    
	c.x = b.x / a ;
	c.y = b.y / a ;
	
	return c;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

//* Complex functions //

__host__ __device__ complex_t CpxExp (real_t a)
{	
	//exp(i*a)
	complex_t b;
	b.x = cos(a) ;	b.y = sin(a) ;
	
	return b;
}


__host__ __device__ complex_t CpxConj (complex_t a)
{
	// conjugate a*
	complex_t b;
	b.x = +a.x ; b.y = -a.y ;
	
	return b;
}


__host__ __device__ real_t CpxAbs (complex_t a)
{
	// |a|
	real_t b;
	b = sqrt(a.x*a.x + a.y*a.y);
	
	return b;
}


__host__ __device__ real_t CpxAbs2 (complex_t a)
{
	// |a|^2
	real_t b;
	b = a.x*a.x + a.y*a.y;
	
	return b;
}


__global__ void ComplexProductGPU (complex_t *c, complex_t *a, complex_t *b, uint nx, uint ny)
{
	// kernel complex multiplication
	unsigned long int column = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long int row = blockIdx.y;
	
	
	if( column < nx and row < ny){
		c[row*nx+column] = a[row*nx+column] * b[row*nx+column] ;
	}
	return ;
}




/** Noise generator for initial signal/idler vectors  */
void NoiseGeneratorCPU ( complex_t *A,  unsigned int SIZE )
{
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
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
	std::cout << "Wave:                   Gaussian pulse\n" << std::endl;
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0 * exp(-(T[i]*T[i])/(2*T0*T0));
		A[i].y = 0;
	}
	
	return ;
}


/** Define an input field (usually the pump field) in the time
 * domain. This function is overloaded and its use depends on
 * the employed regime (cw, nanosecond or user-defined)*/
void input_field_T(complex_t *A, real_t A0, int SIZE )
{
	std::cout << "Wave:                   Continuous Wave\n" << std::endl;
	for (int i = 0; i < SIZE; i++){
		A[i].x = A0;
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
__global__ void CUFFTscale(complex_t *A, int SIZE, real_t s)
{
	unsigned long int idx = blockIdx.x*blockDim.x+threadIdx.x;
	
	if ( idx < SIZE){
		A[idx] = A[idx] / s;
	}
	
	return ;
}


/** Add phase (delta) and losses (R) after a single-pass through the nonlinear crystal */
__global__ void AddPhase(complex_t *A, complex_t *aux, real_t R, real_t delta, int nn, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		aux[idx] = sqrtf(R) * A[idx] * CpxExp(PI*(nn+delta));
	}
	if (idx < SIZE){
		A[idx] = aux[idx];
	}
	
	return ;
}


/** This function compensates the dispersion (GVD crystal length)
 *after a single-pass through the nonlinear crystal */
__global__ void AddGDD(complex_t *A, complex_t *aux, real_t *w, real_t GDD, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp( 0.5*w[idx]*w[idx]*GDD );
	}
	if (idx < SIZE){
		A[idx] = aux[idx];
	}
	
	return ;
}


__global__ void AddGDDTOD(complex_t *A, complex_t *aux, real_t *w, real_t GDD,  real_t TOD, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
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
__global__ void ReadPump( complex_t *Ap, complex_t *Ap_total, int N_rt, int nn, int extra_win, int SIZE )
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < (SIZE+extra_win)){
			Ap[idx] = Ap_total[idx];
		}
	}
	else if(nn > 0 && nn < (N_rt-1)){
		int aux1 = extra_win/2;
		if (idx < (SIZE+extra_win)){
			Ap[idx] = Ap_total[idx + (nn*SIZE + aux1)];
		}
	}
	else{
		if (idx < (SIZE+extra_win)){
			Ap[idx] = Ap_total[idx + (SIZE*N_rt-1)-(SIZE+extra_win)];
		}
	}
	
	return ;
}


/** Save the electric field after one round trip. */
__global__ void SaveRoundTrip( complex_t *A_total, complex_t *A, int nn, int extra_win, int N_rt, int SIZE ){
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < SIZE){
			A_total[idx] = A[idx];
		}
	}
	else if(nn > 0 && nn<(N_rt-1)){
		if (idx < SIZE){
			A_total[idx + nn*SIZE] = A[idx + extra_win/2];
		}
	}
	else{
		if (idx < SIZE){
			A_total[idx + nn*SIZE] = A[idx + extra_win];
		}
	}
	
	return ;
}


/** Applies an electro optical modulator to an electric field after one round trip. */
__global__ void PhaseModulatorIntraCavity(complex_t *A, complex_t *aux, real_t m, real_t fpm, real_t *T, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp(m*sinf(2*PI*fpm*T[idx]));
	}
	if (idx < SIZE){
		A[idx] = aux[idx];}
		
		return ;
}


/** Add self-phase modulation to an electric field after one round trip. */
__global__ void SelfPhaseModulation(complex_t *A, complex_t *aux, real_t gamma, real_t L, real_t alphaS, int SIZE)
{
	
	real_t att = expf(-0.5*alphaS*L);
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < SIZE){
		aux[idx] = A[idx] * CpxExp(gamma*L*(A[idx].x*A[idx].x+A[idx].y*A[idx].y));
	}
	if (idx < SIZE){
		A[idx] = aux[idx] * att;
	}
	
	return ;
}


#endif // -> #ifdef _FUNCTIONSCUH
