/*---------------------------------------------------------------------------*/
// This file contains functions to solve the Split-Step Fourier method (SSMF)
// * needed to calculate the electric fields evolution through the nonlinear crystal.
// * 
// * In particular, this file should be used for any three-wave mixing problem, i.e.,
// * sum or difference frequency generation (SFG or DFG) for a single-pass 
// * (just one round trip) or inside a cavity (multiple round trips).
/*---------------------------------------------------------------------------*/


#ifndef _CWES3CUH
#define _CWES3CUH

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
#include "functions.h"


// Data types
using complex_t = cufftComplex;
using real_t = float;


/** Computes the nonlinear part: dA/dz=i.κ.Ax.Ay.exp(i.Δk.L) and saves the result in dAx (x,y are different fields) */
// INPUTS 
// dAp, dAs, dAi : evolved electric fields
//  Ap,  As,  Ai: electric fields
//  lp,  ls,  li: wavelengths
//  kp,  ks,  ki: kappas
//        z: crystal position
//     SIZE: size vector
// OUTOPUT
// save in dAp, dAs, dAi the evolved electric fields
__global__ void dAdz( complex_t *dAp, complex_t *dAs,  complex_t *dAi, complex_t *Ap, complex_t *As, complex_t *Ai, real_t lp, real_t ls, real_t li, real_t kp, real_t ks, real_t ki, real_t dk, real_t z, int SIZE )
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		dAp[idx].x = -kp*( As[idx].x*Ai[idx].y + As[idx].y*Ai[idx].x )*cosf(dk*z) + kp*( As[idx].x*Ai[idx].x - As[idx].y*Ai[idx].y )*sinf(dk*z) ;
		dAp[idx].y = +kp*( As[idx].x*Ai[idx].x - As[idx].y*Ai[idx].y )*cosf(dk*z) + kp*( As[idx].x*Ai[idx].y + As[idx].y*Ai[idx].x )*sinf(dk*z) ;
		
		dAs[idx].x = -ks*( Ap[idx].x*Ai[idx].x + Ap[idx].y*Ai[idx].y )*sinf(dk*z) - ks*( Ap[idx].y*Ai[idx].x - Ap[idx].x*Ai[idx].y )*cosf(dk*z) ;
		dAs[idx].y = +ks*( Ap[idx].x*Ai[idx].x + Ap[idx].y*Ai[idx].y )*cosf(dk*z) - ks*( Ap[idx].y*Ai[idx].x - Ap[idx].x*Ai[idx].y )*sinf(dk*z) ;
		
		dAi[idx].x = -ki*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sinf(dk*z) - ki*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cosf(dk*z) ;
		dAi[idx].y = +ki*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cosf(dk*z) - ki*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sinf(dk*z) ;		
	}
	
	return ;
}


/** Computes a linear combination Ax + s.kx and saves the result in aux_x */
// INPUTS 
// auxp, auxs, auxi : auxiliary vectors
//   Ap,   As,   Ai : electric fields
//   lp,   ls,   li : wavelengths
//   kp,   ks,   ki : vectors for Runge-Kutta
//                 s: scalar for Runge-Kutta
//              SIZE: size vector
// OUTOPUT
// save in auxp, auxs, auxi the evolved electric fields
__global__ void LinealCombination( complex_t *auxp, complex_t *auxs, complex_t *auxi, complex_t *Ap, complex_t *As, complex_t *Ai, complex_t *kp, complex_t *ks, complex_t *ki, double s, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		auxp[idx].x = Ap[idx].x + kp[idx].x * s;
		auxp[idx].y = Ap[idx].y + kp[idx].y * s;
		auxs[idx].x = As[idx].x + ks[idx].x * s;
		auxs[idx].y = As[idx].y + ks[idx].y * s;
		auxi[idx].x = Ai[idx].x + ki[idx].x * s;
		auxi[idx].y = Ai[idx].y + ki[idx].y * s;		
	}
	
	return ;
}


/** This kernel computes the final sum after appling the Rounge-Kutta algorithm */
// INPUTS 
//  Ap,  As, Ai : electric fields
//  kXp,kXs, kXi: vectors for Runge-Kutta
//       dz: step size
//     SIZE: size vector
// OUTOPUT
// Update electric fields using Runge-Kutta after one step size, dz
__global__ void rk4(complex_t *Ap, complex_t *As,  complex_t *Ai, complex_t *k1p, complex_t *k1s, complex_t *k1i, complex_t *k2p, complex_t *k2s, complex_t *k2i, complex_t *k3p, complex_t *k3s, complex_t *k3i, complex_t *k4p, complex_t *k4s, complex_t *k4i, real_t dz, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		Ap[idx].x = Ap[idx].x + (k1p[idx].x + 2*k2p[idx].x + 2*k3p[idx].x + k4p[idx].x) * dz / 6;
		Ap[idx].y = Ap[idx].y + (k1p[idx].y + 2*k2p[idx].y + 2*k3p[idx].y + k4p[idx].y) * dz / 6;
		As[idx].x = As[idx].x + (k1s[idx].x + 2*k2s[idx].x + 2*k3s[idx].x + k4s[idx].x) * dz / 6;
		As[idx].y = As[idx].y + (k1s[idx].y + 2*k2s[idx].y + 2*k3s[idx].y + k4s[idx].y) * dz / 6;
		Ai[idx].x = Ai[idx].x + (k1i[idx].x + 2*k2i[idx].x + 2*k3i[idx].x + k4i[idx].x) * dz / 6;
		Ai[idx].y = Ai[idx].y + (k1i[idx].y + 2*k2i[idx].y + 2*k3i[idx].y + k4i[idx].y) * dz / 6;		
	}
	
	return ;
}


/** Computes the linear part: Ax = Ax.exp(i.f(Ω)*z), where f(Ω) is a frequency dependant functions
 including the group velocity and the group velocity dispersion parameters. */
// INPUTS 
//     auxp,  auxs,  auxi : auxiliary vectors
//      Apw,   Asw,   Aiw : electric fields in frequency domain
//                      w : angular frequency 
//       lp,   ls,     li : wavelengths
//       vp,   vs,     vi : group-velocity
//      b2p,  b2s,    bsi : group-velocity dispersion
// alphap, alphas, alphai : linear absorpion
//           SIZE: size vector
//              z: crystal position
// OUTOPUT
// save in auxp, auxs the electric fields after appling dispersion
__global__ void LinearOperator(complex_t *auxp, complex_t *auxs, complex_t *auxi, complex_t *Apw, complex_t* Asw, complex_t* Aiw, real_t *w, real_t lp, real_t ls, real_t li, real_t vp, real_t vs, real_t vi, real_t b2p, real_t b2s, real_t b2i,  real_t b3p, real_t b3s, real_t b3i, real_t alphap, real_t alphas, real_t alphai, int SIZE, real_t z)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	real_t attenp = expf(-0.5*alphap*z);
	real_t attens = expf(-0.5*alphas*z);
	real_t atteni = expf(-0.5*alphai*z);
	
	if (idx < SIZE){		
		auxp[idx].x = Apw[idx].x * cosf(z*w[idx]*((1/vs-1/vp)+0.5*w[idx]*b2p + w[idx]*w[idx]*b3p/6)) - Apw[idx].y * sinf(z*w[idx]*((1/vs-1/vp)+0.5*w[idx]*b2p + w[idx]*w[idx]*b3p/6));
		auxp[idx].y = Apw[idx].y * cosf(z*w[idx]*((1/vs-1/vp)+0.5*w[idx]*b2p + w[idx]*w[idx]*b3p/6)) + Apw[idx].x * sinf(z*w[idx]*((1/vs-1/vp)+0.5*w[idx]*b2p + w[idx]*w[idx]*b3p/6));
		auxs[idx].x = Asw[idx].x * cosf(z*w[idx]*((1/vs-1/vs)+0.5*w[idx]*b2s + w[idx]*w[idx]*b3s/6)) - Asw[idx].y * sinf(z*w[idx]*((1/vs-1/vs)+0.5*w[idx]*b2s + w[idx]*w[idx]*b3s/6));
		auxs[idx].y = Asw[idx].y * cosf(z*w[idx]*((1/vs-1/vs)+0.5*w[idx]*b2s + w[idx]*w[idx]*b3i/6)) + Asw[idx].x * sinf(z*w[idx]*((1/vs-1/vs)+0.5*w[idx]*b2s + w[idx]*w[idx]*b3s/6));
		auxi[idx].x = Aiw[idx].x * cosf(z*w[idx]*((1/vs-1/vi)+0.5*w[idx]*b2i + w[idx]*w[idx]*b3i/6)) - Aiw[idx].y * sinf(z*w[idx]*((1/vs-1/vi)+0.5*w[idx]*b2i + w[idx]*w[idx]*b3i/6));
		auxi[idx].y = Aiw[idx].y * cosf(z*w[idx]*((1/vs-1/vi)+0.5*w[idx]*b2i + w[idx]*w[idx]*b3i/6)) + Aiw[idx].x * sinf(z*w[idx]*((1/vs-1/vi)+0.5*w[idx]*b2i + w[idx]*w[idx]*b3i/6));
	}
	if (idx < SIZE){
		Apw[idx].x = auxp[idx].x * attenp;
		Apw[idx].y = auxp[idx].y * attenp;
		Asw[idx].x = auxs[idx].x * attens;
		Asw[idx].y = auxs[idx].y * attens;
		Aiw[idx].x = auxi[idx].x * atteni;
		Aiw[idx].y = auxi[idx].y * atteni;
	}
	
	return ;
}


/** Compute the evolution of the electric fields for a single-pass using the SSFM.
 * The nonlinear crystal is divided into steps_z slices with a size of dz. 
 * The SSFM is performed with the following steps:
 * 	1 - The nonlinear part is solved using RK4 for the first semistep dz/2
 * 	2 - The linear part is solved in the frequency domain for the full step dz.
 * 	3 - Repeat 1 for dz/2. 
 * 	4-  Repeat steps 1-3 until finishing the crystal
 */
void EvolutionInCrystal( real_t *w_ext_gpu, dim3 grid, dim3 block, complex_t *Ap, complex_t *As, complex_t *Ai, complex_t *Apw_gpu, complex_t *Asw_gpu, complex_t *Aiw_gpu, complex_t *k1p_gpu, complex_t *k1s_gpu, complex_t *k1i_gpu, complex_t *k2p_gpu, complex_t *k2s_gpu, complex_t *k2i_gpu, complex_t *k3p_gpu, complex_t *k3s_gpu, complex_t *k3i_gpu, complex_t *k4p_gpu, complex_t *k4s_gpu, complex_t *k4i_gpu, complex_t *auxp_gpu, complex_t *auxs_gpu, complex_t *auxi_gpu, real_t lp, real_t ls, real_t li, real_t vp, real_t vs, real_t vi, real_t b2p, real_t b2s, real_t b2i, real_t b3p, real_t b3s, real_t b3i, real_t dk, real_t alphap, real_t alphas, real_t alphai, real_t kp, real_t ks, real_t ki, real_t dz, int steps_z, int SIZE )
{
	
	// Set plan for cuFFT 1D and 2D//
	cufftHandle plan1D;
	cufftPlan1d(&plan1D, SIZE, CUFFT_C2C, 1);
	
	real_t z = 0;
	for (int s = 0; s < steps_z; s++){
		/* First RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, k1i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, k2i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k2p_gpu, k2s_gpu, k2i_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, k3i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k3p_gpu, k3s_gpu, k3i_gpu, 1.0, SIZE );		
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, k4i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/2, SIZE );		
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu, k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu, dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());

		// Linear operator for dz
		cufftExecC2C(plan1D, (complex_t *)As, (complex_t *)Asw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan1D, (complex_t *)Ai, (complex_t *)Aiw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Aiw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan1D, (complex_t *)Ap, (complex_t *)Apw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Apw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		LinearOperator<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Apw_gpu, Asw_gpu, Aiw_gpu, w_ext_gpu, lp, ls, li, vp, vs, vi, b2p, b2s, b2i, b3p, b3s, b3i, alphap, alphas, alphai, SIZE, dz);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan1D, (complex_t *)Asw_gpu, (complex_t *)As, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan1D, (complex_t *)Aiw_gpu, (complex_t *)Ai, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());		
		cufftExecC2C(plan1D, (complex_t *)Apw_gpu, (complex_t *)Ap, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		
		/* Second RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, k1i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, k2i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k2p_gpu, k2s_gpu, k2i_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, k3i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k3p_gpu, k3s_gpu, k3i_gpu, 1.0, SIZE );		
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, k4i_gpu, Ap, As, Ai, lp, ls, li, kp, ks, ki, dk, z+dz/2, SIZE );		
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu, k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu, dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		
		z+=dz;
	}
	
	cufftDestroy(plan1D);
	
	return ;
}



#endif // -> #ifdef _CWES3CUH
