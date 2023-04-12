/*---------------------------------------------------------------------------*/
// * This file contains functions to solve the Split-Step Fourier method (SSMF)
// * needed to calculate the electric fields evolution through the nonlinear crystal.
// * 
// * In particular, this file should be used when only three equation describes the 
// * problem, i.e., sum or difference frequency generation (SFG or DFG).
// *
// * For any specific process, please check the form of the Couple wave equations
// * in the first function called dAdz(). 
/*---------------------------------------------------------------------------*/



#ifndef _CWES3CUH
#define _CWES3CUH

#pragma once


/** Computes the nonlinear part: dA/dz=i.κ.Ax.Ay.exp(i.Δk.L) and saves the result in dAx (x represents the different fields) */
// INPUTS 
// dAp, dAs, dAi : evolved electric fields
//  Ap,  As,  Ai: electric fields
//  kp,  ks,  ki: kappas
//       dk: mismatch
//        z: crystal position

// OUTOPUT
// save in dAp, dAs, dAi the evolved electric fields
__global__ void dAdz( complex_t *dAp, complex_t *dAs,  complex_t *dAi, complex_t *Ap, complex_t *As, complex_t *Ai, real_t kp, real_t ks, real_t ki, real_t dk, real_t z )
{
	complex_t Im; Im.x = 0; Im.y = 1;
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		dAp[idx]  = Im * kp * As[idx] * Ai[idx] * CpxExp(-dk*z) ;
		dAs[idx]  = Im * ks * Ap[idx] * CpxConj(Ai[idx]) * CpxExp(+dk*z);
		dAi[idx]  = Im * ki * Ap[idx] * CpxConj(As[idx]) * CpxExp(+dk*z);
	}
	
	return ;
}


/** Computes a linear combination Ax + s.kx and saves the result in aux_x */
// INPUTS 
// auxp, auxs, auxi : auxiliary vectors
//   Ap,   As,   Ai : electric fields
//   kp,   ks,   ki : vectors for Runge-Kutta
//                 s: scalar for Runge-Kutta
//              SIZE: size vector
// OUTOPUT
// save in auxp, auxs, auxi the evolved electric fields
__global__ void LinealCombination( complex_t *auxp, complex_t *auxs, complex_t *auxi, complex_t *Ap, complex_t *As, complex_t *Ai, complex_t *kp, complex_t *ks, complex_t *ki, double s )
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		auxp[idx] = Ap[idx] + kp[idx] * s;
		auxs[idx] = As[idx] + ks[idx] * s;
		auxi[idx] = Ai[idx] + ki[idx] * s;
	}
	
	return ;
}


/** This kernel computes the final sum after appling the Rounge-Kutta algorithm */
// INPUTS 
//  Ap,  As, Ai : electric fields
//  kXp,kXs, kXi: vectors for Runge-Kutta
//       dz: step size

// OUTOPUT
// Update electric fields using Runge-Kutta after one step size, dz
__global__ void rk4(complex_t *Ap, complex_t *As,  complex_t *Ai, complex_t *k1p, complex_t *k1s, complex_t *k1i, complex_t *k2p, complex_t *k2s, complex_t *k2i, complex_t *k3p, complex_t *k3s, complex_t *k3i, complex_t *k4p, complex_t *k4s, complex_t *k4i, real_t dz )
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < SIZE){
		Ap[idx] = Ap[idx] + (k1p[idx] + 2*k2p[idx] + 2*k3p[idx] + k4p[idx]) * dz / 6;
		As[idx] = As[idx] + (k1s[idx] + 2*k2s[idx] + 2*k3s[idx] + k4s[idx]) * dz / 6;
		Ai[idx] = Ai[idx] + (k1i[idx] + 2*k2i[idx] + 2*k3i[idx] + k4i[idx]) * dz / 6;
	}
	
	return ;
}


/** Computes the linear part: Ax = Ax.exp(i.f(Ω)*z), where f(Ω) is a frequency dependant functions
 * including the group velocity and the group velocity dispersion parameters. */
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
__global__ void LinearOperator(complex_t *auxp, complex_t *auxs, complex_t *auxi, complex_t *Apw, complex_t* Asw, complex_t* Aiw, real_t *w, real_t vp, real_t vs, real_t vi, real_t b2p, real_t b2s, real_t b2i,  real_t b3p, real_t b3s, real_t b3i, real_t z)
{
	
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	real_t attenp = expf(-0.5*alpha_crp*z);
	real_t attens = expf(-0.5*alpha_crs*z);
	real_t atteni = expf(-0.5*alpha_cri*z);
	
	if (idx < SIZE){
		auxp[idx] = Apw[idx] * CpxExp(z*w[idx]*((1/vs-1/vp)+0.5*w[idx]*b2p + w[idx]*w[idx]*b3p/6));
		auxs[idx] = Asw[idx] * CpxExp(z*w[idx]*((1/vs-1/vs)+0.5*w[idx]*b2s + w[idx]*w[idx]*b3s/6));
		auxi[idx] = Aiw[idx] * CpxExp(z*w[idx]*((1/vs-1/vi)+0.5*w[idx]*b2i + w[idx]*w[idx]*b3i/6));
	}
	if (idx < SIZE){
		Apw[idx] = auxp[idx] * attenp;
		Asw[idx] = auxs[idx] * attens;
		Aiw[idx] = auxi[idx] * atteni;
	}
	
	return ;
}


/** Compute the evolution of the electric fields for a single-pass using the SSFM.
 * The nonlinear crystal is divided into NZ slices with a size of dz. 
 * The SSFM is performed with the following steps:
 * 	1 - The nonlinear part is solved using RK4 for the first semistep dz/2
 * 	2 - The linear part is solved in the frequency domain for the full step dz.
 * 	3 - Repeat 1 for dz/2. 
 * 	4-  Repeat steps 1-3 until finishing the crystal
 */
void EvolutionInCrystal( real_t *w_ext_gpu, complex_t *Ap, complex_t *As, complex_t *Ai, complex_t *Apw_gpu, complex_t *Asw_gpu, complex_t *Aiw_gpu, complex_t *k1p_gpu, complex_t *k1s_gpu, complex_t *k1i_gpu, complex_t *k2p_gpu, complex_t *k2s_gpu, complex_t *k2i_gpu, complex_t *k3p_gpu, complex_t *k3s_gpu, complex_t *k3i_gpu, complex_t *k4p_gpu, complex_t *k4s_gpu, complex_t *k4i_gpu, complex_t *auxp_gpu, complex_t *auxs_gpu, complex_t *auxi_gpu, real_t vp, real_t vs, real_t vi, real_t b2p, real_t b2s, real_t b2i, real_t b3p, real_t b3s, real_t b3i, real_t dk, real_t kp, real_t ks, real_t ki, real_t dz )
{
	// Parameters for kernels
	dim3 block(BLKX);
	dim3 grid((SIZE+BLKX-1)/BLKX);
	
	// Set plan for cuFFT //
	cufftHandle plan;
	cufftPlan1d(&plan, SIZE, CUFFT_C2C, 1);
	
	real_t z = 0;
	for (uint s = 0; s < NZ; s++){
		/* First RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, k1i_gpu, Ap, As, Ai, kp, ks, ki, dk, z );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, k2i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/4 );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k2p_gpu, k2s_gpu, k2i_gpu, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, k3i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/4 );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k3p_gpu, k3s_gpu, k3i_gpu, 1.0 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, k4i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/2 );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu, k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu, dz/2 );
		CHECK(cudaDeviceSynchronize());
		
		// Linear operator for dz
		cufftExecC2C(plan, (complex_t *)As, (complex_t *)Asw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan, (complex_t *)Ai, (complex_t *)Aiw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Aiw_gpu, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan, (complex_t *)Ap, (complex_t *)Apw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Apw_gpu, SIZE);
		CHECK(cudaDeviceSynchronize());
		LinearOperator<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Apw_gpu, Asw_gpu, Aiw_gpu, w_ext_gpu, vp, vs, vi, b2p, b2s, b2i, b3p, b3s, b3i, dz);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan, (complex_t *)Asw_gpu, (complex_t *)As, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		cufftExecC2C(plan, (complex_t *)Aiw_gpu, (complex_t *)Ai, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());		
		cufftExecC2C(plan, (complex_t *)Apw_gpu, (complex_t *)Ap, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		
		/* Second RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, k1i_gpu, Ap, As, Ai, kp, ks, ki, dk, z );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, k2i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/4 );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k2p_gpu, k2s_gpu, k2i_gpu, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, k3i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/4 );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, auxi_gpu, Ap, As, Ai, k3p_gpu, k3s_gpu, k3i_gpu, 1.0 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, k4i_gpu, Ap, As, Ai, kp, ks, ki, dk, z+dz/2 );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap, As, Ai, k1p_gpu, k1s_gpu, k1i_gpu, k2p_gpu, k2s_gpu, k2i_gpu, k3p_gpu, k3s_gpu, k3i_gpu, k4p_gpu, k4s_gpu, k4i_gpu, dz/2 );
		CHECK(cudaDeviceSynchronize());
		
		z+=dz;
	}
	
	cufftDestroy(plan);
	
	return ;
}



#endif // -> #ifdef _CWES3CUH
