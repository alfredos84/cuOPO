/*---------------------------------------------------------------------------*/
// * This file contains a set of functions based on the 
// * Sellmeier equations for the MgO:sPPLT nonlinear crystal and other 
// * properties of the χ⁽²⁾ material. Sellmeier equations from reference 
// * Bruner et. al.: Temperature-dependent Sellmeier equation for the 
// * refractive index of stoichiometric lithium tantalate.
/*---------------------------------------------------------------------------*/

// All the functions have two input arguments:
// *     L: wavelenght in um
// *     T: temperature in degrees


#ifndef _SPPLT
#define _SPPLT

#pragma once


const real_t deff		= 10.00e-6;  // Effective second-order susceptibility [um/V]
const real_t k		= 8.4e-6;    // thermal conductivity [W/um K]

const real_t alpha_crp  = 0.17e-6;   // pump linear absorption [1/μm]
const real_t alpha_crs  = 1.57e-6;   // signal linear absorption [1/μm]
const real_t alpha_cri  = 1.57e-6;   // idler linear absorption [1/μm]
const real_t beta_crs   = 5e-5;      // signal 2-photons absorption [μm/W]
const real_t rho        = 0;         // walk-off angle [rad] 


/** This function returns the MgO:sPPLT extraordinary refractive index */

__host__ __device__ real_t n(real_t L,real_t T){
	
	real_t A =  4.502483;
	real_t B =  0.007294;
	real_t C =  0.185087;
	real_t D =  -0.02357;
	real_t E =  0.073423;
	real_t F =  0.199595;
	real_t G =  0.001;
	real_t H =  7.99724;
	real_t b =  3.483933e-8 * pow(T + 273.15,2);
	real_t c =  1.607839e-8 * pow(T + 273.15,2);
	
	return sqrt( A + (B+b)/(pow(L,2)-pow((C+c),2)) + E/(pow(L,2)-pow(F,2)) + G/(pow(L,2)-pow(H,2))+ D*pow(L,2));
	
}


/** Returns the first-order derivative of the
 * refractive index respect to the wavelength dn/dλ. */
__host__ __device__ real_t dndl(real_t L,real_t T){
	
	real_t B =  0.007294;
	real_t C =  0.185087;
	real_t D =  -0.02357;
	real_t E =  0.073423;
	real_t F =  0.199595;
	real_t G =  0.001;
	real_t H =  7.99724;
	real_t b =  3.483933e-8 * pow(T + 273.15,2);
	real_t c =  1.607839e-8 * pow(T + 273.15,2);
	
	return -L/n(L, T)*( (B+b)/pow(pow(L,2)-pow((C+c),2),2) + E/pow((pow(L,2)-pow(F,2)),2) + G/pow((pow(L,2)-pow(H,2)),2) - D );
	
}


/** Returns the second-order derivative of the
 * refractive index respect to the wavelength d²n/dλ². */
__host__ __device__ real_t d2ndl2(real_t L,real_t T){
	
	real_t B =  0.007294;
	real_t C =  0.185087;
	real_t E =  0.073423;
	real_t F =  0.199595;
	real_t G =  0.001;
	real_t H =  7.99724;
	real_t b =  3.483933e-8 * pow(T + 273.15,2);
	real_t c =  1.607839e-8 * pow(T + 273.15,2);
	real_t S1 = dndl(L, T)/L;
	real_t S2 = 4*pow(L,2)/n(L,T)*((B+b)/pow(pow(L,2)-pow((C+c),2),3)+E/pow((pow(L,2)-pow(F,2)),3)+G/pow((pow(L,2)-pow(H,2)),3));
	
	return S1+S2;
	
}


/** Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). */
__host__ __device__ real_t group_vel(real_t L,real_t T){
	return C/(n(L,T)-L*dndl(L,T));
}


/** Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²). */
__host__ __device__ real_t gvd(real_t L,real_t T){
	return pow(L,3)*d2ndl2(L, T)/(2*PI*C*C);
}


#endif // -> #ifdef _SPPLT
