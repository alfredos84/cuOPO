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

const real_t deff		= 11.00e-6;			// Eff. second-order susceptibility (d33) [um/V]
const real_t dQ			= 2.0*deff/PI;		// Eff. second-order susceptibility for QPM [um/V]
const real_t k			= 8.4e-6;    // thermal conductivity [W/um K]

const real_t alpha_crp	= 0.025e-4;  // pump linear absorption [1/μm]
const real_t alpha_crs	= 0.002e-4;  // signal linear absorption [1/μm]
const real_t alpha_cri	= 0.002e-4;  // idler linear absorption [1/μm]
const real_t beta_crs   = 5e-5;      // signal 2-photons absorption [μm/W]
const real_t rho        = 0;         // walk-off angle [rad] 


/** This function returns the MgO:sPPLT extraordinary refractive index */
__host__ __device__ real_t n(real_t L,real_t T)
{
	
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


/** This function is an auxiliary function related with the resonances */
__host__ __device__ real_t resonances(real_t L,real_t T, int p)
{
	
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

	return (B+b)/powf((powf(L,2) - powf((C+c),2)), p) + E/powf((powf(L,2) - powf(F,2)), p) + G/powf((powf(L,2) - powf(H,2)), p);
}


/** Returns the first-order derivative of the
 * refractive index respect to the wavelength dn/dλ. */
__host__ __device__ real_t dndl(real_t L,real_t T)
{
	
	real_t B =  0.007294;
	real_t C =  0.185087;
	real_t D =  -0.02357;
	real_t E =  0.073423;
	real_t F =  0.199595;
	real_t G =  0.001;
	real_t H =  7.99724;
	real_t b =  3.483933e-8 * pow(T + 273.15,2);
	real_t c =  1.607839e-8 * pow(T + 273.15,2);
	
	return -L/n(L,T)*( resonances(L,T,2) - D );
}


/** Returns the second-order derivative of the
 * refractive index respect to the wavelength d²n/dλ². */
__host__ __device__ real_t d2ndl2(real_t L,real_t T)
{
	
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
 

	return (L/powf(n(L,T),2)*dndl(L,T)-1/n(L,T))*(resonances(L,T,2)-D) + 4*L*L/n(L,T)*resonances(L,T,3);	
}


/** Returns the third-order derivative of the
 * refractive index respect to the wavelength d³n/dλ³. */
 __host__ __device__ real_t d3ndl3(real_t L,real_t T)
 {
	 
	real_t A =  4.502483;
	real_t B =  0.007294;
	real_t C =  0.185087;
	real_t D =  -0.02357;
	real_t E =  0.073423;
	real_t F =  0.199595;
	real_t G =  0.001;
	real_t H =  7.99724;
	real_t b =  3.483933e-8 * powf(T + 273.15,2);
	real_t c =  1.607839e-8 * powf(T + 273.15,2);
 
	real_t A1 = (2*dndl(L,T)+L*d2ndl2(L,T))/powf(n(L,T),2);
	real_t A2 = -2*L*powf(dndl(L,T),2)/powf(n(L,T),3);
	real_t AA = (A1 + A2)*(resonances(L,T,2)-D);
	real_t B1 = 12*L/n(L,T);
	real_t B2 = -4*L*L/n(L,T)*d2ndl2(L,T)*(1-1/n(L,T));
	real_t BB = (B1+B2)*resonances(L,T,3);
	real_t CC = -24*L*L*L/n(L,T)*resonances(L,T,4);
 
	 return AA + BB + CC;	
 }

/** Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). */
__host__ __device__ real_t GV(real_t L,real_t T)
{
	return C/(n(L,T)-L*dndl(L,T));
}


/** Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²). */
__host__ __device__ real_t GVD(real_t L,real_t T)
{
	return pow(L,3)*d2ndl2(L, T)/(2*PI*C*C);
}


/** Returns the TOD β3(λ)=-λ^4/(4π²c³)[3.d²n/dλ² + λ.d³n/dλ³]. */
__host__ __device__ real_t TOD(real_t L,real_t T)
{
	return -powf(L,4)/(4*PI*PI*C*C*C)*(3*d2ndl2(L, T)+L*d3ndl3(L, T));
}

#endif // -> #ifdef _SPPLT