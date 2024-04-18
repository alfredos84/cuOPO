/*---------------------------------------------------------------------------*/
// * This file contains a set of functions based on the 
// * Sellmeier equations for the MgO:sPPLT nonlinear crystal and other 
// * properties of the χ⁽²⁾ material. Sellmeier equations from reference 
// * O. Gayer: Temperature and wavelength dependent refractive index 
// * equations for MgO-doped congruent and stoichiometric LiNbO3.
/*---------------------------------------------------------------------------*/

// All the functions have two input arguments:
// *     L: wavelenght in um
// *     T: temperature in degrees



#ifndef _PPLN 
#define _PPLN 

#pragma once


const real_t deff		= 25.2e-6;		// Eff. second-order susceptibility (d33) [um/V]
const real_t dQ			= 2.0*deff/PI;	// Eff. second-order susceptibility for QPM [um/V]
const real_t k			= 4.5e-6;    // thermal conductivity [W/um K]
const real_t alpha_crp	= 0.025e-4;  // pump linear absorption [1/μm]
const real_t alpha_crs	= 0.002e-4;  // signal linear absorption [1/μm]
const real_t alpha_cri	= 0.002e-4;  // idler linear absorption [1/μm]

const real_t beta_crs	= 0;         // signal 2-photons absorption [1/μm]
const real_t rho		= 0;         // walk-off angle [rad] 


/** This function returns the MgO:PPLN extraordinary refractive index */
__host__ __device__ real_t n(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;
	return sqrtf(G1+G2/(powf(L,2) - powf(G3,2))+G4/(powf(L,2) - powf(a5,2))-a6*L*L);
	
}


/** This function is an auxiliary function related with the resonances */
__host__ __device__ real_t resonances(real_t L,real_t T, int p)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;

	return G2/powf((powf(L,2) - powf(G3,2)), p) + G4/powf((powf(L,2) - powf(a5,2)), p);
	
}


/** Returns the first-order derivative of the 
 * refractive index respect to the wavelength dn/dλ. */
__host__ __device__ real_t dndl(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;

	return -L*(resonances(L, T, 2) + a6)/n(L, T);
	
}


/** Returns the second-order derivative of the
 * refractive index respect to the wavelength d²n/dλ². */
__host__ __device__ real_t d2ndl2(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;


	real_t A  = (L*dndl(L,T)/powf(n(L,T),2)-1/n(L,T))*(resonances(L, T, 2) + a6);
	real_t B  = 4*L*L/n(L,T) * resonances(L, T, 3);
	
	return A+B;
	
}


/** Returns the third-order derivative of the
 * refractive index respect to the wavelength d³n/dλ³. */
__host__ __device__ real_t d3ndl3(real_t L,real_t T)
{
	
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;

	real_t A1 = (2*dndl(L,T)+L*d2ndl2(L,T))/powf(n(L,T),2);
	real_t A2 = -2*L*powf(dndl(L,T),2)/powf(n(L,T),3);
	real_t AA = (A1 + A2)*(resonances(L,T,2)+a6);
	real_t B1 = 12*L/n(L,T);
	real_t B2 = - 4*L*L/n(L,T)*d2ndl2(L,T)*(1-1/n(L,T));
	real_t BB = (B1+B2)*resonances(L,T,3);
	real_t CC = -24*L*L*L/n(L,T)*resonances(L,T,4);

	return AA + BB + CC;	
}


/** Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). */
__host__ __device__ real_t GV(real_t L,real_t T)
{
	
	return C/(n(L,T)-L*dndl(L,T));
}


/** Returns the group-velocity β2(λ)=λ^3/(2πc²)(d²n/dλ²). */
__host__ __device__ real_t GVD(real_t L,real_t T)
{
	return powf(L,3)*d2ndl2(L, T)/(2*PI*C*C);
}


/** Returns the TOD β3(λ)=-λ^4/(4π²c³)[3.d²n/dλ² + λ.d³n/dλ³]. */
__host__ __device__ real_t TOD(real_t L,real_t T)
{
	return -powf(L,4)/(4*PI*PI*C*C*C)*(3*d2ndl2(L, T)+L*d3ndl3(L, T));
}



#endif // -> #ifdef _PPLN
