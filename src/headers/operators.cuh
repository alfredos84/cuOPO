/*---------------------------------------------------------------------------*/
// * This file contains a set of overloaded operators to deal with complex numbers.
/*---------------------------------------------------------------------------*/


#ifndef _OPERATORSCUH
#define _OPERATORSCUH

#pragma once

/** Sinc funcion: sin(x)/x */
__host__ __device__ real_t sinc(  real_t x  )
{
	// SINC function
	if (x == 0){return 1.0;} else{ return sinf(x)/x;}
}

/////////////////////////////////////     OPERATORS     ////////////////////////////////////////
__host__ __device__ inline complex_t  operator+(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator+(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   - b.x;
	c.y =     - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x =  b.x - a ;
	c.y =  b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator-(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t  operator*(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x * b.x - a.y * b.y ;
	c.y = a.x * b.y + a.y * b.x ;
	
	return c;
}


__host__ __device__ inline complex_t  operator/(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = b.x / a ;
	c.y = b.y / a ;
	
	return c;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

//* Complex exponential e^(i*a) */
__host__ __device__ complex_t CpxExp (real_t a)
{
	complex_t b;
	b.x = cos(a) ;	b.y = sin(a) ;
	
	return b;
}


//* Complex conjugate */
__host__ __device__ complex_t CpxConj (complex_t a)
{
	complex_t b;
	b.x = +a.x ; b.y = -a.y ;
	
	return b;
}


//* Complex absolute value  */
__host__ __device__ real_t CpxAbs (complex_t a)
{
	real_t b;
	b = sqrt(a.x*a.x + a.y*a.y);
	
	return b;
}


//* Complex square absolute value */
__host__ __device__ real_t CpxAbs2 (complex_t a)
{
	real_t b;
	b = a.x*a.x + a.y*a.y;
	
	return b;
}


#endif // -> #ifdef _OPERATORSCUH
