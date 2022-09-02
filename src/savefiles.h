/*---------------------------------------------------------------------------*/
// * This file contains two functions that save files in .dat extension
// * 1 - SaveFileVectorReal()    : save real vectors
// * 2 - SaveFileVectorComplex() : save complex vectors
 
// Inputs:
// - Vector   : vector to save
// - N        : vector size
// - Filename : name of the saved file
/*---------------------------------------------------------------------------*/

#ifndef _SAVEFILESCUH
#define _SAVEFILESCUH

#pragma once

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

// Complex data type
using complex_t = cufftComplex;
using real_t = float;


template <class T>
void SaveFileVectorReal (T *Vector, const int N, std::string Filename){
	std::ofstream myfile;
	myfile.open(Filename);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy] << "\n";
	myfile.close();
	return;
}


template <class T>
void SaveFileVectorComplex (T *Vector, const int N, std::string Filename){
	std::ofstream myfile;
	std::string extension_r = "_r.dat", extension_i = "_i.dat";
	myfile.open(Filename+extension_r);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy].x << "\n";
	myfile.close();
	myfile.open(Filename+extension_i);
	for (int iy = 0; iy < N; iy++)
	    myfile << std::setprecision(20) << Vector[iy].y << "\n";
	myfile.close();
}


#endif // -> #ifdef _SAVEFILESCUH
