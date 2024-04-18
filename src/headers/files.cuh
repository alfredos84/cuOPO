/*---------------------------------------------------------------------------*/
// * This file contains four functions that save files in .dat extension
// * 1 - SaveFileVectorReal()       : save CPU real vectors 
// * 2 - SaveFileVectorRealGPU()    : save GPU real vectors
// * 3 - SaveFileVectorComplex()    : save CPU complex vectors
// * 4 - SaveFileVectorComplexGPU() : save GPU complex vectors

// Inputs:
// - Vector   : vector to save (stored on CPU or GPU)
// - N        : vector size
// - Filename : name of the saved file
/*---------------------------------------------------------------------------*/


#ifndef _FILESCUH
#define _FILESCUH

#pragma once


void SaveVectorReal (real_t *Vector, uint N, std::string Filename)
{
	std::ofstream myfile;
	myfile.open(Filename);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy] << "\n";
	myfile.close();
	
	return;
	
}


void SaveVectorRealGPU (real_t *Vector_gpu, uint N, std::string Filename)
{
	uint nBytes = N*sizeof(real_t);
	real_t *Vector = (real_t*)malloc(nBytes);
	CHECK(cudaMemcpy(Vector, Vector_gpu, nBytes, cudaMemcpyDeviceToHost));
	SaveVectorReal ( Vector, N, Filename );
	free(Vector);
	
	return;
	
}


void SaveVectorComplex (complex_t *Vector, uint N, std::string Filename)
{
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
	
	return;
	
}


void SaveVectorComplexGPU (complex_t *Vector_gpu, uint N, std::string Filename)
{
	uint nBytes = N*sizeof(complex_t);
	complex_t *Vector = (complex_t*)malloc(nBytes);
	CHECK(cudaMemcpy(Vector, Vector_gpu, nBytes, cudaMemcpyDeviceToHost));
	SaveVectorComplex ( Vector, N, Filename );
	free(Vector);
	
	return;
	
}


#endif // -> #ifdef _FILESCUH
