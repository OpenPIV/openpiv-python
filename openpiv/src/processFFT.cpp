/*
====================================================================================

file: processFFT.cpp -> the FFT processing function
Copyright (C) 2008  Zachary Taylor
* 
* Adapted by Davide Lasagna

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

====================================================================================
*/

#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include "setFFT.h"
#include "processFFT.h"

void processFFT (SetFFT *settingsFFT, double **out, int xStart, int yStart, double **firstFrame, double **secondFrame) {
	
	double **bRot, **aTemp, **bTemp, meanA, meanB;
	int i, j, k, indi, indj, n, p, q, indp, indq;
	double a, b, c, d;
	int ilength, xOffset, yOffset;
	
	// Interrogation length
	ilength = settingsFFT->getFFTIntLength();
	
	// Declaration of the variables pertaining to the FFTs
	fftw_plan goA;
	fftw_plan goB;
	fftw_plan goIfft;
	fftw_complex *fftaIn;
	fftw_complex *fftbIn;
	fftw_complex *ifftIn;
	fftw_complex *fftaOut;
	fftw_complex *fftbOut;
	fftw_complex *ifftOut;
	
	// Creation of dynamic memory for the temporary and rotated image b interrogation window
	bRot = new double *[(ilength + 2)];
	for (i = 0; i < (ilength + 2); i++) {
		*(bRot + i) = new double [(ilength + 2)];
	}

	aTemp = new double *[(ilength + 2)];
	for (i = 0; i < (ilength + 2); i++) {
		*(aTemp + i) = new double [(ilength + 2)];
	}

	bTemp = new double *[(ilength + 2)];
	for (i = 0; i < (ilength + 2); i++) {
		*(bTemp + i) = new double [(ilength + 2)];
	}
	
	// Creation of the temporary interrogation window
	indi = 0;
	for (i = yStart; i < (yStart + ilength); i++) {
		indj = 0;
		for (j = xStart; j < (xStart + ilength); j++) {
			*(*(aTemp + indi) + indj) = *(*(firstFrame + i) + j);
			*(*(bTemp + indi) + indj) = *(*(secondFrame + i) + j);
			indj++;
		}
		indi++;
	}
	
	// Rotation of the image b interrogation window
	indi = 0; indj = 0;

	for (i = (ilength - 1); i >= 0; i--) {
		indj = 0;
		for (j = (ilength - 1); j >= 0; j--) {
			*(*(bRot + indi) + indj) = *(*(bTemp + i) + j);
			indj++;
		}
		indi++;
	}
	
	n = 4 * ilength * ilength;
	
	// Arranging the interrogation windows into forms that the fftw3 library can use
	fftaIn = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );
	fftbIn = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );
	ifftIn = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );

	fftaOut = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );
	fftbOut = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );
	ifftOut = (fftw_complex*) fftw_malloc ( sizeof (fftw_complex) * n );

	// Calculation of the mean interrogation window intensity to be subtracted from each window
	meanA = 0.0;
	meanB = 0.0;
	k = 0;

	for (i = yStart; i < (yStart + ilength); i++) {
		for (j = xStart; j < (xStart + ilength); j++) {
			meanA = meanA + *(*(firstFrame + i) + j);
			meanB = meanB + *(*(secondFrame + i) + j);
			k++;
		}
	}

	meanA = meanA / k;
	meanB = meanB / k;
	
	xOffset = settingsFFT->getFFTxSpacing();
	yOffset = settingsFFT->getFFTySpacing();
	k = 0;

	// Arranging and zero padding the fft complex arrays
	for (p = 0; p < (2*ilength); p++) {
		indp = p - yOffset;
		for (q = 0; q < (2*ilength); q++) {
			indq = q - xOffset;
			if (p >= ilength || q >= ilength) {
				fftaIn[k][0] = 0.0;
				fftbIn[k][0] = 0.0;
				fftaIn[k][1] = 0.0;
				fftbIn[k][1] = 0.0;	
			}
			else {
				fftaIn[k][0] = *(*(aTemp + p) + q) - meanA;
				fftbIn[k][0] = *(*(bRot + p) + q) - meanB;
				fftaIn[k][1] = 0.0;
				fftbIn[k][1] = 0.0;
			}
			k++;
		}
	}
	
	// Performing forward FFT
	goA = fftw_plan_dft_2d((2*ilength),(2*ilength),fftaIn,fftaOut,FFTW_FORWARD,FFTW_ESTIMATE);
	goB = fftw_plan_dft_2d((2*ilength),(2*ilength),fftbIn,fftbOut,FFTW_FORWARD,FFTW_ESTIMATE);

	fftw_execute(goA);
	fftw_execute(goB);

	// Performing complex multiplication
	for (i = 0; i < n; i++) {
		a = fftaOut[i][0];
		b = fftaOut[i][1];
		c = fftbOut[i][0];
		d = fftbOut[i][1];
		
		ifftIn[i][0] = a * c - b * d;
		ifftIn[i][1] = b * c + a * d;
	}
	
	// Performing the inverse FFT
	goIfft = fftw_plan_dft_2d((2*ilength),(2*ilength),ifftIn,ifftOut,FFTW_BACKWARD,FFTW_ESTIMATE);

	fftw_execute(goIfft);
	
	// Outputting correlation map
	k = 0;
	for (p = 0; p < (2*ilength); p++) {
		for (q = 0; q < (2*ilength); q++) {
			*(*(out + p) + q) = ifftOut[k][0];
			k++;
		}
	}
	
	// Delete FFT variables
	fftw_free(fftaIn);
	fftw_free(fftbIn);
	fftw_free(ifftIn);

	fftw_free(fftaOut);
	fftw_free(fftbOut);
	fftw_free(ifftOut);

	fftw_destroy_plan(goA);
	fftw_destroy_plan(goB);
	fftw_destroy_plan(goIfft);
	
	// Delete dynamic memory
	for (i = 0; i < (ilength + 2); i++) {
		delete [] *(bRot + i);
	}
	delete [] bRot;

	for (i = 0; i < (ilength + 2); i++) {
		delete [] *(aTemp + i);
	}
	delete [] aTemp;

	for (i = 0; i < (ilength + 2); i++) {
		delete [] *(bTemp + i);
	}
	delete [] bTemp;
}
	
