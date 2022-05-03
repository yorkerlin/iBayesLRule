#include <Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>
#include "mex.h"
using namespace Eigen;
typedef Map<MatrixXd> MexMat;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
 	if ( nrhs != 2 ) {
  		mexErrMsgTxt("Needs 2 arguments -- alphas (NxD), samples (NxD)");
		return;
  	}

	int N = mxGetM( prhs[0] );
	int D = mxGetN( prhs[0] );
	int N2 = mxGetM( prhs[1] );
	int D2 = mxGetN( prhs[1] );

    if (D != 1 || N != 1) {
        if (D != D2 || N != N2 ) {
            mexErrMsgTxt("alphas do not match samples!");
            return;
        }
    }

    MexMat alphas ( mxGetPr(prhs[0]), N, D);
	MexMat samples ( mxGetPr(prhs[1]), N2, D2);
 	// Create output matrix (using Matlab's alloc)
	plhs[0] = mxCreateDoubleMatrix(N2, D2, mxREAL);
  	MexMat Res ( mxGetPr(plhs[0]), N2, D2);

    for (int i=0; i<Res.rows(); i++) {
        for (int j=0; j<Res.cols(); j++) {
            if (N==1 && D==1)
                Res(i,j) = numext::gamma_sample_der_alpha(alphas(0,0), samples(i,j));
            else
                Res(i,j) = numext::gamma_sample_der_alpha(alphas(i,j), samples(i,j));
        }
    }
}

