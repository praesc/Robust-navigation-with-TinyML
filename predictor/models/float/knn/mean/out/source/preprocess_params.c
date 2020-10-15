#include "PPParams.h"
float pca_components[N_FEATURE][N_ORIG_FEATURE] = { { 1.0, 0.0, 0.0 },
			{ 0.0, 1.0, 0.0 },
			{ 0.0, 0.0, 1.0 } };

#define MINMAX_SCALING

float s_x[N_ORIG_FEATURE] = { 0.004414803682134135,  0.004552634023971012,  0.004345012480354997};
