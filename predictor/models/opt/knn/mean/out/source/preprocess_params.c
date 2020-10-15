#include "PPParams.h"
float pca_components[N_FEATURE][N_ORIG_FEATURE] = { { 1.0, 0.0, 0.0 },
			{ 0.0, 1.0, 0.0 },
			{ 0.0, 0.0, 1.0 } };

#define STANDARD_SCALING

int s_x[N_ORIG_FEATURE] = { 69,  60,  83};
int u_x[N_ORIG_FEATURE] = { 145,  174,  131};
