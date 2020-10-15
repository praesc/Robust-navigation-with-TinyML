#ifndef PPPARAMS_H
#define PPPARAMS_H

#ifndef N_FEATURE
#define N_FEATURE 3
#endif

#ifndef N_ORIG_FEATURE
#define N_ORIG_FEATURE 3
#endif

extern float pca_components[N_FEATURE][N_ORIG_FEATURE];

#define MINMAX_SCALING

extern float s_x[N_ORIG_FEATURE];
#endif