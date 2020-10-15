#ifndef SVM_PARAMS_H
#define SVM_PARAMS_H

#define N_CLASS 2

#define WEIGTH_DIM 1

#ifndef N_FEATURE
#define N_FEATURE 143
#endif

extern float support_vectors[WEIGTH_DIM][N_FEATURE];
extern float bias[WEIGTH_DIM];

#endif