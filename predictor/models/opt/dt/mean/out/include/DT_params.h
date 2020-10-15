#ifndef DT_PARAMS_H
#define DT_PARAMS_H

#define N_CLASS 2

#define N_NODES 59

#define VALUES_DIM 2

extern int children_left[N_NODES];
extern int children_right[N_NODES];
extern int feature[N_NODES];
extern int threshold[N_NODES];
extern int values[N_NODES][VALUES_DIM];
extern int target_classes[N_CLASS];

#endif