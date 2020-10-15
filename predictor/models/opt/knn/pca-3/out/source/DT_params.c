#include "DT_params.h"
int children_left[N_NODES] = { 1,  -1,  3,  -1,  -1};
int children_right[N_NODES] = { 2,  -1,  4,  -1,  -1};
int feature[N_NODES] = { 1,  -2,  126,  -2,  -2};
float threshold[N_NODES] = { 0.535733550786972,  -2.0,  -1.0535374879837036,  -2.0,  -2.0};
int values[N_NODES][VALUES_DIM] = { { 502, 498 },
			{ 499, 0 },
			{ 3, 498 },
			{ 3, 0 },
			{ 0, 498 } };
int target_classes[N_CLASS] = { 0,  1};
