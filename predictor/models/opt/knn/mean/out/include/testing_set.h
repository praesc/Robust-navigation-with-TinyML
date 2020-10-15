#ifndef TESTINGSET_H
#define TESTINGSET_H

#define N_TEST 4980

#ifndef N_FEATURE
#define N_FEATURE 3
#endif

#ifndef N_ORIG_FEATURE
#define N_ORIG_FEATURE 3
#endif

extern int y_test[N_TEST];
extern int X_test[N_TEST][N_FEATURE];
#endif