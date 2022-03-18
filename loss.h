#ifndef __LOSS_H__
#define __LOSS_H__
#include "ndarray.h"

double mean_squared_error(NdArray *y, NdArray *t);
double cross_entropy_error(NdArray *y, NdArray *t);

#endif
