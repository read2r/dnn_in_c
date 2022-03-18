#ifndef __GRAD_H__
#define __GRAD_H__
#include "ndarray.h"
#include "ndshape.h"

NdArray* numerical_gradient(double (*func)(NdArray*), NdArray* x);
NdArray* gradient_descent(double (*func)(NdArray*), NdArray *init_x, double lr, int step_num);

#endif
