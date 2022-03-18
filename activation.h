#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
#include "ndarray.h"

NdArray* step_function(NdArray *array);
NdArray* sigmoid_function(NdArray *array);
NdArray* relu_function(NdArray *array);
NdArray* identity_function(NdArray *array);
NdArray* softmax(NdArray *array);

#endif
