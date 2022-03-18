#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "loss.h"
#include "ndshape.h"
#include "ndarray.h"

void* my_log(void *x) {
    *(double*)x = log(*(double*)x);
    return x;
}

double mean_squared_error(NdArray *y, NdArray *t) {
    double result;
    NdArray *temp = NdArray_copy(y);
    NdArray_sub(temp, t);
    NdArray_mul(temp, temp);
    result = NdArray_sum_double(temp) * 0.5;
    NdArray_free(&temp);
    return result;
}

/*
double cross_entropy_error(NdArray *y, NdArray *t) {
    double result;
    NdArray *temp = NdArray_copy(y);
    NdArray_add_scalar(temp, 1e-7);
    NdArray_broadcast(temp, my_log);
    NdArray_mul(temp, t);
    result = -(NdArray_sum_double(temp));
    return result;
}
*/

double cross_entropy_error(NdArray *y, NdArray *t) {
    double result;
    NdArray *temp = NdArray_copy(y);
    NdArray_broadcast(temp, my_log);
    NdArray_mul(temp, t);
    result = -(NdArray_sum_double(temp)) / y->shape->arr[0];
    return result;
}
