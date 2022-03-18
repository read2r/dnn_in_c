#include <stdio.h>
#include <math.h>
#include "activation.h"
#include "ndshape.h"
#include "ndarray.h"

void* step(void *x) {
    *(double*)x = (*(double*)x > 0) ? 1 : 0;
    return x;
}

void* sigmoid(void *x) {
    *(double*)x = 1 / (1 + exp(-(*(double*)x)));
    return x;
}

void* relu(void* x) {
    *(double*)x = *(double*)x > 0 ? *(double*)x : 0;
    return x;
}

void* exp_func(void *x) {
    *(double*)x = exp(*(double*)x);
    return x;
}

NdArray* step_function(NdArray *array) {
    NdArray_broadcast(array, step);
    return array;
}

NdArray* sigmoid_function(NdArray *array) {
    NdArray_broadcast(array, sigmoid);
    return array;
}

NdArray* relu_function(NdArray *array) {
    NdArray_broadcast(array, relu);
    return array;
}

NdArray* identity_function(NdArray *array) {
    return array;
}

NdArray* softmax(NdArray *array) {
    if(array->shape->dim == 2) {
        NdArray *temp = NdArray_transpose(array);
        NdArray *c = NdArray_max_axis(temp, 0);
        NdArray_sub(temp, c);
        NdArray_broadcast(temp, exp_func);
        NdArray *sum_axis = NdArray_sum_axis(temp, 0);
        NdArray_div(temp, sum_axis);
        NdArray *y = NdArray_transpose(temp);
        NdArray_free(&temp);
        NdArray_free(&c);
        NdArray_free(&sum_axis);
        return y;
    }

    NdArray *array_exp = NdArray_copy(array);
    double c = NdArray_max_double(array_exp);
    NdArray_sub_scalar(array_exp, c);
    NdArray_broadcast(array_exp, exp_func);
    double sum_exp = NdArray_sum_double(array_exp);
    NdArray_div_scalar(array_exp, sum_exp);
    NdArray *y = array_exp;
    return y;
}
