#include <stdlib.h>
#include <assert.h>
#include "grad.h"
#include "ndshape.h"
#include "ndarray.h"

NdArray* numerical_gradient(double (*func)(NdArray*), NdArray* x) {
    assert(x->datatype == DT_DOUBLE);

    double h = 1e-4;
    NdArray *grad = NdArray_new(NULL, x->shape, DT_DOUBLE);

    double* cur_x = x->data;
    double* cur_grad = grad->data;
    double fxh1, fxh2;

    for(int i = 0; i < x->shape->len; i++) {
        double temp = *cur_x;

        *cur_x = temp + h;
        fxh1 = func(x);

        *cur_x = temp - h;
        fxh2 = func(x);

        *cur_grad = (fxh1 - fxh2) / (2 * h);
        *cur_x = temp;

        cur_x++;
        cur_grad++;
    }
    return grad;
}

NdArray* gradient_descent(double (*func)(NdArray*), NdArray *init_x, double lr, int step_num) {
    NdArray *x = NdArray_copy(init_x);

    for(int i = 0; i < step_num; i++) {
        NdArray *grad = numerical_gradient(func, x);
        NdArray_mul_scalar(grad, lr);
        NdArray_sub(x, grad);
        NdArray_free(&grad);
    }

    return x;
}
