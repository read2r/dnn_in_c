#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>
#include "ndarray.h"
#include "ndshape.h"
#include "loss.h"
#include "grad.h"
#include "activation.h"

int AND(double x1, double x2) {
    double x[] = { x1, x2 };
    double w[] = { 0.5, 0.5 };
    double b = -0.7;

    NdShape *shape_x = NdShape_new(1, 2);
    NdShape *shape_w = NdShape_new(1, 2);
    NdArray *array_x = NdArray_new(x, shape_x, DT_DOUBLE);
    NdArray *array_w = NdArray_new(w, shape_w, DT_DOUBLE);
    NdArray *result = NdArray_copy(array_w);

    NdArray_mul(result, array_x);
    double temp = NdArray_sum_double(result) + b;

    NdShape_free(&shape_x);
    NdShape_free(&shape_w);
    NdArray_free(&array_x);
    NdArray_free(&array_w);
    NdArray_free(&result);

    if (temp <= 0) {
        return 0;
    } else {
        return 1;
    }
}

int NAND(double x1, double x2) {
    double x[] = { x1, x2 };
    double w[] = { -0.5, -0.5 };
    double b = 0.7;

    NdShape *shape_x = NdShape_new(1, 2);
    NdShape *shape_w = NdShape_new(1, 2);
    NdArray *array_x = NdArray_new(x, shape_x, DT_DOUBLE);
    NdArray *array_w = NdArray_new(w, shape_w, DT_DOUBLE);
    NdArray *result = NdArray_copy(array_w);

    NdArray_mul(result, array_x);
    double temp = NdArray_sum_double(result) + b;

    NdShape_free(&shape_x);
    NdShape_free(&shape_w);
    NdArray_free(&array_x);
    NdArray_free(&array_w);
    NdArray_free(&result);

    if (temp <= 0) {
        return 0;
    } else {
        return 1;
    }
}

int OR(double x1, double x2) {
    double x[] = { x1, x2 };
    double w[] = { 0.5, 0.5 };
    double b = -0.2;

    NdShape *shape_x = NdShape_new(1, 2);
    NdShape *shape_w = NdShape_new(1, 2);
    NdArray *array_x = NdArray_new(x, shape_x, DT_DOUBLE);
    NdArray *array_w = NdArray_new(w, shape_w, DT_DOUBLE);
    NdArray *result = NdArray_copy(array_w);

    NdArray_mul(result, array_x);
    double temp = NdArray_sum_double(result) + b;

    NdShape_free(&shape_x);
    NdShape_free(&shape_w);
    NdArray_free(&array_x);
    NdArray_free(&array_w);
    NdArray_free(&result);

    if (temp <= 0) {
        return 0;
    } else {
        return 1;
    }
}

int XOR(double x1, double x2) {
    double s1 = NAND(x1, x2);
    double s2 = OR(x1, x2);
    double y = AND(s1, s2);
    return y;
}

void ex_perceptron() {
    printf("AND\n");
    printf("(%d, %d) = %d\n", 0, 0, AND(0, 0));
    printf("(%d, %d) = %d\n", 0, 1, AND(0, 1));
    printf("(%d, %d) = %d\n", 1, 0, AND(1, 0));
    printf("(%d, %d) = %d\n\n", 1, 1, AND(1, 1));

    printf("NAND\n");
    printf("(%d, %d) = %d\n", 0, 0, NAND(0, 0));
    printf("(%d, %d) = %d\n", 0, 1, NAND(0, 1));
    printf("(%d, %d) = %d\n", 1, 0, NAND(1, 0));
    printf("(%d, %d) = %d\n\n", 1, 1, NAND(1, 1));

    printf("OR\n");
    printf("(%d, %d) = %d\n", 0, 0, OR(0, 0));
    printf("(%d, %d) = %d\n", 0, 1, OR(0, 1));
    printf("(%d, %d) = %d\n", 1, 0, OR(1, 0));
    printf("(%d, %d) = %d\n\n", 1, 1, OR(1, 1));

    printf("XOR\n");
    printf("(%d, %d) = %d\n", 0, 0, XOR(0, 0));
    printf("(%d, %d) = %d\n", 0, 1, XOR(0, 1));
    printf("(%d, %d) = %d\n", 1, 0, XOR(1, 0));
    printf("(%d, %d) = %d\n\n", 1, 1, XOR(1, 1));
}

void ex_activation_function() {
    double data[3] = {-1.0, 1.0, 2.0};
    NdShape *shape = NdShape_new(1,3);
    NdArray *array = NdArray_new(data, shape, DT_DOUBLE); 
    
    NdArray *array1 = NdArray_copy(array);
    step_function(array1);
    NdArray *array2 = NdArray_copy(array);
    sigmoid_function(array2);
    NdArray *array3 = NdArray_copy(array);
    relu_function(array3);

    NdArray_printArray(array);
    NdArray_printArray(array1);
    NdArray_printArray(array2);
    NdArray_printArray(array3);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&array1);
    NdArray_free(&array2);
    NdArray_free(&array3);
}

NdArray** init_network() {
    NdArray **network = (NdArray**)malloc(sizeof(NdArray*) * 6);

    double data_w1[2][3] = {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}};
    double data_b1[3] = {0.1, 0.2, 0.3};
    double data_w2[3][2] = {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}};
    double data_b2[2] = {0.1, 0.2};
    double data_w3[2][2] = {{0.1, 0.3}, {0.2, 0.4}};
    double data_b3[2] = {0.1, 0.2,};

    NdShape *shape_w1 = NdShape_new(2, 2, 3);
    NdShape *shape_b1 = NdShape_new(1, 3);
    NdShape *shape_w2 = NdShape_new(2, 3, 2);
    NdShape *shape_b2 = NdShape_new(1, 2);
    NdShape *shape_w3 = NdShape_new(2, 2, 2);
    NdShape *shape_b3 = NdShape_new(1, 2);

    network[0] = NdArray_new(data_w1, shape_w1, DT_DOUBLE);
    network[1] = NdArray_new(data_b1, shape_b1, DT_DOUBLE);
    network[2] = NdArray_new(data_w2, shape_w2, DT_DOUBLE);
    network[3] = NdArray_new(data_b2, shape_b2, DT_DOUBLE);
    network[4] = NdArray_new(data_w3, shape_w3, DT_DOUBLE);
    network[5] = NdArray_new(data_b3, shape_b3, DT_DOUBLE);

    NdShape_free(&shape_w1);
    NdShape_free(&shape_b1);
    NdShape_free(&shape_w2);
    NdShape_free(&shape_b2);
    NdShape_free(&shape_w3);
    NdShape_free(&shape_b3);

    return network;
}

NdArray* forward(NdArray *x) {
    NdArray **network = init_network();
    NdArray *w1, *w2, *w3;
    NdArray *b1, *b2, *b3;
    NdArray *a1, *a2, *a3;
    NdArray *y;

    w1 = network[0];
    b1 = network[1];
    w2 = network[2];
    b2 = network[3];
    w3 = network[4];
    b3 = network[5];

    a1 = NdArray_dot(x, w1);
    NdArray_add(a1, b1);
    sigmoid_function(a1);

    a2 = NdArray_dot(a1, w2);
    NdArray_add(a2, b2);
    sigmoid_function(a2);

    a3 = NdArray_dot(a2, w3);
    NdArray_add(a3, b3);
    identity_function(a3);
    
    y = NdArray_copy(a3);

    NdArray_free(&w1);
    NdArray_free(&w2);
    NdArray_free(&w3);
    NdArray_free(&b1);
    NdArray_free(&b2);
    NdArray_free(&b3);
    NdArray_free(&a1);
    NdArray_free(&a2);
    NdArray_free(&a3);

    return y;
}

void ex_neural_network() {
    double data_x[2] = {1.0, 0.5};
    NdShape *shape_x = NdShape_new(1, 2);
    NdArray *x = NdArray_new(data_x, shape_x, DT_DOUBLE);
    NdArray *y = forward(x);
    NdArray_printArray(y);

    NdShape_free(&shape_x);
    NdArray_free(&x);
    NdArray_free(&y);
}

void ex_softmax_function() {
    double data[3] = {0.3, 2.9, 4.0};
    NdShape *shape = NdShape_new(1,3);
    NdArray *array = NdArray_new(data, shape, DT_DOUBLE);
    NdArray *y = softmax(array);
    NdArray_printArray(array);
    NdArray_printArray(y);
    double sum_y = NdArray_sum_double(y);
    printf("%f\n", sum_y);

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&y);
}


void ex_mean_squared_error() {
    double t[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    double y1[10] = {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
    double y2[10] = {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};

    NdShape *shape = NdShape_new(1, 10);
    NdArray *array_t = NdArray_new(t, shape, DT_DOUBLE);
    NdArray *array_y1 = NdArray_new(y1, shape, DT_DOUBLE);
    NdArray *array_y2 = NdArray_new(y2, shape, DT_DOUBLE);

    double m1 = mean_squared_error(array_y1, array_t);
    double m2 = mean_squared_error(array_y2, array_t);

    printf("%f\n", m1);
    printf("%f\n", m2);

    NdShape_free(&shape);
    NdArray_free(&array_t);
    NdArray_free(&array_y1);
    NdArray_free(&array_y2);
}

void ex_cross_entropy_error() {
    double t[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    NdShape *shape = NdShape_new(1, 10);
    NdArray *array_t = NdArray_new(t, shape, DT_DOUBLE);

    double y1[10] = {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
    double y2[10] = {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
    NdArray *array_y1 = NdArray_new(y1, shape, DT_DOUBLE);
    NdArray *array_y2 = NdArray_new(y2, shape, DT_DOUBLE);

    double m1 = cross_entropy_error(array_y1, array_t);
    double m2 = cross_entropy_error(array_y2, array_t);

    printf("%f\n", m1);
    printf("%f\n", m2);

    NdShape_free(&shape);
    NdArray_free(&array_t);
    NdArray_free(&array_y1);
    NdArray_free(&array_y2);
}

double function_2(NdArray *array) {
    double ret;
    NdArray *temp = NdArray_copy(array);
    NdArray_mul(temp, temp);
    ret = NdArray_sum_double(temp);
    NdArray_free(&temp);
    return ret;
}


void ex_numerical_diff() {
    double data0[2] = {3.0, 4.0};
    double data1[2] = {0.0, 2.0};
    double data2[2] = {3.0, 0.0};

    NdShape *shape = NdShape_new(1, 2);

    NdArray *array0 = NdArray_new(data0, shape, DT_DOUBLE);
    NdArray *array1 = NdArray_new(data1, shape, DT_DOUBLE);
    NdArray *array2 = NdArray_new(data2, shape, DT_DOUBLE);

    NdArray *grad0 = numerical_gradient(function_2, array0);
    NdArray *grad1 = numerical_gradient(function_2, array1);
    NdArray *grad2 = numerical_gradient(function_2, array2);

    NdArray_printArray(grad0);
    NdArray_printArray(grad1);
    NdArray_printArray(grad2);

    NdShape_free(&shape);
    NdArray_free(&array0);
    NdArray_free(&array1);
    NdArray_free(&array2);
    NdArray_free(&grad0);
    NdArray_free(&grad1);
    NdArray_free(&grad2);
}

void ex_gradient_descent() {
    double data[2] = {-3.0, 4.0};
    NdShape *shape = NdShape_new(1, 2);
    NdArray *array = NdArray_new(data, shape, DT_DOUBLE);

    double lr = 0.1;
    int step_num = 100;

    NdArray *grad = gradient_descent(function_2, array, lr, step_num);
    double *cur = grad->data;
    for(int i = 0; i < grad->shape->len; i++) {
        printf("%e\n", *((double*)grad->data + i));
    }

    NdShape_free(&shape);
    NdArray_free(&array);
    NdArray_free(&grad);
}

int main() {
    ex_perceptron();
    ex_activation_function();
    ex_neural_network();
    ex_softmax_function();
    ex_mean_squared_error();
    ex_cross_entropy_error();
    ex_numerical_diff();
    ex_gradient_descent();
    return 0;
}
