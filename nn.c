#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ndarray.h"
#include "ndshape.h"
#include "layer.h"
#include "nn.h"

NdShape* get_batch_shape(NdArray *self, NdArray *indices) {
    NdShape *batch_shape = NdShape_empty(self->shape->dim);
    batch_shape->arr[0] = indices->shape->arr[0];
    batch_shape->len *= indices->shape->arr[0];
    for(int i = 1; i < batch_shape->dim; i++) {
        batch_shape->arr[i] = self->shape->arr[i];
        batch_shape->len *= self->shape->arr[i];
    }
    return batch_shape;
}

NdArray* get_batch_array(NdArray *self, NdArray *indices) {
    int batch_len = indices->shape->arr[0];
    int batch_step = self->shape->len / self->shape->arr[0];
    int batch_size = batch_step * self->item_size; 

    NdShape *batch_shape = get_batch_shape(self, indices);
    NdArray *batch_array = NdArray_new(NULL, batch_shape, self->datatype);

    void *cur_self = self->data;
    void *cur_batch = batch_array->data;
    unsigned int *cur_idx = indices->data;
    for(int i = 0; i < batch_len; i++) {
        unsigned int offset = cur_idx[i] * batch_size;
        memcpy(cur_batch, cur_self + offset, batch_size);
        cur_batch += batch_size;
    }

    NdShape_free(&batch_shape);
    
    return batch_array;
}

void two_layer_net_init(two_layer_net *self, int input_size, int hidden_size, int output_size, double weight_init_std) {
    self->params = (NdArray**)malloc(sizeof(NdArray*) * 4);
    self->layers = (layer**)malloc(sizeof(layer*) * 4);

    // w1
    self->params[0] = NdArray_random_gaussian(input_size * hidden_size);
    NdArray_reshape_variadic(self->params[0], 2, input_size, hidden_size);
    NdArray_mul_scalar(self->params[0], weight_init_std);
    // b2
    self->params[1] = NdArray_zeros(hidden_size, DT_DOUBLE);
    // w2
    self->params[2] = NdArray_random_gaussian(hidden_size * output_size);
    NdArray_reshape_variadic(self->params[2], 2, hidden_size, output_size);
    NdArray_mul_scalar(self->params[2], weight_init_std);
    // b2
    self->params[3] = NdArray_zeros(output_size, DT_DOUBLE);

    // Affine1
    self->layers[0] = (layer*)affine_layer_new(self->params[0], self->params[1]);
    // Relu1
    self->layers[1] = (layer*)relu_layer_new();
    // Affine2
    self->layers[2] = (layer*)affine_layer_new(self->params[2], self->params[3]);
    // Softmax with Loss
    self->last_layer = (layer*)softmax_with_loss_layer_new();
}

two_layer_net* two_layer_net_new(int input_size, int hidden_size, int output_size, double weight_init_std) {
    two_layer_net *net = (two_layer_net*)malloc(sizeof(two_layer_net));
    two_layer_net_init(net, input_size, hidden_size, output_size, weight_init_std);
    return net;
}

void two_layer_net_free(two_layer_net **self) {
}

NdArray* two_layer_net_predict(two_layer_net *self, NdArray *x, NdArray *t) {
    NdArray *ret = NdArray_copy(x);
    NdArray *temp;
    for(int i = 0; i < 3; i++) {
        temp = ret;
        ret = layer_forward(self->layers[i], temp, t);
        NdArray_free(&temp);
    }
    return ret;
}

NdArray* two_layer_net_loss(two_layer_net *self, NdArray *x, NdArray *t) {
    NdArray *y = two_layer_net_predict(self, x, t);
    NdArray *ret = layer_forward(self->last_layer, y, t);
    NdArray_free(&y);
    return ret;
}

double two_layer_net_accuracy(two_layer_net *self, NdArray *x, NdArray *t) {
    NdArray *y = two_layer_net_predict(self, x, t);
    NdArray *temp = y;
    y = NdArray_argmax_axis(temp, 1);
    NdArray_free(&temp);

    if(t->shape->dim != 1) {
        t = NdArray_argmax_axis(t, 1);
    }

    NdArray *comp = NdArray_compare(y, t, CT_EQ);
    int sum_comp = NdArray_sum_char(comp);
    double accuracy = (double)sum_comp / (double)x->shape->arr[0];

    NdArray_free(&comp);
    NdArray_free(&y);

    return accuracy;
}

NdArray** two_layer_net_gradient(two_layer_net *self, NdArray *x, NdArray *t) {
    NdArray *dummy = two_layer_net_loss(self, x, t);
    NdArray_free(&dummy);

    NdArray *dout = layer_backward(self->last_layer, NULL);
    NdArray *temp;
    for(int i = 2; i >= 0; i--) {
        temp = dout;
        dout = layer_backward(self->layers[i], temp);
        NdArray_free(&temp);
    }
    NdArray_free(&dout);

    NdArray **grads = (NdArray**)malloc(sizeof(NdArray*) * 4);
    grads[0] = ((affine_layer*)self->layers[0])->dW;
    grads[1] = ((affine_layer*)self->layers[0])->db;
    grads[2] = ((affine_layer*)self->layers[2])->dW;
    grads[3] = ((affine_layer*)self->layers[2])->db;

    return grads;
}
