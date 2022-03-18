#ifndef __NN_H__
#define __NN_H__
#include "ndarray.h"
#include "layer.h"

typedef struct _two_layer_net {
    NdArray **params;
    layer **layers;
    layer *last_layer;
} two_layer_net;

two_layer_net* two_layer_net_new(int input_size, int hidden_size, int output_size, double weight_init_std);
NdArray* two_layer_net_predict(two_layer_net *self, NdArray *x, NdArray *t);
NdArray* two_layer_net_loss(two_layer_net *self, NdArray *x, NdArray *t);
double two_layer_net_accuracy(two_layer_net *self, NdArray *x, NdArray *t);
NdArray** two_layer_net_gradient(two_layer_net *self, NdArray *x, NdArray *t);

NdArray* get_batch_array(NdArray* self, NdArray* indices);

#endif
