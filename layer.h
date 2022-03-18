#ifndef __LAYER_H__
#define __LAYER_H__
#include "ndarray.h"

typedef struct _layer layer;
typedef NdArray* (*fptr_forward)(layer *self, NdArray *x, NdArray *t);
typedef NdArray* (*fptr_backward)(layer *self, NdArray *dout);

typedef enum _LayerType {
    LT_RELU,
    LT_SIGMOID,
    LT_AFFINE,
    LT_SOFTMAX_WITH_LOSS,
} LayerType;

typedef struct _layer_interface {
    fptr_forward forward;
    fptr_backward backward;
} layer_interface;

typedef struct _layer {
    LayerType layertype;
} layer;

typedef struct _relu_layer {
    LayerType layertype;
    NdArray *mask;
} relu_layer;

typedef struct _sigmoid_layer {
    LayerType layertype;
    NdArray *out;
} sigmoid_layer;

typedef struct _affine_layer {
    LayerType layertype;
    NdArray *W;
    NdArray *b;
    NdArray *x;
    NdArray *dW;
    NdArray *db;
} affine_layer;

typedef struct _softmax_with_loss_layer {
    LayerType layertype;
    NdArray *loss;
    NdArray *y;
    NdArray *t;
} softmax_with_loss_layer;

NdArray* layer_forward(layer *self, NdArray *x, NdArray *t);
NdArray* layer_backward(layer *self, NdArray *dout);

relu_layer* relu_layer_new();
NdArray* relu_layer_forward(relu_layer *self, NdArray *x, NdArray *t);
NdArray* relu_layer_backward(relu_layer *self, NdArray *dout);

sigmoid_layer* sigmoid_layer_new();
NdArray* sigmoid_layer_forward(sigmoid_layer *self, NdArray *x, NdArray *t);
NdArray* sigmoid_layer_backward(sigmoid_layer *self, NdArray *dout);

affine_layer* affine_layer_new(NdArray *W, NdArray *b);
NdArray* affine_layer_forward(affine_layer *self, NdArray *x, NdArray *t);
NdArray* affine_layer_backward(affine_layer *self, NdArray *dout);

softmax_with_loss_layer* softmax_with_loss_layer_new();
NdArray* softmax_with_loss_layer_forward(softmax_with_loss_layer *self, NdArray *x, NdArray *t);
NdArray* softmax_with_loss_layer_backward(softmax_with_loss_layer *self, NdArray *dout);

#endif
