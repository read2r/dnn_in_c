#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "layer.h"
#include "ndshape.h"
#include "ndarray.h"
#include "loss.h"
#include "activation.h"

LayerType get_layertype(layer *l) {
    return l->layertype;
}

NdArray* layer_forward(layer *self, NdArray *x, NdArray *t) {
    NdArray* result = NULL;
    LayerType lt = get_layertype(self);
    switch(lt) {
        case LT_RELU:
            result = relu_layer_forward((relu_layer*)self, x, t);
            break;
        case LT_AFFINE:
            result = affine_layer_forward((affine_layer*)self, x, t);
            break;
        case LT_SIGMOID:
            result = sigmoid_layer_forward((sigmoid_layer*)self, x, t);
            break;
        case LT_SOFTMAX_WITH_LOSS:
            result = softmax_with_loss_layer_forward((softmax_with_loss_layer*)self, x, t);
            break;
        default:
            abort();
    }
    return result;
}

NdArray* layer_backward(layer *self, NdArray *dout) {
    NdArray* result = NULL;
    LayerType lt = get_layertype(self);
    switch(lt) {
        case LT_RELU:
            result = relu_layer_backward((relu_layer*)self, dout);
            break;
        case LT_AFFINE:
            result = affine_layer_backward((affine_layer*)self, dout);
            break;
        case LT_SIGMOID:
            result = sigmoid_layer_backward((sigmoid_layer*)self, dout);
            break;
        case LT_SOFTMAX_WITH_LOSS:
            result = softmax_with_loss_layer_backward((softmax_with_loss_layer*)self, dout);
            break;
        default:
            abort();
    }
    return result;
}

relu_layer* relu_layer_new() {
    relu_layer *rl = (relu_layer*)malloc(sizeof(relu_layer));
    rl->layertype = LT_RELU;
    rl->mask = NULL;
    return rl;
}

NdArray* relu_layer_forward(relu_layer *self, NdArray *x, NdArray *t) {
    if(self->mask != NULL) {
        NdArray_free(&(self->mask));
    }
    self->mask = NdArray_compare_scalar(x, 0, CT_GT);
    NdArray *out = NdArray_mask(x, self->mask);
    return out;
}

NdArray* relu_layer_backward(relu_layer *self, NdArray *dout) {
    NdArray *dx = NdArray_mask(dout, self->mask);
    return dx;
}

sigmoid_layer* sigmoid_layer_new() {
    sigmoid_layer *sl = (sigmoid_layer*)malloc(sizeof(sigmoid_layer));
    sl->layertype = LT_SIGMOID;
    sl->out = NULL;
    return sl;
}

NdArray* sigmoid_layer_forward(sigmoid_layer *self, NdArray *x, NdArray *t) {
    if(self->out != NULL) {
        NdArray_free(&(self->out));
    }
    NdArray* out = sigmoid_function(x);
    self->out = NdArray_copy(out);
    return out;
}

NdArray* sigmoid_layer_backward(sigmoid_layer *self, NdArray *dout) {
    // dx = dout
    NdArray *dx = NdArray_copy(dout);
    // temp = (1.0 - self.out)
    // temp = self.out
    NdArray *temp = NdArray_copy(self->out);
    // temp = -1 * temp
    // temp = -1 * self.out
    NdArray_mul_scalar(temp, -1);
    // temp = 1 + temp
    // temp = 1 + (-1 * self.out)
    // temp = 1 - self.out
    NdArray_add_scalar(temp, 1);
    // dx = dx * temp
    NdArray_mul(dx, temp);
    // dx = dx * temp * self.out
    // dx = dx * (1 - self.out) * self.out
    NdArray_mul(dx, self->out);
    NdArray_free(&temp);
    return dx;
}

affine_layer* affine_layer_new(NdArray *W, NdArray *b) {
    affine_layer *al = (affine_layer*)malloc(sizeof(affine_layer));
    al->layertype = LT_AFFINE;
    al->W = W;
    al->b = b;
    al->x = NULL;
    al->dW = NULL;
    al->db = NULL;
    return al;
}

NdArray* affine_layer_forward(affine_layer *self, NdArray *x, NdArray *t) {
    if(self->x != NULL) {
        NdArray_free(&(self->x));
    }
    self->x = NdArray_copy(x);
    NdArray *out = NdArray_dot(x, self->W);
    NdArray_add(out, self->b);
    return out;
}

NdArray* affine_layer_backward(affine_layer *self, NdArray *dout) {
    NdArray *W_T = NdArray_transpose(self->W);
    NdArray *x_T = NdArray_transpose(self->x);
    NdArray *dx = NdArray_dot(dout, W_T);
    self->dW = NdArray_dot(x_T, dout);
    self->db = NdArray_sum_axis(dout, 0);
    NdArray_free(&W_T);
    NdArray_free(&x_T);
    return dx;
}

softmax_with_loss_layer* softmax_with_loss_layer_new() {
    softmax_with_loss_layer *swll = (softmax_with_loss_layer*)malloc(sizeof(softmax_with_loss_layer));
    swll->layertype = LT_SOFTMAX_WITH_LOSS;
    swll->loss = NULL;
    swll->y = NULL;
    swll->t = NULL;
    return swll;
}

NdArray* softmax_with_loss_layer_forward(softmax_with_loss_layer *self, NdArray *x, NdArray *t) {
    self->t = t;
    self->y = softmax(x);
    self->loss = NdArray_zeros(1, DT_DOUBLE);
    double temp = cross_entropy_error(self->y, self->t);
    *((double*)self->loss->data) = temp;
    return self->loss;
}

NdArray* softmax_with_loss_layer_backward(softmax_with_loss_layer *self, NdArray *dout) {
    double batch_size = (double)self->t->shape->arr[0];
    NdArray *dx = NdArray_copy(self->y);
    NdArray_sub(dx, self->t);
    NdArray_div_scalar(dx, batch_size);
    return dx;
}
