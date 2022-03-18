#include <stdio.h>
#include <stdlib.h>
#include "ndshape.h"
#include "ndarray.h"
#include "layer.h"
#include "nn.h"
#include "mnist.h"


int main() {
    NdArray *x_train = load_train_images_set();
    NdArray *t_train = load_train_labels_set();
    NdArray *x_test = load_test_images_set();
    NdArray *t_test = load_test_labels_set();

    // normalize
    NdArray_div_scalar(x_train, 255);
    NdArray_div_scalar(x_test, 255);

    two_layer_net *network = two_layer_net_new(784, 50, 10, 0.01);

    int iters_num = 10000;
    int train_size = x_train->shape->arr[0];
    int batch_size = 100;
    double learning_rate = 0.1;

    int iter_per_epoch = (train_size/batch_size > 1) ? train_size/batch_size : 1;

    for(int i = 0; i < iters_num; i++) {
        NdArray *batch_indices = NdArray_choice(batch_size, train_size, DT_INT);

        NdArray *x_batch = get_batch_array(x_train, batch_indices);
        NdArray *t_batch = get_batch_array(t_train, batch_indices);

        NdArray **grads = two_layer_net_gradient(network, x_batch, t_batch);

        for(int j = 0; j < 4; j++) {
            NdArray_mul_scalar(grads[j], learning_rate);
            NdArray_sub(network->params[j], grads[j]);
            NdArray_free(&grads[j]);
        }
        free(grads);

        NdArray *loss = two_layer_net_loss(network, x_batch, t_batch);
        printf("[%d] : ", i);
        NdArray_printArray(loss);

        if(i % iter_per_epoch == 0 || i == iters_num-1) {
            double train_acc = 0;
            double test_acc = 0;

            train_acc = two_layer_net_accuracy(network, x_train, t_train);
            test_acc = two_layer_net_accuracy(network, x_test, t_test);

            //printf("[%d] : ", i);
            //NdArray_printArray(loss);
            //printf("train_acc : %f\n", train_acc);
            printf("test_acc  : %f\n\n", test_acc);
        }

        NdArray_free(&batch_indices);
        NdArray_free(&x_batch);
        NdArray_free(&t_batch);
        NdArray_free(&loss);
    }

    free(network);
    NdArray_free(&x_train);
    NdArray_free(&t_train);
    NdArray_free(&x_test);
    NdArray_free(&t_test);

    return 0;
}
