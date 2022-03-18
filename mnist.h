#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ndshape.h"
#include "ndarray.h"

#define MNIST_MSB_IMAGES 2051
#define MNIST_MSB_LABELS 2049

#define MNIST_DEFAULT_HEADER_LEN 16
#define MNIST_DEFAULT_BUFFER_LEN 1024

#define MNIST_MSB_LEN 4
#define MNIST_LABEL_HEADER_LEN 8
#define MNIST_IMAGE_HEADER_LEN 16

struct parser {
    FILE *file;
    unsigned char header[MNIST_DEFAULT_HEADER_LEN];
    unsigned char buffer[MNIST_DEFAULT_BUFFER_LEN];
    unsigned int num_of_items;
    unsigned long unit_size;
    unsigned int unit_length;
};

FILE *fileopen(const char *pathname, const char *mode);

struct parser* parser_new(FILE *file);
void parser_free(struct parser **ptr_parser);
int parser_next(struct parser* p);
void parser_show(struct parser* p);

NdArray* load_images_set(const char* pathname);
NdArray* load_labels_set(const char* pathname);

NdArray* load_train_images_set();
NdArray* load_test_images_set();
NdArray* load_train_labels_set();
NdArray* load_test_labels_set();
