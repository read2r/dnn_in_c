#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"
#include "ndshape.h"
#include "ndarray.h"

FILE *fileopen(const char *pathname, const char *mode) {
    FILE *file = fopen(pathname, mode);
    if(file == NULL) {
        perror(pathname);
        abort();
    }
    return file;
}

// big to little, little to big;
void convert_endian(void *addr, unsigned long size, unsigned int len) {
    unsigned char *cur = addr;
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < size/2; j++) {
            unsigned char temp = *(cur + j);
            *(cur + j) = *(cur + size - j - 1);
            *(cur + size - j - 1) = temp;
        }
        cur += size;
    }
}

struct parser* parser_new(FILE *file) {
    struct parser *p = (struct parser*)malloc(sizeof(struct parser));
    p->file = file;

    // read header
    fread(p->header, sizeof(char), MNIST_MSB_LEN, p->file);

    // extrat msb from header
    int msb = *(int*)p->header;
    // file is aligned by big endian.
    // but in c, it is aligned by little endian.
    // it is necessray to fit to little endian // so that it can be used in C properly.
    convert_endian(&msb, sizeof(int), 1);

    if(msb == MNIST_MSB_IMAGES) {
        fread(p->header + MNIST_MSB_LEN, sizeof(char), MNIST_IMAGE_HEADER_LEN - MNIST_MSB_LEN, file);
    } else if(msb == MNIST_MSB_LABELS) {
        fread(p->header + MNIST_MSB_LEN, sizeof(char), MNIST_LABEL_HEADER_LEN - MNIST_MSB_LEN, file);
    }
    convert_endian(p->header, sizeof(int), 4);

    int num, rows, cols;
    if(msb == MNIST_MSB_IMAGES) {
        num = *(int*)(p->header + sizeof(int));
        rows = *(int*)(p->header + sizeof(int) * 2);
        cols = *(int*)(p->header + sizeof(int) * 3);

        p->num_of_items = num;
        p->unit_length = rows * cols;
        p->unit_size = sizeof(unsigned char);
    } else if(msb == MNIST_MSB_LABELS) {
        num = *(int*)(p->header + 4);

        p->num_of_items = num;
        p->unit_length = 1;
        p->unit_size = sizeof(unsigned char);
    }

    return p;
}

void parser_free(struct parser **ptr_parser) {
    fclose((*ptr_parser)->file);
    free(*ptr_parser);
    *ptr_parser = NULL;
}

int parser_next(struct parser* p) {
    return fread(p->buffer, p->unit_size, p->unit_length, p->file);
}

void parser_show(struct parser* p) {
    printf("file : %p\n", p->file);
    printf("header : %p\n", p->header);
    printf("buffer : %p\n", p->buffer);
    printf("num of items : %d\n", (int)p->num_of_items);
    printf("size of item : %d\n", (int)p->unit_size); 
    printf("length of item  : %d\n", (int)p->unit_length); 
}

NdArray* load_images_set(const char* pathname) {
    FILE *file = fileopen(pathname, "rb");
    struct parser *p = parser_new(file);
    NdShape *shape_data_set = NdShape_new(2, p->num_of_items, p->unit_length);
    NdArray *data_set = NdArray_zeros(p->num_of_items * p->unit_length, DT_DOUBLE);

    int len = 0;
    void *cur = data_set->data;

    while((len = parser_next(p)) != 0) {
        if(len != p->unit_length) {
            abort();
        }

        void* cur_buffer = p->buffer;
        for(int i = 0; i < p->unit_length; i++) {
            *(double*)(cur) = (double)(*(unsigned char*)cur_buffer);
            cur += data_set->item_size;
            cur_buffer += p->unit_size;
        }
    }

    parser_free(&p);

    NdArray_reshape(data_set, shape_data_set);
    return data_set;
}

NdArray* load_labels_set(const char* pathname) {
    FILE *file = fileopen(pathname, "rb");
    struct parser *p = parser_new(file);
    NdShape *shape_data_set = NdShape_new(2, p->num_of_items, p->unit_length * 10);
    NdArray *data_set = NdArray_zeros(p->num_of_items * p->unit_length * 10, DT_DOUBLE);

    int len = 0;
    void *cur = data_set->data;

    while((len = parser_next(p)) != 0) {
        if(len != p->unit_length) {
            abort();
        }

        int offset;
        void* cur_buffer = p->buffer;
        for(int i = 0; i < p->unit_length; i++) {
            offset = (double)(*(unsigned char*)cur_buffer);
            *((double*)cur + offset) =  1;
            cur += data_set->item_size * 10;
            cur_buffer += p->unit_size * 10;
        }
    }
    
    parser_free(&p);

    NdArray_reshape(data_set, shape_data_set);
    return data_set;
}


NdArray* load_train_images_set() {
    NdArray *images_set = load_images_set("train-images.idx3-ubyte");
    return images_set;
}

NdArray* load_test_images_set() {
    NdArray *images_set = load_images_set("t10k-images.idx3-ubyte");
    return images_set;
}

NdArray* load_train_labels_set() {
    return load_labels_set("train-labels.idx1-ubyte");
}

NdArray* load_test_labels_set() {
    return load_labels_set("t10k-labels.idx1-ubyte");
}
