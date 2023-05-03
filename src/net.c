#include <stdlib.h>
#include <stdio.h>
#include "uwnet.h"
#include "math.h"


/*
https://deepnotes.io/softmax-crossentropy
https://www.deeplearningbook.com.br/cross-entropy-cost-function/

*/
float cross_entropy_loss(matrix y, layer l)
{
    matrix preds = l.out[0];
    matrix delta = l.delta[0];
    // assert(y.rows == preds.rows);
    // assert(y.cols == preds.cols);

    check_nan_matrix(preds);
    check_nan_matrix(delta);


    int i;
    float sum = 0;
    float pred_log;
    for (i = 0; i < y.cols * y.rows; ++i) {
        // if (isnan(preds.data[i])) {
        //    printf("\n\n preds.data[i]  is nan \n\n");
        // }

        pred_log = logf(preds.data[i] + 1e-7);

        if ((preds.data[i] + 1e-7) <= 0.0) {
            printf("\n\n preds.data[i]  <= zero %f ... log is %f \n\n", preds.data[i], pred_log);
        }


        // if (isnan(pred_log)) {
        //    printf("pred log is nan!\n");
        // }

        sum += -y.data[i] * pred_log;
        delta.data[i] += preds.data[i] - y.data[i];
    }

    check_nan_matrix(preds);
    check_nan_matrix(delta);

    return sum / y.rows;
}


matrix forward_net(net m, matrix X)
{
    int i;
    for (i = 0; i < m.n; ++i) {
        layer l = m.layers[i];
        X = l.forward(l, X);
    }
    return X;
}

void backward_net(net m)
{
    int i;
    for (i = m.n-1; i >= 0; --i) {
        layer l = m.layers[i];
        matrix prev_delta = {0};

        if (i > 0) {
            // we get the previous layer to multiply it by the current layer weights
            prev_delta = m.layers[i - 1].delta[0];
        }
        
        l.backward(l, prev_delta);
    }
}

void update_net(net m, float rate, float momentum, float decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        l.update(l, rate, momentum, decay);
    }
}

void free_layer(layer l)
{
    free_matrix(l.w);
    free_matrix(l.dw);
    free_matrix(l.b);
    free_matrix(l.db);
    if(l.in) free(l.in);
    if(l.out) free(l.out);
    if(l.delta) {
        free_matrix(*l.delta);
        free(l.delta);
    }
}

void free_net(net n)
{
    int i;
    for(i = 0; i < n.n; ++i){
        free_layer(n.layers[i]);
    }
    free(n.layers);
}

void file_error(char *filename)
{
    fprintf(stderr, "Couldn't open file %s\n", filename);
    exit(-1);
}

void save_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) write_matrix(l.b, fp);
        if(l.w.data) write_matrix(l.w, fp);
    }
    fclose(fp);
}

void load_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) read_matrix(l.b, fp);
        if(l.w.data) read_matrix(l.w, fp);
    }
    fclose(fp);
}
