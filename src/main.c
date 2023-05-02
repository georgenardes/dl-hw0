#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"


void split_dataset(data train_all, data *train, data *val, int num_train_samples) {
    int num_val_samples = train_all.X.rows - num_train_samples;

    // split validation and train
    matrix train_x = make_matrix(num_train_samples, 784);
    matrix train_y = make_matrix(num_train_samples, 10);
    matrix val_x = make_matrix(num_val_samples, 784);
    matrix val_y = make_matrix(num_val_samples, 10);

    for (int r = 0; r < num_train_samples; r++) {
        for (int c = 0; c < 784; c++) {
            train_x.data[r * train_x.cols + c] = train_all.X.data[r * train_x.cols + c];
        }

        for (int c = 0; c < 10; c++) {
            train_y.data[r * train_y.cols + c] = train_all.y.data[r * train_y.cols + c];
        }
    }

    for (int r = 0; r < num_val_samples; r++) {
        for (int c = 0; c < 784; c++) {
            val_x.data[r * train_x.cols + c] = train_all.X.data[(r + num_train_samples) * train_x.cols + c];
        }

        for (int c = 0; c < 10; c++) {
            val_y.data[r * train_y.cols + c] = train_all.y.data[(r + num_train_samples) * train_y.cols + c];
        }
    }
    
    (*train).X = train_x;
    (*train).y = train_y;
    (*val).X = val_x;
    (*val).y = val_y;
}



void try_mnist()
{    
    data train_all = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");    
    data test = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");
    
    printf("the size of the full dataset is %d, %d \n", train_all.X.rows, train_all.X.cols);
    data train, val;
    split_dataset(train_all, &train, &val, (int)(train_all.X.rows*0.9));
    free(train_all.X.data);
    free(train_all.y.data);
    printf("used %d samples for training and %d samples for validation \n", train.X.rows, val.X.rows);


    net n = {0};
    n.n = 3; // number of layers
    n.layers = calloc(n.n, sizeof(layer));
    n.layers[0] = make_connected_layer(784, 256, RELU, SGDM); // first layer
    n.layers[1] = make_connected_layer(256, 256, RELU, SGDM); // second layer
    n.layers[2] = make_connected_layer(256, 10, SOFTMAX, SGDM); // third layer
    // n.layers[1] = make_connected_layer(32, 10, SOFTMAX); // third layer

    int batch = 128;
    int iters = 5000;
    int epochs = 15;
    float rate = 0.001;
    float momentum = 0.9;
    float decay = 0.0;

    train_val_image_classifier(n, train, val, batch, epochs, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Validating accuracy: %f\n", accuracy_net(n, val));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}

int main()
{
    //if(argc < 2){
    //    printf("usage: %s [test | trymnist]\n", argv[0]);  
    //} else if (0 == strcmp(argv[1], "trymnist")){
    try_mnist();
    //} else if (0 == strcmp(argv[1], "test")){
    //    run_tests();
    //}

    printf("finalizado o experimento ... \n");


}
