#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
void forward_bias(matrix m, matrix b)
{
    assert(b.rows == 1);
    assert(m.cols == b.cols);
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.data[i*m.cols + j] += b.data[j];
        }
    }
}

// Calculate bias updates from a delta matrix (like a reduced_sum)
// matrix delta: error made by the layer
// matrix db: delta for the biases
void backward_bias(matrix delta, matrix db)
{
    int i, j;
    for(i = 0; i < delta.rows; ++i){
        for(j = 0; j < delta.cols; ++j){            
            db.data[j] += -delta.data[i*delta.cols + j];
        }
    }
}





// Run a connected layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer: f(wx + b)
matrix forward_connected_layer(layer l, matrix in_original)
{
    matrix w = l.w; // !! check if it is passing the correct value !! \\ // 
    matrix b = l.b; // !! check if it is passing the correct value !! \\ // 
    matrix in = in_original;

    if (QUANTIZE) {
        w = quantize_matrix_to_sx4(w, 4, 0, 7);
        b = quantize_matrix_to_sx4(b, 4, 0, 7);        
        //in = quantize_matrix_to_sx4(in, 4, 0, 7);
    }
    
    // sanity check
    check_nan_matrix(in);
    check_nan_matrix(w);
    check_nan_matrix(b);

    // run the network forward
    matrix out = matmul(in, w);  // xw
    check_nan_matrix(out);
    forward_bias(out, b);  // + b
    check_nan_matrix(out);


    // apply activation
    activate_matrix(out, l.activation);
    check_nan_matrix(out);

    // Saving our input and output and making a new delta matrix to hold errors
    // Probably don't change this
    l.in[0] = in_original;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);    


    if (QUANTIZE) { // if quantization was enabled, we must free memory of allocated matrixes
        free_matrix(w);
        free_matrix(b); 
        //free_matrix(in);
    }

    

    return out;
}

// Run a connected layer backward
// layer l: layer to run
// matrix prev_delta: privious layer delta  (TRANSFERIR TODA PARTE DE ADAM PARA A FUNÇÃO DE PARAMETER UPDATE)
void backward_connected_layer(layer l, matrix prev_delta)
{    
    matrix in    = l.in[0];
    matrix out   = l.out[0]; // output is = activation(Z)
    matrix delta = l.delta[0];    

    //check_nan_matrix(in);
    //check_nan_matrix(out);
    //check_nan_matrix(delta);


    // delta is the error made by this layer, dL/dout
    // First modify in place to be dL/d(in*w+b) using the gradient of activation
    gradient_matrix(out, l.activation, delta);
    
    // Calculate the updates for the bias terms using backward_bias
    // The current bias deltas are stored in l.db
    backward_bias(delta, l.db); // dldb
     
    // Then calculate dL/dw. Use axpy to subtract this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw    
    matrix intransp = transpose_matrix(in);
    matrix dldw = matmul(intransp, delta);        
    
    // sets dw  with dw += (1.0-beta1) * dldw 
    // however, some implementations ommit the subtraction by the beta1
    // the -1.0 is used to flip de dw in order to add instead of subtrac
    axpy_matrix(-1.0, dldw, l.dw);     
        

    if (prev_delta.data) {
        // Finally, if there is a previous layer to calculate for,
        // calculate dL/d(in). Again, using axpy, add this into the current
        // value we have for the previous layers delta, prev_delta.
        matrix wtransp = transpose_matrix(l.w);
        matrix w = matmul(delta, wtransp);
        axpy_matrix(1.0, w, prev_delta);

        free_matrix(wtransp);
        free_matrix(w);
    }
    free_matrix(intransp);
    free_matrix(dldw);    

}

// Adam Optimizer
// there is a bug because of sqrt of negative values
void update_connected_layer_with_adam(layer l, float rate, float decay, float iteration, float b1, float b2)
{
    
    matrix vdw_correct = copy_matrix(l.vdw);
    matrix sdw_correct = copy_matrix(l.sdw);
    scal_matrix((1.0 / (1.0 - pow(b1, iteration+1))), vdw_correct);
    scal_matrix((1.0 / (1.0 - pow(b2, iteration+1))), sdw_correct);
    sqrt_matrix(sdw_correct); //must be non negative because of sqrt


    matrix vdb_correct = copy_matrix(l.vdb);
    matrix sdb_correct = copy_matrix(l.sdb);
    scal_matrix((1.0 / (1.0 - pow(b1, iteration+1))), vdb_correct);
    scal_matrix((1.0 / (1.0 - pow(b2, iteration+1))), sdb_correct);
    sqrt_matrix(sdb_correct); // must be non negative because of sqrt

    matrix dw = matdiv(vdw_correct, sdw_correct);
    matrix db = matdiv(vdb_correct, sdb_correct);

    // For our weights we want to include weight decay: axpy_matrix(-decay, l.w, l.dw);
  
    // Then for both weights and biases we want to apply the updates:
    axpy_matrix(rate, dw, l.w);
    axpy_matrix(rate, db, l.b);

    // Finally, we want to scale dw and db by our momentum to prepare them for the next round
    scal_matrix(b1, l.vdw);
    scal_matrix(b2, l.sdw);
    scal_matrix(b1, l.vdb);
    scal_matrix(b2, l.sdb);    

    // free memory
    free_matrix(vdw_correct);
    free_matrix(sdw_correct);
    free_matrix(dw);
    free_matrix(vdb_correct);
    free_matrix(sdb_correct);
    free_matrix(db);

}

// Adam Optimizer
// there is a bug because of sqrt of negative values
void update_connected_layer_with_momentum(layer l, float rate, float momentum, float decay)
{
    // For our weights we want to include weight decay:    
    axpy_matrix(-decay, l.w, l.dw);

    // Then for both weights and biases we want to apply the updates:
    axpy_matrix(rate, l.dw, l.w);
    axpy_matrix(rate, l.db, l.b);

    // Finally, we want to scale dw and db by our momentum to prepare them for the next round
    scal_matrix(momentum, l.dw);
    scal_matrix(momentum, l.db);

}


/* TODO: verificar se optimizer deve ficar em LAYER ou NET*/
layer make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.vdw = make_matrix(inputs, outputs);
    l.sdw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.vdb = make_matrix(1, outputs);
    l.sdb = make_matrix(1, outputs);
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.activation = activation;
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer_with_momentum;
    // l.beta1 = momentum;

    return l;
}




matrix quantize_matrix_to_sx4(matrix m, float num_bits, float min_exp, float max_exp) 
{        

    matrix qm = make_matrix(m.rows, m.cols);

    float org_x, qx;
    float min_linear_value = powf(2.0, -max_exp);
    
    for (int i = 0; i < m.rows * m.cols; i++) {
              
        org_x = qx = m.data[i];

        // clip original X between -1 e 1
        if (org_x > 1.0) {
            m.data[i] = 1.0;     
        }
        else if (org_x < -1.0) {
            m.data[i] = -1.0;
        }

        // start quantization

        float rng = (float)rand() / (float)RAND_MAX; // numero randomico entre 0 e 1.0
            
        qx = fabsf(qx);                 // |x|            
        if (qx < min_linear_value) {    // underflow                
            qx = max_exp;               // clip min (to avoid numerical problems)
            qx = powf(2.0, -qx);        // traz para representação float         
            qx = rng > 0.5 ? qx : -qx;  // coloca um sinal rand
        }
        else {
            qx = log2(qx);              // log2 
            qx = fabsf(qx);             // |qx|

            qx = (qx - (int)(qx)) > rng ? qx + 1 : qx;   // compara a mantissa para arredondar            
            qx = (int)qx;                                      // quantiza            
            qx = min(max_exp, max(min_exp, qx));               // clipa entre o min e max expoente                           
            qx = powf(2.0, -qx);                               // traz para representação float            
            qx = org_x < 0.0 ? -qx : qx;                       // coloca o sinal como estava 
        }

        // set the quantized value on the new matrix
        qm.data[i] = qx;
        
    }

    return qm;
}
