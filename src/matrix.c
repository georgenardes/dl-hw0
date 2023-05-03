#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>


// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.shallow = 0;
    m.data = calloc(m.rows*m.cols, sizeof(float));
    return m;
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix random_matrix(int rows, int cols, float s)
{
    matrix m = make_matrix(rows, cols);
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            m.data[i*cols + j] = 2*s*(rand()%1000/1000.0) - s;    
        }
    }
    return m;
}

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix m)
{
    if (!m.shallow && m.data) {
        free(m.data);
    }
}

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix copy_matrix(matrix m)
{
    matrix c = make_matrix(m.rows, m.cols);
    // TODO: 1.1 - Fill in the new matrix
    int i, j;
    for (i = 0; i < m.rows; ++i) {
        for (j = 0; j < m.cols; ++j) {
            c.data[i * m.cols + j] = m.data[i * m.cols + j];
        }
    }
    return c;
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
matrix transpose_matrix(matrix m)
{
    // TODO: 1.2 - Make a matrix the correct size, fill it in
    matrix t = make_matrix(m.cols, m.rows);
    int i, j;
    for (i = 0; i < m.rows; ++i) {
        for (j = 0; j < m.cols; ++j) {
            t.data[j * m.rows + i] = m.data[i * m.cols + j];
        }
    }
    return t;
}

// Perform y = ax + y
// float a: scalar for matrix x
// matrix x: left operand to the scaled addition
// matrix y: unscaled right operand, also stores result
void axpy_matrix(float a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    // TODO: 1.3 - Perform the weighted sum, store result back in y
    matrix copy = copy_matrix(x);
    scal_matrix(a, copy);
    int i, j;
    for (i = 0; i < y.rows; ++i) {
        for (j = 0; j < y.cols; ++j) {
            y.data[i * y.cols + j] += copy.data[i * y.cols + j];
        }
    }
    free_matrix(copy);
}

// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix matmul(matrix a, matrix b)
{
    matrix c = make_matrix(a.rows, b.cols);    
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < b.rows; ++k) {
            for (int j = 0; j < b.cols; ++j) {
                float product = a.data[i * a.cols + k] * b.data[k * b.cols + j];
                c.data[i * b.cols + j] += product;
            }
        }
    }
    return c;
}


// Perform element-wise matrix division a/b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix matdiv(matrix a, matrix b) 
{
    if (a.cols != b.cols || a.rows != b.rows)
        printf("ERRO on element-wise matrix mult!! \n");

    matrix c = make_matrix(a.rows, a.cols);    
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {                        
            c.data[i * a.cols + k] = a.data[i * a.cols + k] / b.data[i * a.cols + k];
        }
    }
    return c;
}


// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix mathamm(matrix a, matrix b)
{
    if (a.cols != b.cols || a.rows != b.rows)
        printf("ERRO on element-wise matrix mult!! \n");

    matrix c = make_matrix(a.rows, a.cols);    
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {
            c.data[i * a.cols + k] = a.data[i * a.cols + k] * b.data[i * a.cols + k];
        }
    }
    return c;
}


// In-place, element-wise scaling of matrix
// float s: scaling factor
// matrix m: matrix to be scaled
void scal_matrix(float s, matrix m)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j =0 ; j < m.cols; ++j){
            m.data[i*m.cols + j] *= s;
        }
    }
}


float reduced_max(matrix m) {
    float max = -1e+30;
    for (int i = 0; i < m.rows * m.cols; i++) {
        if (m.data[i] > max)
            max = m.data[i];
    }
    return max;
}


// In-place square root of each matrix element with epsilon
void sqrt_matrix(matrix m) 
{
    int i, j;
    for (i = 0; i < m.rows; ++i) {
        for (j = 0; j < m.cols; ++j) {
            m.data[i * m.cols + j] = sqrtf(m.data[i * m.cols + j] + 1e-8);
        }
    }
}

void check_nan_matrix(matrix m) {
    int q = 0;
    for (int i = 0; i < m.rows * m.cols; ++i) {
        if (((int)(m.data[i])) == (-2147483648)) {
            printf("\nidentificado matrix com nan!\n");
            q = 1;
        }        
    }

    if (q == 1) { 
        print_matrix(m); 
    }
}


// Print a matrix
void print_matrix(matrix m)
{
    int i, j;
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}

// Used for matrix inversion
matrix augment_matrix(matrix m)
{
    int i,j;
    matrix c = make_matrix(m.rows, m.cols*2);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            c.data[i*c.cols + j] = m.data[i*m.cols + j];
        }
    }
    for(j = 0; j < m.rows; ++j){
        c.data[j*c.cols + j+m.cols] = 1;
    }
    return c;
}

// Invert matrix m
matrix matrix_invert(matrix m)
{
    int i, j, k;
    //print_matrix(m);
    matrix none = {0};
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return none;
    }
    matrix c = augment_matrix(m);
    //print_matrix(c);
    float **cdata = calloc(c.rows, sizeof(float *));
    for(i = 0; i < c.rows; ++i){
        cdata[i] = c.data + i*c.cols;
    }


    for(k = 0; k < c.rows; ++k){
        float p = 0.;
        int index = -1;
        for(i = k; i < c.rows; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            free_matrix(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.cols; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.rows; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.rows-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    //print_matrix(c);
    matrix inv = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            inv.data[i*m.cols + j] = cdata[i][j+m.cols];
        }
    }
    free_matrix(c);
    free(cdata);
    //print_matrix(inv);
    return inv;
}

matrix solve_system(matrix M, matrix b)
{
    matrix none = {0};
    matrix Mt = transpose_matrix(M);
    matrix MtM = matmul(Mt, M);
    matrix MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    matrix Mdag = matmul(MtMinv, Mt);
    matrix a = matmul(Mdag, b);
    free_matrix(Mt); free_matrix(MtM); free_matrix(MtMinv); free_matrix(Mdag);
    return a;
}

void write_matrix(matrix m, FILE *fp)
{
    fwrite(m.data, sizeof(float), m.rows*m.cols, fp);
}

void read_matrix(matrix m, FILE *fp)
{
     fread(m.data, sizeof(float), m.rows*m.cols, fp);
}

void save_matrix(matrix m, char *fname)
{
    FILE *fp = fopen(fname, "wb");
    fwrite(&m.rows, sizeof(int), 1, fp);
    fwrite(&m.cols, sizeof(int), 1, fp);
    write_matrix(m, fp);
    fclose(fp);
}

matrix load_matrix(char *fname)
{
    int rows = 0;
    int cols = 0;
    FILE *fp = fopen(fname, "rb");
    fread(&rows, sizeof(int), 1, fp);
    fread(&cols, sizeof(int), 1, fp);
    matrix m = make_matrix(rows, cols);
    read_matrix(m, fp);
    return m;
}

void test_matrix()
{
    int i;
    for(i = 0; i < 100; ++i){
        int s = rand()%4 + 3;
        matrix m = random_matrix(s, s, 10);
        matrix inv = matrix_invert(m);
        if(inv.data){
            matrix res = matmul(m, inv);
            print_matrix(res);
        }
    }
}
