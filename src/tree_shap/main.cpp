#include <stdexcept>
#include "recursive_treeshap.hpp"
#include "stack_treeshap.hpp"



////// Wrapping the C++ functions with a C interface //////



extern "C"
int main_int_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                      int* I_map, double* threshold_, double* value_, int* feature_, 
                      int* left_child_, int* right_child_, double* result) {
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    // Precompute the SHAP weights
    int n_features = I_map[d-1] + 1;
    Matrix<double> W(n_features, vector<double> (n_features));
    compute_W(W);
    
    Matrix<double> phi = int_treeSHAP(X_f, X_b, I_map, 
                                     feature, left_child, right_child, 
                                     threshold, value, W);
    std::cout << std::endl;
    
    // Save the results
    for (int i(0); i < Nx; i++){
        for (int j(0); j < n_features; j++){
            result[i*n_features + j] = phi[i][j];
        }
    }
    return 0;
}




extern "C"
int main_taylor_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                         double* threshold_, double* value_, int* feature_, int* left_child_, int* right_child_,
                         double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    // Precompute the SHAP weights
    Matrix<double> W(d, vector<double> (d));
    compute_W(W);

    Tensor<double> phi = taylor_treeSHAP(X_f, X_b, feature, left_child, right_child, 
                                                 threshold, value, W);
    cout << endl;

    // Save the results
    for (unsigned int i(0); i < phi.size(); i++){
        for (int j(0); j < d; j++){
            for (int k(0); k < d; k++){
                result[i*d*d + j*d + k] = phi[i][j][k];
            }
        }
    }
    return 0;
}





extern "C"
int main_additive_treeshap(int N, int Nt, int d, int depth, double* X,
                         double* threshold_, double* value_, int* feature_, 
                         int* left_child_, int* right_child_,
                         double* result) {
    
    // Load data instances
    Matrix<double> X_ = createMatrix<double>(N, d, X);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    Tensor<double> A = additive_treeSHAP(X_, feature, left_child, right_child, threshold, value);
    cout << endl;

    // Save the results
    for (int i(0); i < N; i++){
        for (int j(0); j < N; j++){
            for (int k(0); k < d; k++){
                result[i*N*d + d*j + k] = A[i][j][k];
            }
        }
    }
    return 0;
}



extern "C"
int main_A_treeshap(int N, int Nt, int d, int depth, double* X,
                         double* threshold_, double* value_, int* feature_, 
                         int* left_child_, int* right_child_,
                         double* result, bool use_stack) {
    
    // Load data instances
    Matrix<double> X_ = createMatrix<double>(N, d, X);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    Matrix<double> A;
    if (use_stack) {
        cout << "Using Stack" << endl;
        A = A_treeSHAP_stack(X_, feature, left_child, right_child, threshold, value);
    }
    else {
        cout << "Using Recursion" << endl;
        A = A_treeSHAP_recurse(X_, feature, left_child, right_child, threshold, value);
    }
    cout << endl;

    // Save the results
    for (int i(0); i < N; i++){
        for (int j(0); j < N; j++){
            result[i*N + j] = A[i][j];
        }
    }
    return 0;
}