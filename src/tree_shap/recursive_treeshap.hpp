#ifndef __RECURS
#define __RECURS

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"

// Recursion function for treeSHAP
pair<double, double> recurse(int n,
                            vector<double> &x, vector<double> &z, 
                            int* I_map,
                            vector<int> &feature,
                            vector<int> &child_left,
                            vector<int> &child_right,
                            vector<double> &threshold,
                            vector<double> &value,
                            vector<vector<double>> &W,
                            int n_features,
                            vector<double> &phi,
                            vector<int> &in_SX,
                            vector<int> &in_SZ)
{
    int current_feature = feature[n];
    int x_child(0), z_child(0);
    // num_players := |S_{XZ}|
    int num_players = 0;

    // Arriving at a Leaf
    if (child_left[n] < 0)
    {
        double pos(0.0), neg(0.0);
        num_players = in_SX[n_features] + in_SZ[n_features];
        if (in_SX[n_features] > 0)
        {
            pos = W[in_SX[n_features]-1][num_players-1] * value[n];
        }
        if (in_SZ[n_features] > 0)
        {
            neg = W[in_SX[n_features]][num_players-1] * value[n];
        }
        return make_pair(pos, neg);
    }
    
    // Find children of x and z
    if (x[current_feature] <= threshold[n]){
        x_child = child_left[n];
    } else {x_child = child_right[n];}
    if (z[current_feature] <= threshold[n]){
        z_child = child_left[n];
    } else {z_child = child_right[n];}

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse(x_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in I(S_X) U I(S_Z).
    // Hence we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (in_SX[ I_map[current_feature] ] || in_SZ[ I_map[current_feature] ]){
        if (in_SX[ I_map[current_feature] ]){
            return recurse(x_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
        else{
            return recurse(z_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child
        in_SX[ I_map[current_feature] ]++;
        in_SX[n_features]++;
        pair<double, double> pairf = recurse(x_child, x, z, I_map, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SX[ I_map[current_feature] ]--;
        in_SX[n_features]--;

        // Go to z's child
        in_SZ[ I_map[current_feature] ]++;
        in_SZ[n_features]++;
        pair<double, double> pairb = recurse(z_child, x, z, I_map, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SZ[ I_map[current_feature] ]--;
        in_SZ[n_features]--;

        // Add contribution to the feature
        phi[ I_map[current_feature] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}



// Recursion function for Taylor-TreeSHAP
int recurse_2(int n,
            vector<double> &x, vector<double> &z, 
            vector<int> &feature,
            vector<int> &child_left,
            vector<int> &child_right,
            vector<double> &threshold,
            vector<double> &value,
            vector<vector<double>> &W,
            int n_features,
            vector<double> &phi,
            vector<int> &in_SX,
            vector<int> &in_SZ)
{
    int current_feature = feature[n];
    int x_child(0), z_child(0);
    // num_players := |S_{AB}|
    int num_players = 0;

    // Arriving at a Leaf
    if (child_left[n] < 0)
    {
        num_players = in_SX[n_features] + in_SZ[n_features];
        if (num_players == 0){
            return 0;
        }
        for (int i(0); i < n_features; i++){
            for (int j(0); j < n_features; j++){
                // Diagonal element
                if (i == j) {
                    // i in S_Z and S_X is empty
                    if (in_SZ[i] && (in_SX[n_features] == 0) ){
                        phi[i * n_features + i] -= value[n];
                    }
                    // S_X = {i}
                    if (in_SX[i] && (in_SX[n_features] == 1) ){
                        phi[i * n_features + i] += value[n];
                    }
                }
                // Non-diagonal element
                else {
                    // i,j in S_X
                    if (in_SX[i] && in_SX[j]){
                        phi[i * n_features + j] += W[in_SX[n_features]-2][num_players-1] * value[n];
                    }
                    // i,j in S_Z
                    else if (in_SZ[i] && in_SZ[j]){
                        phi[i * n_features + j] += W[in_SX[n_features]][num_players-1] * value[n];
                    }
                    // i in S_X  and  j in S_Z   OR
                    // j in S_X  and  i in S_Z
                    else if ((in_SX[i] + in_SZ[j] + in_SX[j] + in_SZ[i]) == 2){
                        phi[i * n_features + j]-= W[in_SX[n_features]-1][num_players-1] * value[n];
                    }
                }
            }
        }
        return 0;
    }

    // Find children of x and z
    if (x[current_feature] <= threshold[n]){
        x_child = child_left[n];
    } else {x_child = child_right[n];}
    if (z[current_feature] <= threshold[n]){
        z_child = child_left[n];
    } else {z_child = child_right[n];}

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_2(x_child, x, z, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in S_X U S_Z.
    // Hence we go down the correct edge to ensure that S_X and S_Z are kept disjoint
    if (in_SX[current_feature] || in_SZ[current_feature]){
        if (in_SX[current_feature]){
            return recurse_2(x_child, x, z, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
        else{
            return recurse_2(z_child, x, z, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child
        in_SX[current_feature]++; in_SX[n_features]++;
        recurse_2(x_child, x, z, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SX[current_feature]--; in_SX[n_features]--;

        // Go to z's child
        in_SZ[current_feature]++; in_SZ[n_features]++;
        recurse_2(z_child, x, z, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SZ[current_feature]--; in_SZ[n_features]--;
        return 0;
    }
}



// Recursion function for computing Anova 1
int recurse_3(int n,
            Matrix<double> &X,
            int i, int j,
            vector<int> &feature,
            vector<int> &child_left,
            vector<int> &child_right,
            vector<double> &threshold,
            vector<double> &value,
            int n_features,
            Tensor<double> &A,
            vector<int> &in_SX,
            vector<int> &in_SZ)
{
    int current_feature = feature[n];
    int x_child(0), z_child(0);
    // num_players := |S_{XZ}|

    // Arriving at a Leaf
    if (child_left[n] < 0)
    {
        // |S_X| = 0 so EACH element of S_Z gets a contribution
        if (in_SX[n_features]==0){
            int k(0), counter(0);
            while (counter < in_SZ[n_features]){
                if (in_SZ[k]){
                    A[i][j][k] -= value[n];
                    counter++;
                }
                k++;
            }
        }
        // |S_X| = 1 so the SINGLE element of S_X gets a contribution
        else if (in_SX[n_features]==1){
            int k(0);
            while (in_SX[k] == 0){
                k++;
            }
            A[i][j][k] += value[n];
        }

        // |S_Z| = 0 so EACH element of S_X gets a contribution
        if (in_SZ[n_features]==0){
            int k(0), counter(0);
            while (counter < in_SX[n_features]){
                if (in_SX[k]){
                    A[j][i][k] -= value[n];
                    counter++;
                }
                k++;
            }
        }
        // |S_Z| = 1 so the SINGLE element of S_Z gets a contribution
        else if (in_SZ[n_features]==1){
            int k(0);
            while (in_SZ[k] == 0){
                k++;
            }
            A[j][i][k] += value[n];
        }
        return 0;
    }

    // Find children of x and z
    if (X[i][current_feature] <= threshold[n]){
        x_child = child_left[n];
    } else {x_child = child_right[n];}
    if (X[j][current_feature] <= threshold[n]){
        z_child = child_left[n];
    } else {z_child = child_right[n];}

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_3(x_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in S_X U S_Z.
    // Hence we go down the correct edge to ensure that S_X and S_Z are kept disjoint
    if (in_SX[current_feature] || in_SZ[current_feature]){
        if (in_SX[current_feature]){
            return recurse_3(x_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
        }
        else{
            return recurse_3(z_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child if it is allowed
        if (in_SX[n_features] == 0 || in_SZ[n_features] <= 1){
            in_SX[current_feature]++; in_SX[n_features]++;
            recurse_3(x_child, X, i, j, feature, child_left, child_right,
                                threshold, value, n_features, A, in_SX, in_SZ);
            in_SX[current_feature]--; in_SX[n_features]--;
        }

        // Go to z's child if it is allowed
        if (in_SX[n_features] <= 1 || in_SZ[n_features] == 0){
            in_SZ[current_feature]++; in_SZ[n_features]++;
            recurse_3(z_child, X, i, j, feature, child_left, child_right,
                                threshold, value, n_features, A, in_SX, in_SZ);
            in_SZ[current_feature]--; in_SZ[n_features]--;
        }
        return 0;
    }
}




// Recursion function for computing A
int recurse_4(int n,
            Matrix<double> &X,
            int i, int j,
            vector<int> &feature,
            vector<int> &child_left,
            vector<int> &child_right,
            vector<double> &threshold,
            vector<double> &value,
            int n_features,
            Matrix<double> &A,
            vector<int> &in_SX,
            vector<int> &in_SZ)
{
    int current_feature = feature[n];
    int x_child(0), z_child(0);
    // num_players := |S_{XZ}|

    // Arriving at a Leaf
    if (child_left[n] < 0)
    {
        // Diagonal element
        if (i == j){
            A[i][i] += value[n];
            return 0;
        }
        // |S_X| = 0 so EACH element of S_Z gets a contribution
        if (in_SX[n_features]==0){
            A[i][j] += (1 - in_SZ[n_features]) * value[n];
        }
        // |S_X| = 1 so the SINGLE element of S_X gets a contribution
        else if (in_SX[n_features]==1){
            A[i][j] += value[n];
        }

        // |S_Z| = 0 so EACH element of S_X gets a contribution
        if (in_SZ[n_features]==0){
            A[j][i] += (1 - in_SX[n_features]) * value[n];
        }
        // |S_Z| = 1 so the SINGLE element of S_Z gets a contribution
        else if (in_SZ[n_features]==1){
            A[j][i] += value[n];
        }
        return 0;
    }

    // Find children of x and z
    if (X[i][current_feature] <= threshold[n]){
        x_child = child_left[n];
    } else {x_child = child_right[n];}
    if (X[j][current_feature] <= threshold[n]){
        z_child = child_left[n];
    } else {z_child = child_right[n];}

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_4(x_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in S_X U S_Z.
    // Hence we go down the correct edge to ensure that S_X and S_Z are kept disjoint
    if (in_SX[current_feature] || in_SZ[current_feature]){
        if (in_SX[current_feature]){
            return recurse_4(x_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
        }
        else{
            return recurse_4(z_child, X, i, j, feature, child_left, child_right,
                            threshold, value, n_features, A, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child if it is allowed
        if (in_SX[n_features] == 0 || in_SZ[n_features] <= 1){
            in_SX[current_feature]++; in_SX[n_features]++;
            recurse_4(x_child, X, i, j, feature, child_left, child_right,
                                threshold, value, n_features, A, in_SX, in_SZ);
            in_SX[current_feature]--; in_SX[n_features]--;
        }

        // Go to z's child if it is allowed
        if (in_SX[n_features] <= 1 || in_SZ[n_features] == 0){
            in_SZ[current_feature]++; in_SZ[n_features]++;
            recurse_4(z_child, X, i, j, feature, child_left, child_right,
                                threshold, value, n_features, A, in_SX, in_SZ);
            in_SZ[current_feature]--; in_SZ[n_features]--;
        }
        return 0;
    }
}



// Main function for Interventional TreeSHAP
Matrix<double> int_treeSHAP(Matrix<double> &X_f, 
                            Matrix<double> &X_b,
                            int* I_map, 
                            Matrix<int> &feature,
                            Matrix<int> &left_child,
                            Matrix<int> &right_child,
                            Matrix<double> &threshold,
                            Matrix<double> &value,
                            Matrix<double> &W)
    {
    // Setup
    int n_features = I_map[X_f[0].size()-1] + 1;
    int n_trees = feature.size();
    int Nz = X_b.size();
    int Nx = X_f.size();

    // Initialize the SHAP values to zero
    Matrix<double> phi_f_b(Nx, vector<double> (n_features, 0));
    progressbar bar(n_trees);
    // Iterate over all foreground instances
    for (int i(0); i < Nx; i++){
        // Iterate over all trees
        for (int t(0); t < n_trees; t++){
            // Iterate over all background instances
            for (int j(0); j < Nz; j++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);
                vector<double> phi(n_features, 0);

                // Start the recursion
                recurse(0, X_f[i], X_b[j], I_map, feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], W, n_features, phi, in_SX, in_SZ);

               // Add the contribution of the tree and background instance
                for (int f(0); f < n_features; f++){
                    phi_f_b[i][f] += phi[f];
                }
            }
        }
        // Rescale w.r.t the number of background instances
        for (int f(0); f < n_features; f++){
            phi_f_b[i][f] /= Nz;
        }
        bar.update();
    }
    return phi_f_b;
}




// Main function for Taylor-TreeSHAP
vector<Matrix<double>> taylor_treeSHAP(Matrix<double> &X_f, 
                                        Matrix<double> &X_b, 
                                        Matrix<int> &feature,
                                        Matrix<int> &left_child,
                                        Matrix<int> &right_child,
                                        Matrix<double> &threshold,
                                        Matrix<double> &value,
                                        Matrix<double> &W)
{
    // Setup
    int n_features = X_f[0].size();
    int n_trees = feature.size();
    int size_background = X_b.size();
    int size_foreground = X_f.size();

    // Initialize the taylor SHAP values to zero
    vector<Matrix<double>> phi_f_b(size_foreground, Matrix<double> (n_features, vector<double> (n_features, 0)));

    progressbar bar(size_foreground);
    // Iterate over all foreground instances
    for (int i(0); i < size_foreground; i++){
        // Iterate over all background instances
        for (int j(0); j < size_background; j++){
            // Iterate over all trees in the ensemble
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);
                vector<double> phi(n_features * n_features, 0);

                // Start the recursion
                recurse_2(0, X_f[i], X_b[j], feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], W, n_features, phi, in_SX, in_SZ);

                // Add the contribution of the tree and background instance
                for (int f1(0); f1 < n_features; f1++){
                    for (int f2(0); f2 < n_features; f2++){
                        phi_f_b[i][f1][f2] += phi[f1 * n_features + f2];
                    }
                }
            }
        }
        bar.update();
    }
    // Rescale taylor SHAP values w.r.t the number of background instances
    for (int i(0); i < size_foreground; i++){
        for (int f1(0); f1 < n_features; f1++){
            for (int f2(0); f2 < n_features; f2++){
                phi_f_b[i][f1][f2] /= size_background;
            }
        }
    }
    return phi_f_b;
}



// Main function for Taylor-TreeSHAP
vector<Matrix<double>> additive_treeSHAP(Matrix<double> &X, 
                                        Matrix<int> &feature,
                                        Matrix<int> &left_child,
                                        Matrix<int> &right_child,
                                        Matrix<double> &threshold,
                                        Matrix<double> &value)
{
    // Setup
    int n_features = X[0].size();
    int n_trees = feature.size();
    int N = X.size();

    // Initialize the taylor SHAP values to zero
    Tensor<double> A(N, Matrix<double> (N, vector<double> (n_features, 0)));

    progressbar bar(N*(N-1)/2);
    // Iterate over all foreground instances
    for (int i(0); i < N; i++){
        // Iterate over all background instances
        for (int j(i+1); j < N; j++){
            // Iterate over all trees in the ensemble
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);

                // Start the recursion
                recurse_3(0, X, i, j, feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], n_features, A, in_SX, in_SZ);
            }
            bar.update();
        }
    }
    return A;
}



// Main function for compute A recursively
Matrix<double> A_treeSHAP_recurse(Matrix<double> &X, 
                                Matrix<int> &feature,
                                Matrix<int> &left_child,
                                Matrix<int> &right_child,
                                Matrix<double> &threshold,
                                Matrix<double> &value)
        {
    // Setup
    int n_features = X[0].size();
    int n_trees = feature.size();
    int N = X.size();

    // Initialize the taylor SHAP values to zero
    Matrix <double> A(N, vector<double> (N, 0));

    progressbar bar(N*(N+1)/2);
    // Iterate over all foreground instances
    for (int i(0); i < N; i++){
        // Iterate over all background instances
        for (int j(i); j < N; j++){
            // Iterate over all trees in the ensemble
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);

                // Start the recursion
                recurse_4(0, X, i, j, feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], n_features, A, in_SX, in_SZ);
            }
            bar.update();
        }
    }
    return A;
}

# endif