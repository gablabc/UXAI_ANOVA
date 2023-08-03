// Utility functions and classes for TreeSHAP

#ifndef __UTILS
#define __UTILS

#include <vector>
#include <stack>

using namespace std;

// Custom Types
template <typename T>
using Matrix = vector<vector<T>>;
template <typename T>
using Tensor = vector<vector<vector<T>>>;


template<typename T>
Matrix<T> createMatrix(int n, int m, T* data){
    Matrix<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(vector<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back((data[m * i + j]));
        }
    }
    return mat;
}

template<typename T>
Tensor<T> createTensor(int n, int m, int l, T* data){
    Tensor<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(Matrix<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back(vector<T> ());
            for (int k(0); k < l; k++){
                mat[i][j].push_back((data[l * m * i + l * j + k]));
            }
        }
    }
    return mat;
}

template<typename T>
void printMatrix(Matrix<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            cout << mat[i][j] << " ";
        }
        cout << "\n";
    }
}

template<typename T>
void printTensor(Tensor<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    int l = mat[0][0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            for (int k(0); k < l; k++){
                cout << mat[i][j][k] << " ";
            }
            cout << "\n";
        }
        cout << "\n\n";
    }
}



void compute_W(Matrix<double> &W)
{
    int D = W.size();
    for (double j(0); j < D; j++){
        W[0][j] = 1 / (j + 1);
        W[j][j] = 1 / (j + 1);
    }
    for (double j(2); j < D; j++){
        for (double i(j-1); i > 0; i--){
            W[i][j] = (j - i) / (i + 1) * W[i+1][j];
        }
    }
}



class FeatureSet {
    // Class that represents the sets S_X and S_Z of features from the root to leaf
    // As one traverses the decision tree, the features are added and removed 
    // to these sets following the root-leaf path. 

    public:
        // default constructor
        FeatureSet(int d);

        // default destructor
        ~FeatureSet() = default;

        // cardinality |S_X|
        int size_SX();
        // cardinality |S_Z|
        int size_SZ();
        // i in SX ?
        int in_SX(int i);
        // i in SZ ?
        int in_SZ(int i);
        // get the vector of root-leaf features
        // vector<int> get_feature_path();
        // Add feature to the path
        // tag = 0 add nothing to S_X or S_Z
        // tag = 1 add to S_X
        // tag = 2 add to S_Z
        void add_feature(int feature, int tag);
        // Remove the d first features in the path
        // This will update SX and SZ accordingly
        // given the tags
        void remove_features(int d);
        bool is_path_empty();

    private:
        int d_;
        int size_SX_;
        int size_SZ_;
        vector<int> in_SX_;
        vector<int> in_SZ_;
        stack<int> feature_path_;
        stack<int> tags_;
};

inline FeatureSet::FeatureSet(int d) :
    d_(d),
    size_SX_(0),
    size_SZ_(0),
    in_SX_(vector<int> (d, 0)),
    in_SZ_(vector<int> (d, 0)),
    feature_path_(stack<int> ()),
    tags_(stack<int> ()) {}


inline int FeatureSet::size_SX() {
    return size_SX_;
}

inline int FeatureSet::size_SZ() {
    return size_SZ_;
}

inline int FeatureSet::in_SX(int i) {
    return in_SX_[i];
}

inline int FeatureSet::in_SZ(int i) {
    return in_SZ_[i];
}

// inline vector<int> FeatureSet::get_feature_vector() {
//     vector<int> res (0);
//     for (int i(0); i < d_; i++){
//         // i is in S_L
//         if (S_L_counts_[i] > 0){
//             res.push_back(i);
//         }
//     }
//     return res;
// }

inline void FeatureSet::add_feature(int feature, int tag) {
    feature_path_.push(feature);
    tags_.push(tag);
    // Add feature to SX
    if (tag == 1) {
        in_SX_[feature] = 1;
        size_SX_ += 1;
    }
    // Add feature to SZ
    else if (tag == 2) {
        in_SZ_[feature] = 1;
        size_SZ_ += 1;
    }
}

inline void FeatureSet::remove_features(int d) {
    int last_feature, last_tag;
    for (int i(0); i < d; i++){
        last_feature = feature_path_.top();
        last_tag = tags_.top();
        feature_path_.pop();
        tags_.pop();
        // Remove from SX
        if (last_tag == 1){
            in_SX_[last_feature] = 0;
            size_SX_ -= 1;
        }
        else if (last_tag == 2){
            in_SZ_[last_feature] = 0;
            size_SZ_ -= 1;
        }
    }
}

inline bool FeatureSet::is_path_empty() {
    return feature_path_.empty();
}

#endif