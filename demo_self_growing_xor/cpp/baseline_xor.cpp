// Self-growing XOR baseline in C++ with Eigen
// Equivalent to NOMA demo: no biases, Adam optimizer, same architecture
// Compile: g++ -O3 -I/path/to/eigen -o baseline_xor baseline_xor.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// Simple matrix class (no Eigen dependency for portability)
struct Matrix {
    std::vector<double> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    double& operator()(int i, int j) { return data[i * cols + j]; }
    double operator()(int i, int j) const { return data[i * cols + j]; }
    
    Matrix T() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result(j, i) = (*this)(i, j);
        return result;
    }
};

// Matrix operations
Matrix matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.cols; j++)
            for (int k = 0; k < A.cols; k++)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

Matrix sigmoid(const Matrix& A) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = 1.0 / (1.0 + std::exp(-A.data[i]));
    return result;
}

Matrix operator-(const Matrix& A, const Matrix& B) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] - B.data[i];
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] * B.data[i];
    return result;
}

Matrix operator*(const Matrix& A, double s) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] * s;
    return result;
}

Matrix operator/(const Matrix& A, const Matrix& B) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] / B.data[i];
    return result;
}

Matrix operator+(const Matrix& A, double s) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = A.data[i] + s;
    return result;
}

Matrix sqrt_m(const Matrix& A) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = std::sqrt(A.data[i]);
    return result;
}

Matrix pow_m(const Matrix& A, double p) {
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); i++)
        result.data[i] = std::pow(A.data[i], p);
    return result;
}

double mean(const Matrix& A) {
    double sum = 0.0;
    for (double v : A.data) sum += v;
    return sum / A.data.size();
}

// Adam optimizer state
struct AdamState {
    Matrix m, v;
    int t = 0;
    
    AdamState(int rows, int cols) : m(rows, cols), v(rows, cols) {}
    
    void update(Matrix& param, const Matrix& grad, double lr, 
                double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        t++;
        for (size_t i = 0; i < param.data.size(); i++) {
            m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * grad.data[i];
            v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * grad.data[i] * grad.data[i];
            double m_hat = m.data[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v.data[i] / (1.0 - std::pow(beta2, t));
            param.data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
    
    // Reset for new shape (growth)
    void reset(int rows, int cols) {
        m = Matrix(rows, cols);
        v = Matrix(rows, cols);
        t = 0;
    }
};

// Random normal initialization
void rand_normal(Matrix& M, double std, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, std);
    for (double& v : M.data) v = dist(rng);
}

int main(int argc, char* argv[]) {
    std::string csv_path = argc > 1 ? argv[1] : "out/cpp_loss.csv";
    std::ofstream csv(csv_path);
    csv << "step,loss,hidden\n";
    
    std::mt19937 rng(42);
    
    // XOR dataset
    Matrix X(4, 2), Y(4, 1);
    X(0,0)=0; X(0,1)=0; Y(0,0)=0;
    X(1,0)=0; X(1,1)=1; Y(1,0)=1;
    X(2,0)=1; X(2,1)=0; Y(2,0)=1;
    X(3,0)=1; X(3,1)=1; Y(3,0)=0;
    
    // Phase 1: hidden=2
    int hidden = 2;
    Matrix W1(2, hidden), W2(hidden, 1);
    rand_normal(W1, 0.5, rng);
    rand_normal(W2, 0.5, rng);
    
    AdamState adam_W1(2, hidden), adam_W2(hidden, 1);
    
    int step = 0;
    double lr1 = 0.05;
    
    // Phase 1: 200 iterations
    for (int iter = 0; iter < 200; iter++) {
        // Forward
        Matrix h = sigmoid(matmul(X, W1));
        Matrix pred = sigmoid(matmul(h, W2));
        Matrix err = pred - Y;
        double loss = mean(err * err);
        
        csv << step << "," << loss << "," << hidden << "\n";
        step++;
        
        // Backward
        Matrix dloss = err * (2.0 / X.rows);
        Matrix dout = dloss * pred * (pred * (-1.0) + 1.0);
        Matrix dW2 = matmul(h.T(), dout);
        
        Matrix dh = matmul(dout, W2.T());
        Matrix dh_act = dh * h * (h * (-1.0) + 1.0);
        Matrix dW1 = matmul(X.T(), dh_act);
        
        // Adam updates
        adam_W1.update(W1, dW1, lr1);
        adam_W2.update(W2, dW2, lr1);
    }
    
    // Growth: hidden 2 -> 16
    int hidden_big = 16;
    Matrix W1_new(2, hidden_big), W2_new(hidden_big, 1);
    rand_normal(W1_new, 0.3, rng);
    rand_normal(W2_new, 0.3, rng);
    
    // Copy old weights
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < hidden; j++)
            W1_new(i, j) = W1(i, j);
    for (int i = 0; i < hidden; i++)
        W2_new(i, 0) = W2(i, 0);
    
    W1 = W1_new;
    W2 = W2_new;
    hidden = hidden_big;
    
    // RESET Adam state (this is the plumbing cost!)
    adam_W1.reset(2, hidden);
    adam_W2.reset(hidden, 1);
    
    double lr2 = 0.12;
    
    // Phase 2: 342 iterations
    for (int iter = 0; iter < 342; iter++) {
        // Forward
        Matrix h = sigmoid(matmul(X, W1));
        Matrix pred = sigmoid(matmul(h, W2));
        Matrix err = pred - Y;
        double loss = mean(err * err);
        
        csv << step << "," << loss << "," << hidden << "\n";
        step++;
        
        // Backward
        Matrix dloss = err * (2.0 / X.rows);
        Matrix dout = dloss * pred * (pred * (-1.0) + 1.0);
        Matrix dW2 = matmul(h.T(), dout);
        
        Matrix dh = matmul(dout, W2.T());
        Matrix dh_act = dh * h * (h * (-1.0) + 1.0);
        Matrix dW1 = matmul(X.T(), dh_act);
        
        // Adam updates
        adam_W1.update(W1, dW1, lr2);
        adam_W2.update(W2, dW2, lr2);
    }
    
    csv.close();
    std::cout << "[cpp] wrote " << csv_path << " (total steps=" << step << ")\n";
    std::cout << "[cpp] note: growth RESETS optimizer state — unlike NOMA demo\n";
    
    return 0;
}
