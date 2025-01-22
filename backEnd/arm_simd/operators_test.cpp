#include "operators.h"
#include <iostream>
#include <random>
#include <chrono>

using namespace ml_compiler::backend;

// Helper function: generate random data
void generate_random_data(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

// Test MatMul operator
void test_matmul() {
    std::cout << "Testing MatMul operator..." << std::endl;
    
    const int M = 128;
    const int N = 128;
    const int K = 128;
    
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    
    generate_random_data(A.data(), M * K);
    generate_random_data(B.data(), K * N);
    
    MatMulOperator matmul(A.data(), B.data(), C.data(), M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    matmul.Execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "MatMul execution time: " << duration.count() << " microseconds" << std::endl;
}

// Test LayerNorm operator
void test_layernorm() {
    std::cout << "\nTesting LayerNorm operator..." << std::endl;
    
    const int batch_size = 32;
    const int hidden_size = 512;
    
    std::vector<float> input(batch_size * hidden_size);
    std::vector<float> output(batch_size * hidden_size);
    std::vector<float> gamma(hidden_size);
    std::vector<float> beta(hidden_size);
    
    generate_random_data(input.data(), batch_size * hidden_size);
    generate_random_data(gamma.data(), hidden_size);
    generate_random_data(beta.data(), hidden_size);
    
    LayerNormOperator layernorm(input.data(), output.data(), gamma.data(), beta.data(),
                               batch_size, hidden_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    layernorm.Execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "LayerNorm execution time: " << duration.count() << " microseconds" << std::endl;
}

// Test Softmax operator
void test_softmax() {
    std::cout << "\nTesting Softmax operator..." << std::endl;
    
    const int batch_size = 32;
    const int seq_length = 128;
    
    std::vector<float> input(batch_size * seq_length);
    std::vector<float> output(batch_size * seq_length);
    
    generate_random_data(input.data(), batch_size * seq_length);
    
    SoftmaxOperator softmax(input.data(), output.data(), batch_size, seq_length);
    
    auto start = std::chrono::high_resolution_clock::now();
    softmax.Execute();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Softmax execution time: " << duration.count() << " microseconds" << std::endl;
}

int main() {
    test_matmul();
    test_layernorm();
    test_softmax();
    return 0;
}
