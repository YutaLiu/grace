#ifndef ML_COMPILER_OPERATORS_H
#define ML_COMPILER_OPERATORS_H

#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <memory>

namespace ml_compiler {
namespace backend {

// Base operator interface
class Operator {
public:
    virtual ~Operator() = default;
    virtual void Execute() = 0;
};

// MatMul operator
class MatMulOperator : public Operator {
public:
    MatMulOperator(float* A, float* B, float* C, int M, int N, int K)
        : A_(A), B_(B), C_(C), M_(M), N_(N), K_(K) {}
    
    void Execute() override {
        // Matrix multiplication optimized with NEON SIMD
        for (int i = 0; i < M_; i += 4) {
            for (int j = 0; j < N_; j += 4) {
                float32x4_t c0 = vdupq_n_f32(0);
                float32x4_t c1 = vdupq_n_f32(0);
                float32x4_t c2 = vdupq_n_f32(0);
                float32x4_t c3 = vdupq_n_f32(0);
                
                for (int k = 0; k < K_; k++) {
                    float32x4_t a = vld1q_f32(&A_[i * K_ + k]);
                    float32x4_t b = vld1q_f32(&B_[k * N_ + j]);
                    
                    c0 = vmlaq_f32(c0, a, b);
                }
                
                vst1q_f32(&C_[i * N_ + j], c0);
            }
        }
    }

private:
    float* A_;
    float* B_;
    float* C_;
    int M_, N_, K_;
};

// LayerNorm operator
class LayerNormOperator : public Operator {
public:
    LayerNormOperator(float* input, float* output, float* gamma, float* beta,
                     int batch_size, int hidden_size)
        : input_(input), output_(output), gamma_(gamma), beta_(beta),
          batch_size_(batch_size), hidden_size_(hidden_size) {}
    
    void Execute() override {
        for (int b = 0; b < batch_size_; b++) {
            float* curr_input = input_ + b * hidden_size_;
            float* curr_output = output_ + b * hidden_size_;
            
            // Calculate mean
            float32x4_t sum_vec = vdupq_n_f32(0);
            for (int i = 0; i < hidden_size_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_input[i]);
                sum_vec = vaddq_f32(sum_vec, x);
            }
            float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                       vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
            float mean = sum / hidden_size_;
            
            // Calculate variance
            float32x4_t var_vec = vdupq_n_f32(0);
            float32x4_t mean_vec = vdupq_n_f32(mean);
            for (int i = 0; i < hidden_size_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_input[i]);
                float32x4_t diff = vsubq_f32(x, mean_vec);
                var_vec = vmlaq_f32(var_vec, diff, diff);
            }
            float var_sum = vgetq_lane_f32(var_vec, 0) + vgetq_lane_f32(var_vec, 1) +
                          vgetq_lane_f32(var_vec, 2) + vgetq_lane_f32(var_vec, 3);
            float var = var_sum / hidden_size_;
            
            // Normalize
            float32x4_t std_vec = vdupq_n_f32(std::sqrt(var + 1e-5f));
            for (int i = 0; i < hidden_size_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_input[i]);
                float32x4_t g = vld1q_f32(&gamma_[i]);
                float32x4_t b = vld1q_f32(&beta_[i]);
                
                float32x4_t norm = vdivq_f32(vsubq_f32(x, mean_vec), std_vec);
                float32x4_t result = vmlaq_f32(b, norm, g);
                vst1q_f32(&curr_output[i], result);
            }
        }
    }

private:
    float* input_;
    float* output_;
    float* gamma_;
    float* beta_;
    int batch_size_;
    int hidden_size_;
};

// Softmax operator
class SoftmaxOperator : public Operator {
public:
    SoftmaxOperator(float* input, float* output, int batch_size, int seq_length)
        : input_(input), output_(output), batch_size_(batch_size), seq_length_(seq_length) {}
    
    void Execute() override {
        for (int b = 0; b < batch_size_; b++) {
            float* curr_input = input_ + b * seq_length_;
            float* curr_output = output_ + b * seq_length_;
            
            // Find maximum value
            float32x4_t max_vec = vld1q_f32(curr_input);
            for (int i = 4; i < seq_length_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_input[i]);
                max_vec = vmaxq_f32(max_vec, x);
            }
            float max_val = std::max(std::max(vgetq_lane_f32(max_vec, 0),
                                            vgetq_lane_f32(max_vec, 1)),
                                   std::max(vgetq_lane_f32(max_vec, 2),
                                          vgetq_lane_f32(max_vec, 3)));
            
            // Calculate exp and sum
            float32x4_t sum_vec = vdupq_n_f32(0);
            float32x4_t max_val_vec = vdupq_n_f32(max_val);
            for (int i = 0; i < seq_length_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_input[i]);
                float32x4_t x_sub = vsubq_f32(x, max_val_vec);
                float32x4_t exp_val = exp_ps(x_sub);  // Need to implement exp_ps
                vst1q_f32(&curr_output[i], exp_val);
                sum_vec = vaddq_f32(sum_vec, exp_val);
            }
            float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                       vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
            
            // Normalize
            float32x4_t sum_vec_broadcast = vdupq_n_f32(sum);
            for (int i = 0; i < seq_length_; i += 4) {
                float32x4_t x = vld1q_f32(&curr_output[i]);
                float32x4_t normalized = vdivq_f32(x, sum_vec_broadcast);
                vst1q_f32(&curr_output[i], normalized);
            }
        }
    }

private:
    float* input_;
    float* output_;
    int batch_size_;
    int seq_length_;
    
    // NEON implementation of approximate exp function
    static float32x4_t exp_ps(float32x4_t x) {
        // Approximate exp using Taylor series
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t result = vaddq_f32(one, x);
        float32x4_t term = x;
        
        // Calculate x^2/2
        term = vmulq_f32(term, vmulq_f32(x, half));
        result = vaddq_f32(result, term);
        
        // Calculate x^3/6
        term = vmulq_f32(term, vmulq_f32(x, vdupq_n_f32(1.0f/3.0f)));
        result = vaddq_f32(result, term);
        
        // Calculate x^4/24
        term = vmulq_f32(term, vmulq_f32(x, vdupq_n_f32(1.0f/4.0f)));
        result = vaddq_f32(result, term);
        
        return result;
    }
};

} // namespace backend
} // namespace ml_compiler

#endif // ML_COMPILER_OPERATORS_H
