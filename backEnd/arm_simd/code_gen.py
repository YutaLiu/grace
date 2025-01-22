from typing import Dict, List, Any
import numpy as np

class ARMSIMDCodeGen:
    """ARM SIMD 代碼生成器"""
    
    def __init__(self):
        self.simd_intrinsics = {
            'MatMul': self._gen_matmul_simd,
            'Add': self._gen_add_simd,
            'LayerNorm': self._gen_layernorm_simd,
            'Softmax': self._gen_softmax_simd,
        }
        
    def generate(self, op_info: Dict[str, Any]) -> str:
        """Generate ARM SIMD code"""
        op_type = op_info['op_type']
        if op_type not in self.simd_intrinsics:
            raise ValueError(f"Unsupported operation: {op_type}")
            
        return self.simd_intrinsics[op_type](op_info)
        
    def _gen_matmul_simd(self, op_info: Dict[str, Any]) -> str:
        """Generate matrix multiplication SIMD code"""
        code = """
        #include <arm_neon.h>
        
        void matmul_simd(float* A, float* B, float* C, int M, int N, int K) {
            for (int i = 0; i < M; i += 4) {
                for (int j = 0; j < N; j += 4) {
                    float32x4_t c0 = vdupq_n_f32(0);
                    float32x4_t c1 = vdupq_n_f32(0);
                    float32x4_t c2 = vdupq_n_f32(0);
                    float32x4_t c3 = vdupq_n_f32(0);
                    
                    for (int k = 0; k < K; k++) {
                        float32x4_t a = vld1q_f32(&A[i * K + k]);
                        float32x4_t b = vld1q_f32(&B[k * N + j]);
                        
                        c0 = vmlaq_f32(c0, a, b);
                    }
                    
                    vst1q_f32(&C[i * N + j], c0);
                }
            }
        }
        """
        return code
        
    def _gen_add_simd(self, op_info: Dict[str, Any]) -> str:
        """Generate vector addition SIMD code"""
        code = """
        #include <arm_neon.h>
        
        void add_simd(float* A, float* B, float* C, int size) {
            int i;
            for (i = 0; i <= size - 4; i += 4) {
                float32x4_t a = vld1q_f32(&A[i]);
                float32x4_t b = vld1q_f32(&B[i]);
                float32x4_t c = vaddq_f32(a, b);
                vst1q_f32(&C[i], c);
            }
            
            // 處理剩餘元素
            for (; i < size; i++) {
                C[i] = A[i] + B[i];
            }
        }
        """
        return code
        
    def _gen_layernorm_simd(self, op_info: Dict[str, Any]) -> str:
        """Generate layer normalization SIMD code"""
        code = """
        #include <arm_neon.h>
        #include <math.h>
        
        void layernorm_simd(float* input, float* output, float* gamma, float* beta,
                           int hidden_size) {
            // 計算均值
            float32x4_t sum = vdupq_n_f32(0);
            for (int i = 0; i < hidden_size; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                sum = vaddq_f32(sum, x);
            }
            float mean = (vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                         vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3)) / hidden_size;
                         
            // 計算方差
            float32x4_t var_sum = vdupq_n_f32(0);
            float32x4_t mean_vec = vdupq_n_f32(mean);
            for (int i = 0; i < hidden_size; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                float32x4_t diff = vsubq_f32(x, mean_vec);
                var_sum = vmlaq_f32(var_sum, diff, diff);
            }
            float variance = (vgetq_lane_f32(var_sum, 0) + vgetq_lane_f32(var_sum, 1) +
                            vgetq_lane_f32(var_sum, 2) + vgetq_lane_f32(var_sum, 3)) / hidden_size;
                            
            // 正規化
            float32x4_t std_vec = vdupq_n_f32(sqrt(variance + 1e-5));
            for (int i = 0; i < hidden_size; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                float32x4_t g = vld1q_f32(&gamma[i]);
                float32x4_t b = vld1q_f32(&beta[i]);
                
                float32x4_t norm = vdivq_f32(vsubq_f32(x, mean_vec), std_vec);
                float32x4_t result = vmlaq_f32(b, norm, g);
                vst1q_f32(&output[i], result);
            }
        }
        """
        return code
        
    def _gen_softmax_simd(self, op_info: Dict[str, Any]) -> str:
        """Generate Softmax SIMD code"""
        code = """
        #include <arm_neon.h>
        #include <math.h>
        
        void softmax_simd(float* input, float* output, int size) {
            // 找最大值
            float32x4_t max_vec = vld1q_f32(input);
            for (int i = 4; i < size; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                max_vec = vmaxq_f32(max_vec, x);
            }
            float max_val = fmaxf(fmaxf(vgetq_lane_f32(max_vec, 0),
                                      vgetq_lane_f32(max_vec, 1)),
                                fmaxf(vgetq_lane_f32(max_vec, 2),
                                     vgetq_lane_f32(max_vec, 3)));
                                     
            // 計算exp並求和
            float32x4_t sum = vdupq_n_f32(0);
            float32x4_t max_val_vec = vdupq_n_f32(max_val);
            for (int i = 0; i < size; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                float32x4_t x_sub = vsubq_f32(x, max_val_vec);
                
                // exp近似計算
                float32x4_t exp_val = exp_ps(x_sub);
                vst1q_f32(&output[i], exp_val);
                sum = vaddq_f32(sum, exp_val);
            }
            float sum_val = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                           vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
                           
            // 正規化
            float32x4_t sum_vec = vdupq_n_f32(sum_val);
            for (int i = 0; i < size; i += 4) {
                float32x4_t x = vld1q_f32(&output[i]);
                float32x4_t normalized = vdivq_f32(x, sum_vec);
                vst1q_f32(&output[i], normalized);
            }
        }
        """
        return code
