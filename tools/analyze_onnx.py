import onnx
from collections import Counter
import sys
import os

def analyze_onnx_model(model_path):
    print(f"分析ONNX模型: {model_path}")
    
    # 載入模型
    model = onnx.load(model_path)
    
    # 收集所有算子
    op_counter = Counter()
    for node in model.graph.node:
        op_counter[node.op_type] += 1
    
    # 打印算子統計
    print("\nOperation Counter:")
    print("-" * 40)
    print("Operation               Count")
    print("-" * 40)
    for op, count in op_counter.most_common():
        print(f"{op:<20} {count:>10}")
    
    # 檢查輸入和輸出
    print("\nModel Input:")
    for input in model.graph.input:
        print(f"- {input.name}: {input.type.tensor_type.elem_type}")
        
    print("\nModel Output:")
    for output in model.graph.output:
        print(f"- {output.name}: {output.type.tensor_type.elem_type}")
    
    # 檢查權重
    print("\nModel Weights Count :", len(model.graph.initializer))
    
    return op_counter

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_onnx.py <onnx_model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
        
    analyze_onnx_model(model_path)
