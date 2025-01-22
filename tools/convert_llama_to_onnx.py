import torch
import torch.nn as nn
import os
import onnx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # x shape: [sequence_length, batch_size, input_dim]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

def convert_model_to_onnx(output_path="model/simple_transformer.onnx",
                         sequence_length=32,
                         batch_size=1,
                         input_dim=512):
    print("Creating simple Transformer model...")
    
    # Create model
    model = SimpleTransformer(input_dim=input_dim)
    model.eval()
    
    # Prepare example input
    dummy_input = torch.randn(sequence_length, batch_size, input_dim)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Starting ONNX conversion...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'sequence', 1: 'batch'},
                'output': {0: 'sequence', 1: 'batch'}
            },
            opset_version=17
        )
    
    print(f"Model saved to: {output_path}")
    
    # Analyze operators in the model
    print("\nAnalyzing operators in the model...")
    model = onnx.load(output_path)
    
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    
    print("\nOperators used in the model:")
    print("\n".join(sorted(ops)))
    
if __name__ == "__main__":
    convert_model_to_onnx()
