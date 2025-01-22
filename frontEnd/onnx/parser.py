import onnx
from typing import Dict, List, Any
import numpy as np

class ONNXParser:
    """ONNX 模型解析器"""
    
    def __init__(self):
        # LLaMA model common operations
        self.supported_ops = {
            'MatMul',           
            'Add',              
            'LayerNorm',        
            'Softmax',          
            'Attention',        
            'RoPE',            
            'SiLU',            
        }
        self.graph = None
        self.weights: Dict[str, np.ndarray] = {}
        
    def parse(self, model_path: str) -> Dict[str, Any]:
        """parsing ONNX model and extract necessary information"""
        model = onnx.load(model_path)
        self.graph = model.graph
        
        # 提取模型結構
        nodes = []
        for node in self.graph.node:
            if node.op_type not in self.supported_ops:
                raise ValueError(f"Unsupported operation type: {node.op_type}")
            
            nodes.append({
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: self._parse_attribute(attr) for attr in node.attribute}
            })
            
        # 提取權重
        for init in self.graph.initializer:
            self.weights[init.name] = onnx.numpy_helper.to_array(init)
            
        return {
            'nodes': nodes,
            'inputs': [input.name for input in self.graph.input],
            'outputs': [output.name for output in self.graph.output],
            'weights': self.weights
        }
        
    def _parse_attribute(self, attr):
        """analysis node attributes"""
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return list(attr.strings)
        else:
            raise ValueError(f"Unsupported attribute type: {attr.type}")
