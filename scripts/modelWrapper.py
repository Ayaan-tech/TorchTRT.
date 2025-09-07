
import torch
import json
import time
import psutil
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
import onnxruntime as ort


class ModelBenchmark:
    def __init__(self, model_path , model_type , device='cuda'):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.input_shape=None
        self.results = {
            "model_path": str(model_path),
            "model_type": model_type,
            "benchmark_date": datetime.now().isoformat(),
            "device": str(device),
            "metrics": {}
        }
    def load_model(self):
        """Load the model based on its type"""
        if self.model_type == "pytorch":
           self.model = torch.load(self.model_path).to(self.device)
           self.model.eval()
           if hasattr(torch, 'compile'):
               self.model = torch.compile(self.model)
           if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
        elif self.model_type == "tensorrt":
           
           with open(self.model, 'engine')as f:
               engine_data = f.read()
           logger = trt.Logger(trt.Logger.WARNING)
           runtime = trt.Runtime(logger)
           self.engine = runtime.deserialize_cuda_engine(engine_data)
           self.context = self.engine.create_execution_context()
           self.inputs = []
           self.outputs = []
           self.bindings = []
           self.stream = cuda.Stream()
           for bindings in self.engine:
               size = trt.volume(self.engine.get_binding_shape(bindings))
               dtype = trt.nptype(self.engine.get_binding_dtype(bindings))
               host_memory = cuda.pagelocked_empty_like(size, dtype)
               device_memory = cuda.mem_alloc_like(host_memory.nbytes)
               self.bindings.append(int(device_memory))
               if self.engine.binding_is_input(bindings):
                   self.inputs.append({'host': host_memory, 'device': device_memory})
               else:
                   self.outputs.append({'host': host_memory, 'device': device_memory})
        elif self.model_type == 'onnx':
            providers = ['CUDAExecutionProvider']
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            input_info = self.model.get_inputs()[0]
            self.input_shape = input_info.shape
    def infer(self , input_data):
        if self.model_type == "pytorch":
            with torch.no_grad():
                input_data = input_data.to(self.device)
                return self.model(input_data)
        elif self.model_type == "tensorrt":
            # Prepare input data for TensorRT
            self.inputs[0]['host'] = input_data
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async(bindings=self.bindings, stream=self.stream)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            return self.outputs[0]['host']
        elif self.model_type == "onnx":
            return self.model.run(None, {self.model.get_inputs()[0].name: input_data})[0]
