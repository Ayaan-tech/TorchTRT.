
## TorchTRT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.10.0.31-green.svg)](https://developer.nvidia.com/tensorrt)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5.svg?logo=kubernetes)](https://kubernetes.io/)

A high-performance MLOps benchmarking suite designed to quantify and visualize the impact of optimization techniques on deep learning inference. This project demonstrates a complete pipeline from a raw PyTorch model to a scalable, Kubernetes-orchestrated inference microservice, leveraging NVIDIA's full stack (TensorRT, Triton).

## 🚀 Project Objectives & Purpose

The core mission of **InferenceForge** is to provide a clear, reproducible, and production-grade framework for evaluating inference performance. It answers critical questions:
*   **What are the tangible gains** from precision quantization (FP32 -> FP16/INT8) using TensorRT?
*   **What is the overhead** of deploying a model as a microservice vs. running it locally?
*   **How can we reliably orchestrate** and scale high-performance inference in a modern DevOps environment?

This project serves as a practical demonstration of skills essential for ML/LLMOps and AI Infrastructure roles, focusing on low-latency, high-throughput deployment.

---

## 📊 Performance Highlights

| Optimization Stage | Average Latency | Throughput (FPS) | Model Size | Key Enabler |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (FP32)** | ~45 ms | ~22 FPS | 18.8 MB | Baseline |
| **TensorRT (FP32)** | ~25 ms | ~40 FPS | 9.4 MB | Graph Optimization |
| **TensorRT (FP16)** | -  | - | -  | **TensorRT Quantization** |
| **+ Triton Server** | -  | -  | - | **Triton Inference Server** |

*Results measured on NVIDIA GeForce RTX 3050 Laptop GPU. [See full benchmark details](./results/benchmark_results.md).*

---

## 🧩 What's in the Box? (Features)

*   **🧪 End-to-End Optimization Pipeline:** Scripts to convert PyTorch -> ONNX -> TensorRT (FP32/FP16/INT8) engines.
*   **⏱️ Rigorous Benchmarking:** Uses `torch.cuda.Event` for nanosecond-precision GPU timing across various batch sizes and precisions.
*   **🐳 Full Dockerization:** Containerized environments for 100% reproducibility of results.
*   **⚙️ Triton Inference Server:** Deployment of optimized models as a high-performance inference microservice with configurable batching and concurrent execution.
*   **☸️ Kubernetes Orchestration:** A sample Kubernetes deployment manifest for scaling the Triton service in a production-like environment.
*   **📈 Automated Visualization:** Python scripts to generate performance comparison charts (FPS vs. Latency) from benchmark logs.

---

## ✅ Project Checklist

This project is designed to be a complete journey. The main goals are:

- [x] **Phase 1: Core Optimization**
    - [x] Export YOLOv8n model to ONNX format.
    - [x] Compile ONNX model to TensorRT FP32 and FP16 engines.
    - [x] Implement accurate benchmarking with CUDA Events.
    - [x] Document performance gains (FPS, Latency, Model Size).

- [x] **Phase 2: Reproducibility & DevOps**
    - [x] Dockerize the entire benchmarking environment.
    - [x] Learn and implement Triton Inference Server model deployment.
    - [x] Write a client script to benchmark Triton server performance.

- [x] **Phase 3: Production Readiness**
    - [x] Create Kubernetes manifests for deploying Triton.
    - [x] Deploy and test the inference service on a local Kubernetes cluster (Minikube).
    - [x] Finalize comprehensive documentation and visualizations.

---

## 🛠️ Installation & Quick Start

### Prerequisites
*   **NVIDIA GPU** with **CUDA >= 12.0** and **cuDNN** installed.
*   **Docker** and **NVIDIA Container Toolkit**.
*   Python 3.8+ with `pip`.
*   
## 🔗 Resources & Acknowledgements
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)  
- [NVIDIA Triton Inference Server Docs](https://github.com/triton-inference-server/server)  
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)  
- [Official Kubernetes Documentation](https://kubernetes.io/docs/)
