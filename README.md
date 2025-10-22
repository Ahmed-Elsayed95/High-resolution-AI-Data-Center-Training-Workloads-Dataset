# High-resolution-AI-Data-Center-Training-Workloads-Dataset

This dataset provides high-resolution, sub-second measurements of various AI training workloads executed on both single and multi-GPU nodes. It includes 32 training sessions performed on high-performance NVIDIA H100 and B200 8-GPU systems, as well as 40 sessions conducted on consumer-grade NVIDIA GeForce RTX 3060 GPUs. In total, the dataset comprises over 1.8 million samples, capturing detailed system-level metrics across diverse AI applications.

The figure below illustrates the overall scope of the AI training workload dataset, which is structured around three principal aspects: (1) platform and deployment scale, (2) applications and training objectives, and (3) AI architectures and hyperparameters. The experiments were designed based on these aspects to provide an accurate representation of AI training workloads across diverse computational environments.

<img width="1306" height="612" alt="Study Scope" src="https://github.com/user-attachments/assets/9d36238a-b8a4-41de-bbc6-5432ea06a4c0" />

*Figure: Overall scope of the AI training workload dataset illustrating platform deployment scale, applications, and AI architectures*

The platform and deployment scale aspect covers environments ranging from single-CPU and single-GPU systems to multi-GPU and multi–virtual CPU (vCPU) nodes. The applications and training objectives aspect includes a range of AI workloads such as image generation, text generation with LLMs, and feature forecasting. The workloads are assigned to appropriate environments according to their computational requirements, where tasks that demand substantial computing resources, such as image generation and LLM training, are executed in node-scale environments, while tasks requiring less computation, such as forecasting and image captioning, are conducted in single-machine environments.

## Machine Specifications

| Type | Local Single Machine Environment | Datacenter Node Environment - H100 | Datacenter Node Environment - B200 |
|------|-----------------------------------|------------------------------------|------------------------------------|
| **CPU** | 12th Gen Intel(R) Core(TM) i7-12700 2.10 GHz | Intel Xeon 208 vCPU | Intel Xeon 208 vCPU |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB) | 8x H100 SXM (80GB VRAM) | 8x B200 (180GB VRAM) |
| **RAM** | 32 GB | 1800 GB | 2900 GB |
| **OS** | Windows 10, 64-bit, x64-based processor | Ubuntu Server 22.04 arm64, x86-64 | Ubuntu Server 22.04 arm64, x86-64 |

*Table: Specifications of tested machines used in the experimental configurations*

## AI Architecture and Hyperparameters

| Task | AI Architecture and Hyperparameters Value |
|------|-------------------------------------------|
| **Single Machine Scale** | |
| Feature Forecasting | Batch size: (50,100,150), Model Size: (474K,1.6M,3.6M)<br>Input Sequence Length: (96,192,672), # Layers: (6,8,10) |
| Reinforcement Learning | Batch size: (150,250,350), Layer Size: (256,512,1024)<br>Input Sequence Length: (150,250,350), Input Type: (Feature, Sequence) |
| Image Classification | Batch size: (750,1500,2250), Image Size: (112,224,280)<br># Filters: (8,16,32), Optimizer Type: (Adam, SGD, RMS) |
| Text-Generation | Batch size: (32,128,512), Input Sequence Length: (100,250,500)<br>Embedding Dimension: (100,300,1000) |
| Image Captioning | Batch size: (128,256,1024), Layer Size: (512,1024,2048)<br>Embedding Dimension: (256,512,1024) |
| **H100/B200 Node Scale** | |
| Image Generation (Diffusion Models) | Batch size: (128,256,512), Image Size: (32,64,128)<br>Model Size: (107M,430M,1.7B) |
| Text-Generation (LLMs) | Batch size: (2,16,32), Parallelization Settings: ds(Z1,Z2,Z3)<br>Cutoff length: (1024,2048,4096), Model Size: (1B,3B,8B) |

*Table: AI architecture and hyperparameters used across different training tasks and computational scales*

## Dataset Structure

- **Total Training Sessions**: 72 sessions (32 high-performance + 40 consumer-grade)
- **Total Samples**: >1.8 million high-resolution measurements
- **Metrics Captured**: GPU utilization, memory usage, power consumption, temperature, throughput, and other system-level metrics
- **Temporal Resolution**: Sub-second measurements

## Applications Covered

- Image Generation (Stable Diffusion, GANs)
- Large Language Model (LLM) Training
- Time Series Forecasting
- Image Captioning
- And various other AI training workloads
