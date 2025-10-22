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
