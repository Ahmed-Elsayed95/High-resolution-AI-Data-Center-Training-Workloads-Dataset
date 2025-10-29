<img width="1174" height="415" alt="image" src="https://github.com/user-attachments/assets/91632d59-2cef-4505-ba2e-687e072aadf8" />





# High-resolution-AI-Data-Center-Training-Workloads-Dataset

This dataset provides high-resolution, sub-second measurements of various AI training workloads executed on both single and multi-GPU nodes. It includes 32 training sessions performed on high-performance NVIDIA H100 and B200 8-GPU systems, as well as 40 sessions conducted on consumer-grade NVIDIA GeForce RTX 3060 GPUs. In total, the dataset comprises over 1.8 million samples, capturing detailed system-level metrics across diverse AI applications.

The figure below illustrates the overall scope of the AI training workload dataset, which is structured around three principal aspects: (1) platform and deployment scale, (2) applications and training objectives, and (3) AI architectures and hyperparameters. The experiments were designed based on these aspects to provide an accurate representation of AI training workloads across diverse computational environments. Each session provides a 15-minute time-series recording, sampled at 100 ms in the single-machine setup and 20 ms in the multi-GPU node environment. 

<img width="1306" height="612" alt="Study Scope" src="https://github.com/user-attachments/assets/9d36238a-b8a4-41de-bbc6-5432ea06a4c0" />

*Figure: Overall scope of the AI training workload dataset illustrating platform deployment scale, applications, and AI architectures*

The platform and deployment scale aspect covers environments ranging from single-CPU and single-GPU systems to multi-GPU and multi–virtual CPU (vCPU) nodes. The applications and training objectives aspect includes a range of AI workloads such as image generation, text generation with LLMs, and feature forecasting. The workloads are assigned to appropriate environments according to their computational requirements, where tasks that demand substantial computing resources, such as image generation and LLM training, are executed in node-scale environments, while tasks requiring less computation, such as forecasting and image captioning, are conducted in single-machine environments.

## Machine Specifications
The following table presents the hardware specifications of the single-GPU workstation and the multi-GPU compute node.

| Type | Local Single Machine Environment | Datacenter Node Environment - H100 | Datacenter Node Environment - B200 |
|------|-----------------------------------|------------------------------------|------------------------------------|
| **CPU** | 12th Gen Intel(R) Core(TM) i7-12700 2.10 GHz | Intel Xeon 208 vCPU | Intel Xeon 208 vCPU |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB) | 8x H100 SXM (80GB VRAM) | 8x B200 (180GB VRAM) |
| **RAM** | 32 GB | 1800 GB | 2900 GB |
| **OS** | Windows 10, 64-bit, x64-based processor | Ubuntu Server 22.04 arm64, x86-64 | Ubuntu Server 22.04 arm64, x86-64 |

*Table: Specifications of tested machines used in the experimental configurations*

## AI Architecture and Hyperparameters
The following table summarizes the AI model architectures and corresponding hyperparameters employed in each application.

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

## Metrics Captured

The dataset includes comprehensive system-level measurements collected through two distinct monitoring approaches:

### HWiNFO-based Monitoring (Single Machine)
- **CPU Metrics**: Core/package temperatures, frequencies, C-states, P-states, package power, memory controller load
- **GPU Metrics**: Utilization (compute and memory), core temperature, fan speeds, board power draw and limits, GPU/memory/SM clocks, voltages, ECC errors, active process information
- **System Metrics**: Motherboard voltages, fan speeds
- **Reporting Interval**: 100 ms minimum

### Python Package-based Monitoring (Node Scale)
- **CPU Metrics**: Utilization, frequency, energy and power statistics (via psutil and os packages)
- **GPU Metrics**: Utilization, memory usage, power demand (percentage and absolute values), temperature (via pynvml)
- **Reporting Interval**: 20 ms minimum (constrained by NVIDIA driver update frequency)

Both methods provide high-resolution, low-overhead monitoring with minimal interference to AI training workloads, capturing sub-second measurements across all hardware components.

## Dataset Structure

- **Total Training Sessions**: 72 sessions (32 high-performance + 40 consumer-grade)
- **Total Samples**: >1.8 million high-resolution measurements
- **Temporal Resolution**: Sub-second measurements
- **Data Format**: CSV with timestamped metrics

## Applications Covered

- Image Generation (Diffusion Models)
- Large Language Model (LLM) Training
- Time Series Forecasting
- Image Captioning
- Reinforcement Learning
- Image Classification
- Text Generation

## Authors 
- Ahmed Abd Elaziz Elsayed, EECS Department, York University, Toronto, Canada: https://scholar.google.ca/citations?user=PNaoAwsAAAAJ&hl=ar
- Abdullah Azhar Al-Obaidi: https://scholar.google.com/citations?hl=en&user=nMa8rEAAAAAJ&view_op=list_works&sortby=pubdate
- Hany E.Z. Farag (PI), EECS Department, York University, Toronto, Canada: https://smartgrid.eecs.yorku.ca/

## Citation
Please cite the following paper if you have used this dataset in your research/study ( if you can't access the paper, please reach out to the following email: elsayed7@yorku.ca)

Paper: Ahmed Abd Elaziz Elsayed, Abdullah Azhar Al-Obaidi, Hany E.Z. Farag. Characterization of high-resolution AI data center training workloads on single and multiple GPU nodes, 29 October 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-7943457/v1]

## LICENSE 
This Dataset is under the LICENSE: CC BY-NC-ND 4.0. 
Please check the LICENSE file for more details.


## State
![GitHub Repo views](https://visitor-badge.glitch.me/badge?page_id=Ahmed-Elsayed95/High-resolution-AI-Data-Center-Training-Workloads-Dataset)
![GitHub stars](https://img.shields.io/github/stars/Ahmed-Elsayed95/High-resolution-AI-Data-Center-Training-Workloads-Dataset)
![GitHub forks](https://img.shields.io/github/forks/Ahmed-Elsayed95/High-resolution-AI-Data-Center-Training-Workloads-Dataset)
