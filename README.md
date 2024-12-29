# 🌍 Geoscience AI Showdown 🌋

**A Python project for benchmarking and comparing LLMs for geoscience tasks.**

<div align="center">

[![Programming Language](https://img.shields.io/badge/Python-3.12-blue)]()
[![Code Style](https://img.shields.io/badge/code%20style-pep8-orange.svg)]()
[![Stability](https://img.shields.io/badge/stability-alpha-f4d03f.svg)]()

</div>

---

## 🚀 About GeoAI Showdown

Geoscience AI Showdown is a Python-based toolkit designed to benchmark and compare the performance of Large Language Models (LLMs) on various geoscience tasks. This project is a step forward in integrating AI and geoscience, enabling the evaluation of state-of-the-art models in tasks like geoscience reasoning, Q&A, and more.

---

## 🛠️ Setup

### ✅ Prerequisites

#### 1. Microsoft C++ Build Tools
Download the installer from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
During installation, select the following workloads:
  - Desktop development with C++
  - MSVC (Microsoft C++ compiler)
  - C++ CMake tools for Windows
  - Windows 11 SDK

#### 2. CUDA Toolkit
Verify CUDA Installation:
```bash
nvcc --version
```
If CUDA is not installed download it from [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

Check the current Nvidia driver version:
```bash
nvidia-smi
```
verify the driver version and [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package).
If the driver is incompatible or not installed, download the correct driver from [Nvidia Drivers](https://www.nvidia.com/en-us/drivers/).

### 🖥️ Environment Configuration

#### 1. Create a Virtual Environment
Inside the project directory, create a virtual environment:
```bash
python -m venv .venv
```
Then, activate the virtual environment:
```bash
.venv\Scripts\activate
```

#### 2. Install Dependencies
Install Poetry to manage dependencies:
```bash
pip install poetry
```

Then, install the project dependencies:
```bash
poetry install --with dev
```

#### 3. Set Environment Variables
OpenAI API Key
```bash
setx OPENAI_API_KEY "your-api-key-here"
```

HuggingFace Access Token
```bash
huggingface-cli login
```
Ensure access to the following gated repositories:
- [Meta-LLaMA 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)
- [Meta-LLaMA 3.2](https://huggingface.co/collections/meta-llama/metas-llama-32-language-models-and-evals-675bfd70e574a62dd0e40586)

## 🌀 Usage

### Run the CLI
```bash
gas --help
```

## 🧪 Features
- Benchmarking: Compare the performance of LLMs on geoscience-specific tasks.
- Custom Evaluation: Test LLMs against geoscience dictionaries, QA tasks, and more.
- GPU Support: Optimized for CUDA-enabled systems.
