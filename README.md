# GeoAI Showdown

# Setup

## Windows System Requirements

### Install Microsoft C++ Build Tools  
- Visit the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 
- Download the installer (`Build Tools for Visual Studio`).
- Run the installer, and during installation, ensure you select the following workloads:
    - Desktop development with C++ (recommended for most Python packages).
    - Include MSVC (Microsoft C++ compiler), C++ CMake tools for Windows and Windows 11 SDK (or the latest version)

### Install CUDA Toolkit
- First check your CUDA version with the following command:
   ``` bash
    nvcc --version
    ``` 
     - If you do not have CUDA installed, install it using the following command:
     - Download it from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

- Check your nvidia driver version 
   ``` bash
   nvidia-smi
   ```
    - This should display your GPU details. 
    If it does not, Check CUDA compatibility with your driver version [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package).
    - If your driver is not compatible, download the correct driver from [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) and install it.
    
## Configure Environment Variables
- OpenAi API Key
    ``` bash
    setx OPENAI_API_KEY "api-key-here"
    ```
- HuggingFace Access Token
    ``` bash
    huggingface-cli login
    ```
    - Follow the instructions to login and get your access token.
    - You should have access for the following gated repositories:
        - [meta-llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)
        - [meta-llama 3.2](https://huggingface.co/collections/meta-llama/metas-llama-32-language-models-and-evals-675bfd70e574a62dd0e40586)

## Create a Virtual Environment
1. Clone the repository
2. Go to the project directory and create a virtual environment:
    ``` bash
    python -m venv .venv
    ```
3. Activate the virtual environment:
    ``` bash
    .venv\Scripts\activate
    ```
## Install Dependencies
4. Install poetry:
    ``` bash
    pip install poetry
    ```
5. Install dependencies:
    ``` bash
    poetry install --with dev
    ```

# Usage
- To run the main script:
    ``` bash
    python .\gas\main.py
    ```
- To run the tests:
    ``` bash
    pytest -vvv
    ```