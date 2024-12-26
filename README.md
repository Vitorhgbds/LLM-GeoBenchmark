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
- Download it from NVIDIA CUDA Toolkit.

1.  Export your OPENAI_API_KEY  

Windows

```bash
setx OPENAI_API_KEY "api-key-here"
```
Linux
```bash
export OPENAI_API_KEY "api-key-here"
```
2. Export HuggingFace Acess token
```bash
huggingface-cli login
```

3. Poetry install  
```bash
pip install poetry
```  

4. install dependencies
``` bash
poetry install --with dev
```


