# Conda-Installation-Tutorial-Windows-
This is a tutorial for installing CUDA (v11.8) and cuDNN (8.6.9) to enable programming Pytorch with GPU. It also mentioned about the solution of unabling for Pytorch to detect the CUDA core.

**Suggestion: Install the CUDA first then install the corresponding CUDA-compatible Pytorch**

### 1. Check the compatibility:

1. Check whether your GPU is compatible with CUDA (and the supported CUDA version) at https://developer.nvidia.com/cuda-gpus, and update your driver.
2. Confirm the version of CUDA that you want and can install with (you can search for a specific version). In this tutorial, it's v11.8.
3. Find the compatible version of cuDNN (plug-in for optimizing AI training) at https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/. In this tutorial, it's 8.6.0.

### 2. Install CUDA:

If the high version fails to install, you can try the older version. <be>
Here I check all the checkboxes.

### 3. Verify the installation of CUDA:

First go the settings -> edit system environment variable (path) -> ```Environment Variables...``` <br>
-> Under System variables, ensure the existence of these two paths as below:
![image](https://github.com/TyBruceChen/Conda-Installation-Tutorial-Windows-/assets/152252677/1c2685ad-58b3-4188-b908-00753c04accf)

-> Also under this column, find ```Path```, double click to open, and add these two paths if they did not exist: 
![image](https://github.com/TyBruceChen/Conda-Installation-Tutorial-Windows-/assets/152252677/150d6a50-4352-46e4-a370-9d36d4968d53)

Second, to verify the system has already detected installed CUDA, type the command ```nvcc --version``` in the command prompt, the displayed version should match with the CUDA version you just installed.

### 4. Add Plug-in: cuDNN to CUDA:

Copy all the files (folders) of the downloaded cuDNN zip file that is compatible with your CUDA version, and paste them under the CUDA folder (in my case, it's ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\```) to finish those paste action at one time.

### 5. Installation of compatible Pytorch:

This step was where I got stuck and spent most of my time working it out.
Run the similar command to install cuda corresponding version of Pytorch: ```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118``` (here cu118 indicates the CUDA version is 11.8). The proper size of the main package of this installation should be near 2.4GB.

### 6. Verification:

```
import torch
torch.cuda.is_available()
```
should return ```True``` value

### 7. Some torch functions that may helps debugging if error occurs:

Click Here
