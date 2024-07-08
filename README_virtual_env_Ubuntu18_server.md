## This is tutorial (learning record) of implementing pytorch on cuda in Ubuntu18-server system (remote PC (3080+2CPU+4GB_RAM))

### Categories:

### 1.Platform Info:

This tutorial is based on the cloud computer (Ubuntu-18-server-image) with 2-core-4-GHz 4G-RAM, 3080-GPU (10G), thanks [USTC (University of Science and Technology of China)'s CENI](https://ceni.ustc.edu.cn/land) providing the source.

### 2.Source switching for ubuntu18 (optional):
Back up the source.list first (optional)
```
sudo cp sources.list some_where_you_want
```
Replace the content of source.list by ```sudo nano /etc/apt/sources.list``` with (```ctl+a``` to save ```ctl+x``` to exit):
```
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```
* Note: For different versions of Ubuntu OS, the URL name has some slight differences in the source.list:
```
Ubuntu 22.04：jammy
Ubuntu 20.04：focal
Ubuntu 18.04：bionic
Ubuntu 16.04：xenial
```
Update necessities:
```
sudo apt-get update
sudo apt-get upgrade
```

### 3. Mounting extra disk space (optional):

### 4. SSH remote login through [FRP](https://github.com/fatedier/frp) (Fast Reverse Proxy) (recommended, optional):

### 5. Install NVIDIA driver:
Preparation: Install necessities first:
```
sudo apt install build-essential dkms
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```
Go to [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx), and find the driver that is compatible with your Nvidia GPU (in my case, it's GeForce RTX 3080). Then copy the link of .sh download (you may go through some agreements/acknowledge in the webpage), downloading by ```wget```. For 3080, it's 
```
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.90.07/NVIDIA-Linux-x86_64-550.90.07.run
```
Install the package:
```
sudo sh NVIDIA-Linux-x86_64-name_of_the_downloaded_file.run
```
Follow all the recommended options (there might be a disable of the original GPU kernel). To check the installation, use the command: ```nvidia-smi```.

### 6. Install CUDA:
Find the latest (recommended) version of CUDA that is compatible with your OS, for Ubuntu18, the latest version supported is [v11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local), also find the .sh (runfile (local)) file download link, download it through (v11.8)
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```
Install:
```
sudo sh cuda_11.8.0_name-of-the-downloaded-file_linux.run
```
Go through the options (only the CUDA-toolkit-related options, do not install the driver again). <be>
Add the CUDA path to the OS environment: 
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Modify the initialization file for the new-login terminal: ```sudo nano ~/.bashrc```, and add the above two lines at the end. <br>
Verify the CUDA installation:
```
nvcc -V
```

### 7. Add cuDNN plugins:
Find the compatible cuDNN plugins at [https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/linux-x86_64/](https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/linux-x86_64/), copy and download the compressed package (v9.20):
```
wget https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.2.0.82_cuda11-archive.tar.xz
```
Extract (-x) the specified tar file (-f):
```
tar -xf cudnn-linux-x86_64-9.2.0.82_cuda11-archive.tar.xz
```
Implement cuDNN plugins by superseding (adding) to CUDA files and then making them excutable:
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
Verify the CUDA installation:
```
nvcc -V
```
* Sometimes, there may be errors raised with ```Can't communicate with Nvidia drivers```, to solve this, you may try to install the driver again.

### 8. Install anaconda3:
Download anaconda3 at [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success), with 
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```
To install (recommend in ```root``` state): 
```
sudo bash <conda-installer-name>-latest-Linux-x86_64.sh
```
Add anaconda3 to the OS environment path (in my case, it's under ```root``` identity), then also add the two lines into ```~/.bashrc```:
```
export ANACONDA=/root/anaconda3/
export PATH=$PATH:/root/anaconda3/bin
```
Then logout from the terminal, and log in again, to verify installation, just type ```conda```.

### 9. Replace conda's download source (optional, this part is from a [CSDN blog](https://blog.csdn.net/qq_44827847/article/details/133315853)):
Replace with USTC's conda source:
```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
conda config --set show_channel_urls yes
```
Show the current conda source:
```
conda config --show-sources
```
Delete a specified channel:
```
conda config --remove channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```
Switch to the original channel:
```
conda config --remove-key channels
```

### 10. Replace pip installation source (optional):
Temporary:
```pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ xxx(the_name_of_python_package)```
Permanent:
```pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/```
* Verification: ```pip config get global.index-url```

### 11. Replace some download sources in Python packages & Implement convinient tools (optional): 
1. scp: ```scp -p ### usr_name@xxx.xxx.xxx.xxx:/source_location/file_name usr_name@xxx.xxx.xxx.xxx:/destination```, where ```###``` is the port number that is exposed from the FRP server.
2. git:
```sudo apt install git-all```
3. [gpustat](https://github.com/wookayin/gpustat):
```pip install gpustat``` (gpustat -cp --watch -i 1)
4. ...
