# Configure pytorch for distributed GPU training on Ubuntu 18

In this tutorial, we'll try to realize distributed computing in DNN model training with ```torch.distributed``` (which is based on [NVIDIA Collective Communications Library (NCCL) function](https://developer.nvidia.com/nccl))

### Quick start example:

If there's no NCCL installed on your computer, download here: https://developer.nvidia.com/nccl/nccl-legacy-downloads

Download the official example by pytorch: [https://github.com/pytorch/examples/tree/main/imagenet](https://github.com/pytorch/examples/tree/main/imagenet). The example directory should contain the following folders: ```gpu  imagenet  main.py```, where ```imagenet``` should contain the ```train, val, test, ...``` ([download here](https://www.image-net.org/download.php)).

Multiple nodes (distributed GPU) training: <br>
Node 0 (main/center node):

    python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
    
Node 1:

    python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]

where ```IP_OF_NODE0``` is the accessible IP address of the main (center) node, ```FREEPORT``` is a random accessible port specified by the main (center) node. ```Main (center) node``` is marked with ```rank``` number ```0```. ```world-size``` indicates the scope (number) of the participated nodes. ```rank``` serves as the index to identify each node.

Nvidia NCCL test case: [https://github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)

### Common Problems:
Display the detailed debug information of nccl: ```export NCCL_DEBUG=INFO```
1. ethernet card identify error:
```
[0] misc/socket.cc:533 NCCL WARN socketPollConnect: Connect to 128::1d<40861> returned 113(No route to host) errno 11(Resource temporarily unavailable)
...
[0] misc/socket.cc:533 NCCL WARN socketPollConnect: Connect to 9001::281<50557> returned 113(No route to host) errno 11(Resource temporarily unavailable)
...
...
Last error:
[Proxy Service 0] Failed to execute operation Connect from rank 0, retcode 3
```
This is due to multiple IP addresses (```128::1d<40861>``` and ```9001::281<50557>``` in this case) being recognized as accessible paths between nodes, however inaccessible. Thus, we need to specify the correct accessible IP address's ethernet card. First, check with ```ifconfig``` to find the accessible ethernet card's cognomen. Then specify it to nccl by ```export NCCL_SOCKET_IFNAME='ens3 (in my case)'```. To make it applicable at each session: add the ```export``` command to ```~/.bashrc```.

#### Step-by-step Guide through DistributedData Parallele (DDP) with pytorch (torch.nn.parallel.DistributedDataParallel): 
https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

#### Official quick start tutorial with torch: 
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

### Huggingface Accelerate (a library that specializes in training models with different devices):
##### Articulate Tutorials/Manuals of Huggingface Accelerate:
1. Overall tutorial: https://huggingface.co/blog/zh/pytorch-ddp-accelerate-transformers
2. Write accelerate-enabled code with distributed learning config: https://huggingface.co/docs/accelerate/v0.12.0/en/basic_tutorials/launch
3. Manual of the ```accelerate``` command: https://huggingface.co/docs/accelerate/v0.34.0/en/package_reference/cli#accelerate-launch

Basic syntax: <BR>```accelerate launch {--accelerate_configuration_command} {your_python_file.py} {the argument you want to pass in to the python_file, e.g.: --arg1 --arg2}```
<BR>
To modify ```your_python_file.py``` able with ```Accelerate```: <BR>
1. Adding lines to titivate the original code (model, optimizer, dataset, etc (refer to above-mentioned tutorials))
2. encapsulate your training program into main() function so that can be run from ```accelerate``` CLI.
