# Configure pytorch for distributed GPU training on Ubuntu 18

In this tutorial, we'll try to realize distributed computing in DNN model training with ```torch.distributed``` (which is based on [NVIDIA Collective Communications Library (NCCL) function](https://developer.nvidia.com/nccl/nccl-legacy-downloads))

### Quick start example:

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
