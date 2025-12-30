# AGNES: Accelerating Storage-based GNN Training for Graph Neural Networks

This repository provides a reference implementation of *AGNES* as described in the following paper:
> AGNES: Accelerating Storage-based GNN Training for Graph Neural Networks<br>
> Myung-Hwan Jang, Jeong-Min Park, Yunyong Ko, and Sang-Wook Kim*<br>
> The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)<br>

This project is written in standard python, C++ and CUDA.

# Requirements
The code has been tested running under Python 3.8.10 and Ubuntu 22.04.2. The required packages are as follows:
 - Boost 1.61+
 - G++ 5.0+
 - CUDA 12.01+


# Installation guide

(1) Unzip AGNES.zip into the preferred directory

(2) Install necessary Python modules
- PyTorch with version of >= 1.9.0. Visit here for details.
- pip3 install tqdm
- pip3 install ogb
- PyG. Visit here (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details.
- Ninja
  ```bash
  sudo wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
  sudo unzip ninja-linux.zip -d /usr/local/bin/
  sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
  ```
  
(3) Prepare dataset 
```bash
./preprocessing datapath /home/nvme1/Ginex/ data ogbn_edge_list.txt file ../ogbn1M_64bit/Ogbn memoryuse_mb 65536 threads 16
```
% preprocess parameters
 - datapath: path to input file
 - data : raw input file name (txt file, edge list)
 - file : preprocessed input file name
 - memoryuse_mb : maximum available size of main memory in megabytes
 - threads : maximum number of threads
 - iters : maximum number of iterations

 
(4) Run 
```bash
python3 run_agnes.py --dataset Ogbn --dataset_path /home/nvme1/ogbn128K_64bit/ --sb-size 1024 --num-workers 16 --sizes 10,10,10
```
% run parameters
 - num-epochs: number of epochs to train the model
 - num-workers: number of worker threads
 - num-hiddens: number of hidden units in the model's layers
 - dataset: name of the dataset to be used for training and testing
 - sizes: comma-separated values representing the sizes of neighborhoods to sample at each GNN layer
 - model: model type to be used for training (sage|gcn|gat)
 - buffer-size: maximum available size of buffer in megabytes
    
