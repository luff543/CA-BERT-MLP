# CA-BERT-MLP

This document describes how to install code on Ubuntu 16.04/18.04.

# Download Anaconda and Install

Go to **[here](https://www.anaconda.com/products/distribution)** and download **Anaconda3-2022.05-Linux-x86_64.sh**

## Requirement install

    $ source /home/<user>/anaconda3/etc/profile.d/conda.sh
    
    $ conda create --name ca_bert_mlp_env python=3.7
    $ conda activate ca_bert_mlp_env
    
    $ conda install cudatoolkit=10.0.130 -y
    $ conda install cudnn=7.6.4 -y
    $ conda install tensorflow=1.14.0 -y
    $ conda install tensorflow-gpu=1.14.0 -y
    $ conda install tensorflow-hub=0.8.0 -y
    $ conda install pandas
    $ pip install scikit-learn==0.24.2
    $ pip install pyzmq==19.0.1

# Code

### Train and evaluation model

* Train **Sentence-level event field detection model**

  ``` 
  python train.py

* Train **Multi-task word-level event field extraction model**

  Usage see **[here](https://github.com/luff543/BERT-event-information-extractor)** 