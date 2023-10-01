# mldev

Notes and examples on setting up a reproducible ML dev environment.

# Core Components (Tested Version)

* Ubuntu (22.04)
* NVIDIA driver (535)
* Docker (24.06)
* NVIDIA Container Toolkit (1.14.2)
* NVIDIA GPU Cloud (NGC) Container (nvcr.io/nvidia/pytorch:23.09-py3)

# Setup 

## Install NVIDIA driver

```bash
sudo apt-get install nvidia-driver-535
```

After installing the NVIDIA driver, the `nvidia-smi` command should show CUDA version 12.2, 
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.113.01             Driver Version: 535.113.01   CUDA Version: 12.2     |
```


## Install Docker

https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04

After installing Docker, you should be able to run the hello world image, 

```bash
docker run hello-world
```


## Install NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```bash
install-nct.sh
```

After installing NVIDIA Container Toolkit, you should be able to run `nvidia-smi` from within a docker container,

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## NVIDIA Docker Containers

NVIDIA GPU Cloud (NGC) provides many Docker containers,

https://catalog.ngc.nvidia.com/orgs/nvidia/containers

We tested with the `nvcr.io/nvidia/pytorch:23.09-py3` container

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags


A set of default base flags for docker run are,
* `--gpus all`
* `--ipc=host` or `--shm-size 1gb`
* `--ulimit memlock=-1`
* `--ulimit stack=67108864`


An example interactive session that will remove the container on exit is, 

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:23.09-py3
```

## Customize NVIDIA base Docker container

see `Dockerfile`


## Notes on efficient loading / training / inference

https://huggingface.co/docs/transformers/perf_train_gpu_one
https://huggingface.co/docs/transformers/perf_infer_gpu_one
https://huggingface.co/docs/transformers/perf_infer_cpu

https://huggingface.co/blog/hf-bitsandbytes-integration
https://huggingface.co/blog/4bit-transformers-bitsandbytes

https://huggingface.co/docs/transformers/main_classes/quantization


## HF Llama models

https://huggingface.co/blog/llama2

