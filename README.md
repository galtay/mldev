# mldev

One take on a docker powered ML dev environment.

# Setup Nvidia ML dev environment

These notes are tested on a system running Ubuntu 22.04.

## Install Docker

https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04


## Install NVIDIA driver

```bash
sudo apt-get install nvidia-driver-535
```

this should provide CUDA Version: 12.2


## Install NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```bash
install-nct.sh
```

test installation,

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## NVIDIA base Docker containers

NGC is a container registry

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

testing was done with `nvcr.io/nvidia/pytorch:23.09-py3`


Recommended base flags are, 

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```

For interactive session with cleanup afterward, 

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

