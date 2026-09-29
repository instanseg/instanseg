# Installation

InstanSeg is published on PyPI as [`instanseg-torch`](https://pypi.org/project/instanseg-torch/)
and supports Python 3.9 and later.

## Install from PyPI

For a minimal installation, with everything needed to run inference:

```console
$ pip install instanseg-torch
```

InstanSeg also provides optional extras:

`io`
: Extra image readers (bioio, tiffslide, slideio) and GeoJSON export.

`full`
: Everything in `io`, plus the dependencies needed to train models, export them to the
  BioImage Model Zoo, and run the example notebooks.

```console
$ pip install "instanseg-torch[io]"
$ pip install "instanseg-torch[full]"
```

:::{tip}
We recommend installing into a fresh virtual environment, for example with
[mamba](https://mamba.readthedocs.io/) or [uv](https://docs.astral.sh/uv/):

```console
$ mamba create -n instanseg-env python=3.11
$ mamba activate instanseg-env
```
:::

## GPU support

InstanSeg runs on the CPU, on NVIDIA GPUs (CUDA) and on Apple silicon (MPS). The device is chosen
automatically, or can be set with the `device` argument of {class}`~instanseg.InstanSeg`.

To use an NVIDIA GPU, install a CUDA-enabled build of PyTorch by following the
[PyTorch installation guide](https://pytorch.org/get-started/locally/). Then check that PyTorch
can see the GPU:

```console
$ python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Install from source

To train your own models or contribute to InstanSeg, clone the repository and install it in
editable mode:

```console
$ git clone https://github.com/instanseg/instanseg.git
$ cd instanseg
$ pip install -e ".[full]"
```

To also install the test dependencies:

```console
$ pip install -e ".[full,test]"
```

## Check the installation

```console
$ python -c "from instanseg import InstanSeg; print('InstanSeg installed successfully!')"
```

Next, head to the {doc}`quickstart`.
