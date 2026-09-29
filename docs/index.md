---
sd_hide_title: true
---

# InstanSeg

::::{div} sd-text-center sd-fs-4 sd-font-weight-bold sd-mt-4 sd-mb-2

Fast, accurate and portable cell and nucleus segmentation for microscopy images

::::

::::{div} sd-text-center sd-mb-4

[![PyPI](https://img.shields.io/pypi/v/instanseg-torch?color=blue&label=PyPI)](https://pypi.org/project/instanseg-torch/)
[![Downloads](https://img.shields.io/pypi/dm/instanseg-torch?color=green&label=Downloads)](https://pypi.org/project/instanseg-torch/)
[![License](https://img.shields.io/github/license/instanseg/instanseg?color=orange)](https://github.com/instanseg/instanseg/blob/main/LICENSE)

::::

InstanSeg is a PyTorch-based instance segmentation pipeline for brightfield and fluorescence
microscopy. It segments nuclei and whole cells, and works on multiplexed images with any number
of channels.

```python
from instanseg import InstanSeg

instanseg = InstanSeg("brightfield_nuclei")
labels = instanseg.eval("HE_example.tif", save_output=True, save_overlay=True)
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.2em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install InstanSeg from PyPI, with optional extras for extra image formats, GPU support and training.
:::

:::{grid-item-card} {octicon}`rocket;1.2em;sd-mr-1` Quickstart
:link: quickstart
:link-type: doc

Segment your first image in a few lines of Python.
:::

:::{grid-item-card} {octicon}`image;1.2em;sd-mr-1` Running inference
:link: inference
:link-type: doc

Pretrained models, pixel sizes, large and whole slide images, saving results and the command line.
:::

:::{grid-item-card} {octicon}`code;1.2em;sd-mr-1` API reference
:link: api/index
:link-type: doc

Full reference for the `InstanSeg` class and utility functions.
:::

::::

## Why InstanSeg?

- **Free and open source**, released under the Apache 2.0 license.
- **Fast**: often much faster than other cell segmentation methods.
- **Nuclei and whole cells**: accurately segments both, in a single model.
- **Channel invariant**: runs on multiplexed images with new biomarker panels, without retraining
  or manual intervention.
- **Portable**: the whole model, including postprocessing, compiles to TorchScript. It runs with
  LibTorch alone, which is how InstanSeg runs directly inside [QuPath](qupath).

## How it works

```{figure} ../assets/instanseg_main_figure.png
:alt: Overview of the InstanSeg architecture
:width: 85%
:align: center

Overview of the InstanSeg pipeline. See the {doc}`papers <citing>` for details.
```

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

inference
training
qupath
```

```{toctree}
:hidden:
:caption: Reference

api/index
citing
```

```{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/instanseg/instanseg>
PyPI <https://pypi.org/project/instanseg-torch/>
```
