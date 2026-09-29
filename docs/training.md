# Training

Training InstanSeg needs a [source installation](installation.md#install-from-source) with the
`full` extra. The scripts are in the
[`instanseg/scripts`](https://github.com/instanseg/instanseg/tree/main/instanseg/scripts) folder
of the repository.

## Prepare datasets

The
[`load_datasets.ipynb`](https://github.com/instanseg/instanseg/blob/main/notebooks/load_datasets.ipynb)
notebook downloads public segmentation datasets and example images, and bundles them into a
dataset file for training. To train on your own data, extend the notebook with one of the
templates it provides.

## Train a model

Use `train.py` to train a model. For example, to train InstanSeg on the TNBC_2018 dataset for 250
epochs at 0.25 microns per pixel:

```console
$ cd instanseg/scripts
$ python train.py -data segmentation_dataset.pth -source "[TNBC_2018]" --num_epochs 250 --experiment_str my_first_instanseg --requested_pixel_size 0.25
```

To train a channel invariant model on the CPDMI_2023 dataset that predicts both nuclei and cells:

```console
$ python train.py -data segmentation_dataset.pth -source "[CPDMI_2023]" --num_epochs 250 --experiment_str my_first_instanseg -target NC --channel_invariant True --requested_pixel_size 0.5
```

With a CUDA or MPS GPU, each epoch takes about 1 to 3 minutes. For all options, run
`python train.py --help`.

## Evaluate a model

Use `test.py` to compute F1 scores. First optimise the postprocessing hyperparameters on the
validation set, then evaluate on the test set with the best parameters:

```console
$ python test.py --model_folder my_first_instanseg -test_set Validation --optimize_hyperparameters True
$ python test.py --model_folder my_first_instanseg -test_set Test --params best_params
```

## Use a trained model

Pass the name of the model folder to the `inference` command (see {doc}`inference`):

```console
$ inference --model_folder my_first_instanseg --image_path path/to/images
```

To export a trained model as TorchScript, for use with LibTorch or {doc}`QuPath <qupath>`, see the
[`export_model.ipynb`](https://github.com/instanseg/instanseg/blob/main/notebooks/export_model.ipynb)
notebook.
