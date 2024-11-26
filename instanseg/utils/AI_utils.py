import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from instanseg.utils.metrics import _robust_average_precision, _robust_f1_mean_calculator

from instanseg.utils.augmentations import Augmentations
import time

from instanseg.utils.display import show_images
import warnings


global_step = 0
def train_epoch(train_model: torch.nn.Module,
                train_device: torch.device,
                train_dataloader: torch.utils.data.DataLoader, 
                train_loss_fn: callable,
                train_optimizer: torch.optim.Optimizer, 
                args
                ) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    :param train_model: The model to train.
    :param train_device: The device to train on.
    :param train_dataloader: The dataloader for training data.
    :param train_loss_fn: The loss function.
    :param train_optimizer: The optimizer.
    :param args: Additional arguments.

    :return: The average training loss and the time taken for the epoch.
    """
    global global_step
    start = time.time()
    train_model.train()
    train_loss = []
    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)
        output = train_model(image_batch)
        loss = train_loss_fn(output, labels.clone()).mean()
        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start
    

global_step_test = 0
def test_epoch(test_model: torch.nn.Module,
               test_device:  torch.device, 
               test_dataloader: torch.utils.data.DataLoader,
               test_loss_fn: callable,
               args,
               postprocessing_fn: callable,
               method: str,
               iou_threshold: float,
               debug: bool=False, 
               save_str: Optional[str]=None, 
               save_bool: bool=False,
               best_f1: Optional[float]=None) -> Tuple[float, np.ndarray, float]:
    """
    Test the model for one epoch.

    :param test_model: The model to test.
    :param test_device: The device to test on.
    :param test_dataloader: The dataloader for  testing data.
    :param test_loss_fn: The loss function.
    :param args: (Namespace): Additional arguments.
    :param postprocessing_fn: The postprocessing function.
    :param method: The method used for testing.
    :param iou_threshold: The IoU threshold.
    :param debug: Debug mode. Defaults to False.
    :param save_str:  String for saving results. Defaults to None.
    :param save_bool: Whether to save results. Defaults to False.
    :param best_f1: The best F1 score. Defaults to None.

    :return: A tuple of the average test loss, the mean F1 scores, and the time taken for the epoch.
    """
    global global_step_test
    start = time.time()

    test_model.eval()
    test_loss = []

    current_f1_list = []
    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device) 
            output = test_model(image_batch)  
            loss = test_loss_fn(output, labels.clone()).mean()
            test_loss.append(loss.detach().cpu().numpy())



            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                predicted_labels = torch.stack([postprocessing_fn(out) for out in output])
                f1i = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                               threshold=iou_threshold)

                current_f1_list.append((f1i))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)  # either N,2 or N,

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    #  mean_f1 = current_f1_list

    if mean_f1 > best_f1 or save_bool:
        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]
        labels_dst = labels[0]
        lab = postprocessing_fn(output[0])

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss), mean1_f1, end - start


def collate_fn(data: List[Tuple]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Custom collate function for DataLoader.

    :param data: List of tuples with (example, label, length).

    :return: A tuple of torch.Tensor: Padded images, torch.Tensor: Stacked labels, torch.Tensor: Lengths of the images.
    """
    imgs, labels = zip(*data)
    lengths = [img.shape[0] for img in imgs]

    max_len = max(lengths)
    C, H, W = data[0][0].shape
    images = torch.zeros((len(data), max_len, H, W))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i, img in enumerate(imgs):
        images[i, :len(img)] = img

    return images, labels, lengths.int()


# import fastremap
class Segmentation_Dataset():
    """
    Custom dataset for segmentation tasks.

    :param img: List of images.
    :param label: List of labels.
    :param common_transforms: Whether to apply common transforms. Defaults to True.
    :param metadata: List of metadata. Defaults to None.
    :param size: Size of the images. Defaults to (256, 256).
    :param augmentation_dict: Dictionary of augmentations. Defaults to None.
    :param dim_in: Number of input dimensions. Defaults to 3.
    :param debug: Debug mode. Defaults to False.
    :param cells_and_nuclei: Whether to use cells and nuclei. Defaults to False.
    :param target_segmentation: Target segmentation type. Defaults to "N".
    :param channel_invariant: Whether to use channel invariant. Defaults to False.
    """
    def __init__(self,
                 img: List,
                 label: List,
                 common_transforms: bool=True,
                 metadata: Optional[List]=None,
                 size: Tuple(int, int)=(256, 256),
                 augmentation_dict: Optional[Dict]=None,
                 dim_in: int=3,
                 debug: bool=False,
                 cells_and_nuclei: bool=False,
                 target_segmentation: str="N",
                 channel_invariant: bool=False):
        self.X = img
        self.Y = label
        self.common_transforms = common_transforms

        assert len(self.X) == len(self.Y), "The number of images and labels must be the same"
        if len(metadata) == 0:
            self.metadata = [None] * len(self.X)
        else:
            self.metadata = metadata

        assert len(self.X) == len(self.metadata), print("The number of images and metadata must be the same")
        self.size = size
        self.Augmenter = Augmentations(augmentation_dict=augmentation_dict, debug=debug, shape=self.size,
                                       dim_in=dim_in, cells_and_nuclei=cells_and_nuclei,
                                       target_segmentation=target_segmentation, channel_invariant = channel_invariant)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i]
        label = self.Y[i]
        meta = self.metadata[i]

        if self.common_transforms:
            data, label = self.Augmenter(data, label, meta)

        if len(label.shape) == 2:
            label = label[None, :]
        if len(data.shape) == 2:
            data = data[None, :]

        assert not data.isnan().any(), "Tranformed images contains NaN"
        assert not label.isnan().any(), "Transformed labels contains NaN"

        return data.float(), label



def plot_loss(_model: torch.nn.Module):
    """
    Plot the loss of the model.

    :param _model: The model to plot the loss for.
    """
    loss_fig = plt.figure()
    timer = loss_fig.canvas.new_timer(interval=300000)
    timer.add_callback(plt.close)

    losses = [param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None]
    names = [name for name, param in _model.named_parameters() if param.grad is not None]

    plt.plot(losses)
    plt.xticks(np.arange(len(names))[::1], names[::1])
    plt.xticks(fontsize=8, rotation=90)
    spacing = 0.5
    loss_fig.subplots_adjust(bottom=spacing)
    timer.start()
    plt.show()


def check_max_grad(_model: torch.nn.Module) -> float:
    """
    Check the maximum gradient of the model.

    :param _model: The model to check the gradient for.

    :return: The maximum gradient.
    """
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.max()

def check_min_grad(_model: torch.nn.Module) -> float:
    """
    Check the minimum gradient of the model.

    :param _model: The model to check the gradient for.

    :return: The minimum gradient.
    """
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.min()

def check_mean_grad(_model: torch.nn.Module) -> float:):
    """
    Check the mean gradient of the model.

    :param _model: The model to check the gradient for.
    
    :return: The mean gradient.
    """
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.mean()

def optimize_hyperparameters(model: torch.nn.Module,
                             postprocessing_fn: callable,
                             data_loader: Optional[torch.utils.data.DataLoader] = None,
                             val_images: Optional[List] = None,
                             val_labels: Optional[List] = None,
                             max_evals: int = 50,
                             verbose: bool = False,
                             threshold: List[float] = [0.5, 0.7, 0.9],
                             show_progressbar: bool = True,
                             device: Optional[torch.device] = None) -> Dict:
    """
    Optimize hyperparameters using Bayesian optimization.

    
    :param model: The model to optimize.
    :param postprocessing_fn: The postprocessing function.
    :param data_loader: The dataloader for training data. Defaults to None.
    :param val_images: List of validation images. Defaults to None.
    :param val_labels: List of validation labels. Defaults to None.
    :param max_evals: Maximum number of evaluations. Defaults to 50.
    :param verbose: Verbose mode. Defaults to False.
    :param threshold: List of thresholds. Defaults to [0.5, 0.7, 0.9].
    :param show_progressbar: Whether to show the progress bar. Defaults to True.
    :param device: The device to use. Defaults to None.

    :return: The best hyperparameters.
    """

    from instanseg.utils.metrics import _robust_average_precision
    from instanseg.utils.utils import _choose_device

    from hyperopt import fmin
    from hyperopt import hp
    from hyperopt import Trials
    from hyperopt import tpe
    import copy

    if device is None:
        device = _choose_device()

    bayes_trials = Trials()

    space = {  # instanseg
        'mask_threshold': hp.uniform('mask_threshold', 0.3, 0.7),
        'seed_threshold': hp.uniform('seed_threshold', 0.5, 1),
        'overlap_threshold': hp.uniform('overlap_threshold', 0.1, 0.9),
        #'min_size': hp.uniform('min_size', 0, 30),
        'peak_distance': hp.uniform('peak_distance', 3, 10),
        'mean_threshold': hp.uniform('mean_threshold', 0.0, 0.3)} #the max could be increased, but may cuase the method not to converge for some reason.
    
    _model = model # copy.deepcopy(model)
    _model.eval()
    predictions = []

    with torch.no_grad():
        if data_loader is not None:
            for image_batch, labels_batch, _ in data_loader:
                    image_batch = image_batch.to(device)
                    output = _model(image_batch).cpu()
                    predictions.extend([pred,masks] for pred,masks in zip(output,labels_batch))


            def objective(params={}):
                pred_masks = []
                gt_masks = []
                for pred, masks in predictions:
                    lab = postprocessing_fn(pred.to(device), **params).cpu()
                    pred_masks.append(lab)
                    gt_masks.append(masks)

                mean_f1 = _robust_average_precision(torch.stack(gt_masks),torch.stack(pred_masks),threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        
        elif val_images is not None and val_labels is not None:
            from instanseg.utils.tiling import _instanseg_padding, _recover_padding
            def objective(params={}):
                pred_masks = []
                gt_masks = []
                #randomly shuffle val_images and val_labels

                np.random.seed(0)
                indexes = np.random.permutation(len(val_images))[:300]
                indexes.sort()

                for i in indexes:
                    imgs = val_images[i]
                    gt_mask = val_labels[i]
                    with torch.no_grad():
                        imgs = imgs.to(device)
                        imgs, pad = _instanseg_padding(imgs, min_dim = 32)
                        output = _model(imgs[None,])
                        output = _recover_padding(output, pad).squeeze(0)
                        lab = postprocessing_fn(output.to(device), **params).cpu()
                        pred_masks.append(lab)
                        gt_masks.append(gt_mask)

                mean_f1 = _robust_average_precision(gt_masks,pred_masks,threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        else:
            raise ValueError("Either data_loader or val_images and val_labels must be provided")

        print("Optimizing hyperparameters")
        # Optimize
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=bayes_trials, show_progressbar = show_progressbar)
    
    if verbose:
        print(best)
    return best



