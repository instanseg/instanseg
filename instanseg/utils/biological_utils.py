from typing import Tuple, Any

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from instanseg.utils.pytorch_utils import torch_sparse_onehot, torch_fastremap, remap_values, torch_onehot

def get_intersection_over_union(label: torch.Tensor, return_lab: bool = True) -> torch.Tensor:

    from instanseg.utils.pytorch_utils import fast_sparse_dual_iou, torch_sparse_onehot
    label = torch.stack((torch_fastremap(label[0,0]),torch_fastremap(label[0,1])))[None]
    nucleus_onehot = torch_sparse_onehot(label[0, 0], flatten=True)[0]
    cell_onehot = torch_sparse_onehot(label[0, 1], flatten=True)[0]
    iou = fast_sparse_dual_iou(nucleus_onehot, cell_onehot)
    if return_lab:
        return iou, label
    return iou


def get_intersection_over_nucleus_area(label: torch.Tensor, return_lab: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the intersection over nucleus area in a 2 channel labeled image
     
    label must be a 1,2,H,W tensor where the first channel is nuclei and the second is whole cell
    """
    label = torch.stack((torch_fastremap(label[0,0]),torch_fastremap(label[0,1])))[None]
    nuclei_onehot = torch_sparse_onehot(label[0, 0], flatten=True)[0]
    cell_onehot = torch_sparse_onehot(label[0, 1], flatten=True)[0]
    intersection = torch.sparse.mm(nuclei_onehot, cell_onehot.T).to_dense()
    sparse_sum1 = torch.sparse.sum(nuclei_onehot, dim=(1,))[None].to_dense()
    nuclei_area = sparse_sum1.T

    if return_lab:
        return ((intersection / nuclei_area)), label

    return (intersection / nuclei_area), nuclei_area


def get_intersection_over_cell_area(label: torch.Tensor, return_lab = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the intersection over cell area in a 2 channel labeled image
     
    label must be a 1,2,H,W tensor where the first channel is nuclei and the second is whole cell
    """
    label = torch.stack((torch_fastremap(label[0,0]),torch_fastremap(label[0,1])))[None]
    nuclei_onehot = torch_sparse_onehot(label[0, 0], flatten=True)[0]
    cell_onehot = torch_sparse_onehot(label[0, 1], flatten=True)[0]
    intersection = torch.sparse.mm(cell_onehot, nuclei_onehot.T).to_dense()
    sparse_sum1 = torch.sparse.sum(cell_onehot, dim=(1,))[None].to_dense()
    cell_area = sparse_sum1.T
    if return_lab:
        return ((intersection / cell_area)), label

    return intersection / cell_area, cell_area

def nc_heatmap(label: torch.Tensor) -> torch.Tensor:
    """
    label is a 1,2,H,W tensor where the first channel is nuclei and the second is whole cell
    This function takes two labeled images and returns the nucleus/cell ratio heatmap
    """

    if label[0, 0].max() == 0 or label[0, 1].max() == 0:
        return torch.zeros_like(label[0, 0])


    cells = label[0, 1]

    
    iou, cell_area = get_intersection_over_cell_area(label)
    predicted_iou = iou.sum(1)#[0]
    onehot = torch_onehot(cells)
    onehot = onehot.float() * predicted_iou[:,None, None]
    map = onehot.max(1)[0]

    return map



def get_nonnucleated_cell_ids( lab: torch.Tensor,iou: torch.Tensor = None, threshold: float = 0.5, return_lab: bool = True) -> torch.Tensor:

    if iou is None:
        iou,lab = get_intersection_over_nucleus_area(lab, return_lab=True)
        lab = lab[0,1]
    
    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    nonnucleated = (iou.sum(0)) == 0
    if return_lab:
        return lab_ids[nonnucleated], lab * torch.isin(lab, lab_ids[nonnucleated]), lab * torch.isin(lab, lab_ids[
            ~nonnucleated])

    return lab_ids[nonnucleated]


def get_nucleated_cell_ids( lab: torch.Tensor,iou: torch.Tensor = None, threshold: float = 0.5, return_lab: bool = True) -> torch.Tensor:

    if iou is None:
        iou,lab = get_intersection_over_nucleus_area(lab, return_lab=True)
        lab = lab[0,1]
    

    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    nucleated = (iou.sum(0)) >= 1

    if return_lab:
        return lab_ids[nucleated], lab * torch.isin(lab, lab_ids[nucleated]), lab * torch.isin(lab, lab_ids[~nucleated])
    return lab_ids[nucleated]


def get_multinucleated_cell_ids( lab: torch.Tensor,iou: torch.Tensor = None, threshold: float = 0.5, return_lab: bool = True):
    #lab is 1,2,H,W
    if iou is None:
        iou,lab = get_intersection_over_nucleus_area(lab, return_lab=True)
        lab = lab[0,1]
    
    iou = iou > threshold
    lab_ids = torch.unique(lab[lab > 0])
    multinucleated = (iou.sum(0)) > 1

    if return_lab:
        return (lab_ids[multinucleated], 
                lab * torch.isin(lab, lab_ids[multinucleated]), 
                lab * torch.isin(lab, lab_ids[~multinucleated]))
    return lab_ids[multinucleated]

def keep_only_largest_nucleus_per_cell(labels: torch.Tensor, return_lab: bool = True)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    labels: tensor of shape 1,2,H,W containing nucleus and cell labels respectively
    return_lab: if True, returns the labels with only the largest nucleus per cell, and only cells that have a nucleus.
    """
    labels = torch_fastremap(labels)
    iou, nuclei_area = get_intersection_over_nucleus_area(labels)
    iou_biggest_area = ((iou > 0.5).float() * nuclei_area) == (((iou > 0.5).float() * nuclei_area).max(0)[0])
    iou_biggest_area = ((iou_biggest_area.float() * iou) > 0.5)
    nuclei_ids = torch.unique(labels[0, 0][labels[0, 0] > 0])
    cell_ids = torch.unique(labels[0, 1][labels[0, 1] > 0])
    largest_nucleus = (iou_biggest_area.sum(1)) == 1
    nucleated_cells = ((iou > 0.5).float().sum(0)) >= 1
    if return_lab:
        return nuclei_ids[largest_nucleus], torch.stack(
            (labels[0, 0] * torch.isin(labels[0, 0], nuclei_ids[largest_nucleus]),
             labels[0, 1] * torch.isin(labels[0, 1], cell_ids[nucleated_cells]))).unsqueeze(0)
    return (nuclei_ids[largest_nucleus],nuclei_ids[largest_nucleus]) #the duplication is to keep torchscript happy



def resolve_cell_and_nucleus_boundaries(lab: torch.Tensor, allow_unnucleated_cells: bool = True) -> torch.Tensor:
    """
    lab: tensor of shape 1,2,H,W containing nucleus and cell labels respectively

    returns: tensor of the same shape as lab

    This function will resolve the boundaries between cells and nuclei. 
    It will first match the labels of the largest nucleus and its cell.
    It will then erase from the cell masks all the nuclei pixels. This resolves nuclei "just" overlapping adjacent cell.
    It will then recover the nuclei pixels that were erased by adding them back to the cell masks.

    allow_unnucleated_cells: If False, this will remove all cells that don't have a nucleus.

    """

    if lab[0,0].max() == 0: # No nuclei
        if lab[0,1].max() == 0 or not allow_unnucleated_cells: # No cells
            return torch.zeros_like(lab)
        else:
            return lab
    elif lab[0,1].max() == 0:
        return torch.stack((lab[0,0],lab[0,0])).unsqueeze(0) #Nuclei but no cells, just duplicate the nuclei.

    lab = torch.stack((torch_fastremap(lab[0, 0]), torch_fastremap(lab[0, 1]))).unsqueeze(0)  # just relabel the nuclei and cells from 1 to N

    original_nuclei_labels = lab[0, 0].clone()
    original_cell_labels = lab[0, 1].clone()
    
    _, lab = keep_only_largest_nucleus_per_cell(lab, return_lab=True) # There will now be as many cells as there are nuclei. But the labels are not yet matched

    lab = torch.stack((torch_fastremap(lab[0, 0]), torch_fastremap(lab[0, 1]))).unsqueeze(0)
     

    if lab[0,0].max() == 0: # No nuclei
        if lab[0,1].max() == 0 or not allow_unnucleated_cells: # No cells
            return torch.zeros_like(lab)
        else:
            return lab
    elif lab[0,1].max() == 0:
        return torch.stack((lab[0,0],lab[0,0])).unsqueeze(0) #Nuclei but no cells, just duplicate the nuclei.
    
    clean_lab = lab 
    
    iou, _ = get_intersection_over_nucleus_area(clean_lab)
    onehot_remapping = (torch.nonzero(iou > 0.5).T + 1).flip(0)
    remapping = torch.cat((torch.zeros(2, 1, device=onehot_remapping.device), onehot_remapping), dim=1)
    clean_lab[0, 1] = remap_values(remapping, clean_lab[0, 1]).int()  # Every matching cell and nucleus now have the same label.


    nuclei_labels = clean_lab[0, 0]
    cell_labels = clean_lab[0, 1]

    original_nuclei_labels[nuclei_labels > 0] = 0
    original_nuclei_labels = torch_fastremap(original_nuclei_labels)
    original_nuclei_labels[original_nuclei_labels > 0]+= nuclei_labels.max()
    nuclei_labels += original_nuclei_labels

    cell_labels[nuclei_labels > 0] = 0
    cell_labels += nuclei_labels

    if allow_unnucleated_cells:

        cell_labels[cell_labels == 0] = (original_cell_labels[cell_labels == 0] + cell_labels.max()) * (original_cell_labels > 0)[cell_labels == 0].float() #this step can create small fragments. This is not a bug - but may have to be cleaned up in the future.
        
    return torch.stack((nuclei_labels, cell_labels)).unsqueeze(0)


def get_mean_object_features(image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # image is C,H,W
    # label is H,W
    # returns a tensor of size N,C for N objects and C channels

    if label.max() == 0:
        return torch.tensor([])
    label = label.squeeze()
    from instanseg.utils.pytorch_utils import torch_sparse_onehot
    sparse_onehot = torch_sparse_onehot(label, flatten=True)[0]
    out = torch.mm(sparse_onehot, image.flatten(1).T)  # object features
    sums = torch.sparse.sum(sparse_onehot, dim=1).to_dense()  # object areas
    out = out / sums[None].T  # mean object features
    return out


def get_features_by_location(input_tensor: torch.Tensor, lab: torch.Tensor, to_numpy: bool = True) -> tuple:
    # input tensor is C,H,W
    # lab is 1,2,H,W where the first channel is nuclei and the second is whole cell

    X_cell = get_mean_object_features(input_tensor, lab[0, 1])
    X_nuclei = get_mean_object_features(input_tensor, lab[0, 0])

    cytoplasm_lab = (lab[0, 0] == 0).float() * lab[0, 1]
    X_nuclei = get_mean_object_features(input_tensor, lab[0, 0])
    X_cytoplasm = get_mean_object_features(input_tensor, cytoplasm_lab)


    if to_numpy:
        X_cell = X_cell.cpu().numpy()
        X_nuclei = X_nuclei.cpu().numpy()
        X_cytoplasm = X_cytoplasm.cpu().numpy()

    return X_cell, X_nuclei, X_cytoplasm


from instanseg.utils.utils import _move_channel_axis
def get_nc_ratio(lab):
    # lab is 1,2,H,W where the first channel is nuclei and the second is whole cell
    
    lab = _move_channel_axis(torch.atleast_3d(lab.squeeze()))[None].long()
    nuclei_mask = lab[0,0]
    cell_mask = lab[0,1]

    nuclei_ids, nuclei_areas = torch.unique(nuclei_mask[nuclei_mask>0].flatten(), return_counts=True)
    cell_ids, cell_areas = torch.unique(cell_mask[cell_mask>0].flatten(), return_counts=True)

    max_labels = cell_mask.max()
    nc_ratio = torch.zeros(max_labels.int(), device=nuclei_mask.device)
    nc_ratio[nuclei_ids - 1] = nuclei_areas.float()
    nc_ratio[cell_ids - 1] /= cell_areas.float()
    nc_ratio = nc_ratio[cell_ids -1 ]
    
    if len(nc_ratio > 1):
        assert nc_ratio.max() <= 1, "Something whent terribly wrong.{}".format(nc_ratio.max())
    

    return nc_ratio




def violin_plot_feature_location(X_nuclei: np.ndarray, X_cytoplasm: np.ndarray, channel_names = None, labels = None, title = None, clamp = None):
    df = pd.DataFrame(columns=['Location', 'Channel', 'Value'])

    for i in range(X_nuclei.shape[1]):
        if channel_names is not None:
            channel_str = channel_names[i]
        else:
            channel_str = str(i)

        if labels is None:
            labels = ['Nuclei', 'Cytoplasm']

        # Assuming you have three arrays: array1, array2, and array3
        # Replace these with your actual data
        array1 = X_nuclei[:, i]
        array2 = X_cytoplasm[:, i]

        if clamp is not None:
            array1 = np.clip(array1, clamp[0], clamp[1])
            array2 = np.clip(array2, clamp[0], clamp[1])

        # Create a DataFrame with the arrays
        df = pd.concat([df, pd.DataFrame({
            'Value': np.concatenate([array1.flatten(), array2.flatten()]),
            ' ': np.repeat(labels, [len(array1), len(array2)]),
            'Channel': np.repeat([channel_str, channel_str], [len(array1), len(array2)])
        })], ignore_index=True)

        # Print the DataFrame

    plt.figure(figsize=(20, 10))
    sns.violinplot(data=df, x="Channel", y="Value", hue=" ", split=True, inner="quart", cut=0)

    if title is not None:
        plt.title(title)
    plt.show()


def show_umap_and_cluster(X_features):
    import scanpy as sc
    # Create a Scanpy AnnData object
    adata = sc.AnnData(X_features)

    # Z-normalise the data
    #sc.pp.scale(adata)

        # Normalizing to median total counts
    sc.pp.normalize_total(adata)
   # sc.pp.scale(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)

    # Perform PCA and UMAP
    # sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=5, n_pcs=50)
    sc.tl.umap(adata)

    # Perform leiden clustering
    sc.tl.leiden(adata, resolution=0.1)

    # Plot the UMAP result with leiden clusters
    sc.pl.umap(adata, color='leiden', palette='tab20', legend_loc='on data', title='UMAP with leiden Clustering')

    # Plot the UMAP result with leiden clusters
    plt.show()

    return adata


if __name__ == "__main__":
    from skimage import io
    from instanseg.utils.pytorch_utils import torch_sparse_onehot
    from instanseg.utils.utils import _choose_device

    instanseg = torch.jit.load("../torchscripts/1793450.pt")
    device = _choose_device()
    instanseg.to(device)
    input_data = io.imread("../examples/LuCa1.tif")
   # input_data = io.imread(Path(drag_and_drop_file()))
    from instanseg.utils.augmentations import Augmentations

    Augmenter = Augmentations()
  #  input_tensor, _ = Augmenter.to_tensor(input_data, normalize=True)
    input_tensor,_ =Augmenter.to_tensor(input_data,normalize=False) #this converts the input data to a tensor and does percentile normalization (no clipping)
    input_tensor,_ = Augmenter.normalize(input_tensor)

    print("Running InstanSeg ...")

    lab = instanseg(input_tensor[:,128:256,:128].unsqueeze(0).to(device))

    # lab = _sliding_window_inference(input_tensor, instanseg, window_size=(512, 512),
    #                                sw_device=device, device='cpu', output_channels=2)
    
    lab = resolve_cell_and_nucleus_boundaries(lab)

    from instanseg.utils.utils import export_to_torchscript
    from instanseg.utils.model_loader import load_model
    model_str = "1793450"

    device = "cpu"


    model, model_dict = load_model(folder=model_str)
    model.eval()
    model.to(device)

    cells_and_nuclei = model_dict['cells_and_nuclei']
    pixel_size = model_dict['pixel_size']
    n_sigma = model_dict['n_sigma']

    from instanseg.utils.loss.instanseg_loss import InstanSeg_Torchscript
    super_model = InstanSeg_Torchscript(model, cells_and_nuclei=cells_and_nuclei, 
                                        pixel_size = pixel_size, 
                                        n_sigma = n_sigma, 
                                        feature_engineering_function = str(model_dict["feature_engineering"]), 
                                        backbone_dim_in= 3, 
                                        to_centre = bool(model_dict["to_centre"]),
                                        mixed_precision = False).to(device)
    out = super_model(input_tensor[None,][:,:,128:256,:128])

    export_to_torchscript(model_name, show_example=True, mixed_predicision=False)

    print("Calculating cellular features ...")
    X_cell, X_nuclei, X_cytoplasm = get_features_by_location(input_tensor, lab)

    print("Plotting violing plots ...")

    violin_plot_feature_location(X_nuclei, X_cytoplasm)

    print("Clustering and umap ...")

    show_umap_and_cluster(X_cell)

    print("Done !")
