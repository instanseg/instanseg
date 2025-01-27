import torch
import numpy as np
import pdb

from typing import Tuple, List, Union

from instanseg.utils.loss.lovasz_losses import binary_xloss
from instanseg.utils.pytorch_utils import torch_fastremap, torch_onehot, remap_values, fast_iou, fast_sparse_iou, eccentricity_batch, connected_components
from instanseg.utils.tiling import _instanseg_padding, _recover_padding

import torch.nn.functional as F
import torch.nn as nn

binary_xloss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

from instanseg.utils.utils import show_images

from instanseg.utils.utils import timer

integer_dtype = torch.int64



def convert(prob_input: torch.Tensor, coords_input: torch.Tensor, size: Tuple[int, int],
            mask_threshold: float = 0.5) -> torch.Tensor:
    # Create an array of labels for each pixel
    all_labels = torch.arange(1, 1 + prob_input.shape[0], dtype=torch.float32, device=prob_input.device)
    labels = torch.ones_like(prob_input) * torch.reshape(all_labels, (-1, 1, 1, 1))

    # Get flattened arrays
    labels = labels.flatten()
    prob = prob_input.flatten()
    x = coords_input[0, ...].flatten()
    y = coords_input[1, ...].flatten()

    # Predict image dimensions if we don't have them
    if size is None:
        size = (int(y.max() + 1), int(x.max() + 1))

    # Find indices with above-threshold probability values
    inds_prob = prob >= mask_threshold
    n_thresholded = torch.count_nonzero(inds_prob)
    if n_thresholded == 0:
        return torch.zeros(size, dtype=torch.float32, device=labels.device)

    # Create an array of [linear index, y, x, label], skipping low-probability values
    arr = torch.zeros((int(n_thresholded), 5), dtype=coords_input.dtype, device=labels.device)
    arr[:, 1] = y[inds_prob]
    arr[:, 2] = x[inds_prob]
    # NOTE: UNEXPECTED Y,X ORDER!
    arr[:, 0] = arr[:, 2] * size[1] + arr[:, 1]
    arr[:, 3] = labels[inds_prob]

    # Sort first by descending probability
    inds_sorted = prob[inds_prob].argsort(descending=True, stable=True)
    arr = arr[inds_sorted, :]

    # Stable sort by linear indices
    inds_sorted = arr[:, 0].argsort(descending=False, stable=True)
    arr = arr[inds_sorted, :]

    # Find the first occurrence of each linear index - this should correspond to the label
    # that has the highest probability, because they have previously been sorted
    inds_unique = torch.ones_like(arr[:, 0], dtype=torch.bool)
    inds_unique[1:] = arr[1:, 0] != arr[:-1, 0]

    # Create the output
    output = torch.zeros(size, dtype=torch.float32, device=labels.device)
    # NOTE: UNEXPECTED Y,X ORDER!
    output[arr[inds_unique, 2], arr[inds_unique, 1]] = arr[inds_unique, 3].float()

    return output


def find_all_local_maxima(image: torch.Tensor, neighbourhood_size: int, minimum_value: float) -> torch.Tensor:
    """
        helper function for peak_local_max that finds all the local maxima
        within each neighbourhood. (may return multiple per neighbourhood).
        """
    # Perform max pooling with the specified neighborhood size
    kernel_size = 2 * neighbourhood_size + 1
    pooled = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Create a mask for the local maxima
    mask = (pooled == image) * (image >= minimum_value)

    # Apply the mask to the original image to retain only the local maxima values
    local_maxima = image * mask

    return local_maxima


def torch_peak_local_max(image: torch.Tensor, neighbourhood_size: int, minimum_value: float,
                             return_map: bool = False, dtype: torch.dtype = torch.int) -> torch.Tensor:
    """
    UPDATED FOR PERFORMANCE TESTING - NOT IDENTICAL, AS USES *FIRST* MAX, NOT FURTHEST FROM ORIGIN
    """
    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    kernel_size = 2 * neighbourhood_size + 1
    pooled, max_inds = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=neighbourhood_size,
                                    return_indices=True)

    inds = torch.arange(0, image.numel(), device=device, dtype=dtype).reshape(image.shape)

    peak_local_max = (max_inds == inds) * (pooled > minimum_value)

    if return_map:
        return peak_local_max

    # Non-zero causes host-device synchronization, which is a bottleneck
    return torch.nonzero(peak_local_max.squeeze()).to(dtype)

def torch_peak_local_max_LEGACY(image: torch.Tensor, neighbourhood_size: int, minimum_value: float, return_map: bool = False) -> torch.Tensor:
    """
    computes peak local maxima function for an image (or batch of images), returning a maxima mask
    and the coordinates of the peak local max values.
    peak local maxima returns a image that is zero at all points other than local maxima.
    At the local maxima, the pixel retains its value in the original image.
    
    image: a torch tensor of shape [B,1,H,W] or [H,W], B is batch size. H,W are spatial dims.
    neighbourhood_size: int. Only one maximum will be selected within a square patch of width
        equal to the neighbourhood size (specifically the largest maxima in that neighbourhood).
        Where there are multiple local maxima with the largest value within the neighbourhood,
        the maxima furthest away from the origin (furthest by euclidian distance from pixel (0,0))
        is retained (ensuring there is only one maximum per neighbourhood).
    minimum_value: float. Local maxima with pixel intensity below this value are ignored.
    
    returns: a torch tensor of shape equal to image, a list of length B containing (lx, ly) pairs
    where lx and ly are torch tensors containing the x and y coordinates of each local maxima for a given image.
    if image has shape [H,W], returns (lx, ly). 
    """
    assert image.ndim == 2, "image must be of shape [H,W]"

    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    all_local_maxima = find_all_local_maxima(image, neighbourhood_size, minimum_value)

    # perform non-maximal coordinate suppression to only get one maximum per neighbourhood.
    # specifically, where there are two maxima in a neighbourhood, I retain the maxima
    # which has the furthest euclidian distance away from the origin. This is just an
    # 'arbitrary' way for me to split the ties. 
    spatial_dims = [image.shape[-2], image.shape[-1]]

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, spatial_dims[0], 1, device=device, dtype = torch.float32), torch.arange(0, spatial_dims[1], 1, device=device, dtype = torch.float32),
            indexing='ij'
        )
    )

    distance_to_origin = (grid.unsqueeze(0)).square().sum(dim=1).sqrt()

    distance_of_max_poses = torch.mul(all_local_maxima, distance_to_origin)

    retained_maxima = find_all_local_maxima(distance_of_max_poses, neighbourhood_size, minimum_value=minimum_value)
    peak_local_max = all_local_maxima * (retained_maxima > minimum_value)

    if return_map:
        return peak_local_max

    locs = grid[:,peak_local_max.squeeze()>0].T.int()


    return locs


def centre_crop(centroids: torch.Tensor, window_size: int, h:int, w:int) -> torch.Tensor:
    """
    centres the crop around the centroid, ensuring that the crop does not exceed the image dimensions.
    """
    C = centroids.shape[0]
    centroids = centroids.clone()  # C,2
    centroids[:,0] = centroids[:, 0].clamp(min=window_size //2 , max=h - window_size //2)
    centroids[:, 1] = centroids[:,1].clamp(min=window_size //2, max=w - window_size //2)
    window_slices = (centroids[:, None] + torch.tensor([[-1, -1], [1, 1]], device = centroids.device) * (window_size //2))


    grid_x, grid_y = torch.meshgrid(
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype), indexing="ij")
    
    mesh = torch.stack((grid_x, grid_y))

    mesh_grid = mesh.expand(C, 2, window_size, window_size)  # C,2,2*window_size,2*window_size
    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,C,2*window_size*2*window_size
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_grid_flat = mesh_grid_flat + idx
    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)  # 2,C*2*window_size*2*window_size

    return mesh_grid_flat


def compute_crops( x: torch.Tensor, 
                  c: torch.Tensor, 
                  sigma: torch.Tensor,
                  centroids_idx: torch.Tensor,
                  feature_engineering, 
                  pixel_classifier, 
                  window_size: int = 128):

    h, w = x.shape[-2:]
    C = c.shape[0]

    mesh_grid_flat = centre_crop(centroids_idx, window_size, h, w)

    x = feature_engineering(x, c, sigma, window_size //2 , mesh_grid_flat)


    x = pixel_classifier(x)  # C*H*W,1

    x = x.view(C, 1, window_size, window_size)

    idx = torch.arange(1, C + 1, device=x.device, dtype = mesh_grid_flat.dtype)

    rep = torch.ones((C, window_size, window_size), device=x.device, dtype =mesh_grid_flat.dtype)
    rep = rep * (idx[:, None, None] - 1)

    iidd = torch.cat((rep.flatten()[None,], mesh_grid_flat)).to(mesh_grid_flat.dtype)

    return x, iidd



def find_connected_components(adjacency_matrix: torch.Tensor):

    M = (adjacency_matrix + torch.eye(adjacency_matrix.shape[0],
                               device=adjacency_matrix.device))  # https://math.stackexchange.com/questions/1106870/can-i-find-the-connected-components-of-a-graph-using-matrix-operations-on-the-gr
    num_iterations = 10
    out = torch.matrix_power(M, num_iterations)
    col = torch.arange(0, out.shape[0], device=out.device).view(-1, 1).expand(out.shape[0], out.shape[
        0])  # Just a column matrix with numbers from 0 to out.shape[0]
    out_col_idx = ((out > 1).int() - torch.eye(out.shape[0], device=out.device)) * col
    maxes = out_col_idx.argmax(0) * (out_col_idx.max(0)[0] > 0).int()
    maxes = torch.maximum(maxes + 1, (torch.arange(0, out.shape[0],
                                                    device=out.device) + 1))  # recover the diagonal elements that were suppressed
    tentative_remapping = torch.stack(((torch.arange(0, out.shape[0], device=out.device) + 1), maxes))
    # start with two zeros:
    remapping = torch.cat((torch.zeros(2, 1, device=tentative_remapping.device), tentative_remapping),
                            dim=1)  # Maybe this can be avoided in the future by thresholding labels
    
    return remapping
    


def has_pixel_classifier_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'ProbabilityNet' in module_class:
                return True
    return False




def merge_sparse_predictions(x: torch.Tensor, coords: torch.Tensor, mask_map: torch.Tensor, size : list[int], mask_threshold: float = 0.5, window_size = 128, min_size = 10, overlap_threshold = 0.5, mean_threshold = 0.5):

    # top_left = window_slices[:,0,:]
    # shifted_centroid = centroids_idx - top_left
    # cc = connected_components((output>mask_threshold).float())
    # labels_to_keep = cc[torch.arange(cc.shape[0]),0,shifted_centroid[:,0],shifted_centroid[:,1]]
    # in_mask = cc == labels_to_keep[:,None,None,None]
    # output *= in_mask


    labels = convert(x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold)[None]

    idx = torch.arange(1, size[0] + 1, device=x.device, dtype =coords.dtype)
    stack_ID = torch.ones((size[0], window_size, window_size), device=x.device, dtype=coords.dtype)
    stack_ID = stack_ID * (idx[:, None, None] - 1)

    coords = torch.stack((stack_ID.flatten(), coords[0] * size[2] + coords[1])).to(coords.dtype)

    fg = x.flatten() > mask_threshold
    x = x.flatten()[fg]
    coords = coords[:, fg]

    using_mps = False
    if x.is_mps:
        using_mps = True
        device = 'cpu'
        x = x.to(device)
        mask_map = mask_map.to(device)

    sparse_onehot = torch.sparse_coo_tensor(
        coords,
        x.flatten() > mask_threshold,
        size=(size[0], size[1] * size[2]),
        dtype=x.dtype,
        device=x.device,
        requires_grad=False,
        )

    object_areas = torch.sparse.sum(sparse_onehot, dim=1).values()

    sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map.flatten()[None]), dim=1).values()
    mean_mask_value = sum_mask_value / object_areas
    objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

    if window_size **2 * sparse_onehot.shape[0] == sparse_onehot.sum():
        #This can happen at the start of training. This can cause OOM errors and is never a good sign - may aswell abort.
        return labels

    iou = fast_sparse_iou(sparse_onehot)

    remapping = find_connected_components((iou>overlap_threshold).float() )

    if using_mps:
        device = 'mps'
        remapping = remapping.to(device)
        labels = labels.to(device)

    labels = remap_values(remapping, labels)

    labels_to_remove = (torch.arange(0, len(objects_to_remove), device=objects_to_remove.device, dtype = coords.dtype) + 1)[
    objects_to_remove]

    labels[torch.isin(labels, labels_to_remove)] = 0

    return labels

def guide_function(params: torch.Tensor,device ='cuda', width: int = 256):

    #params must be depth,3  

    depth = params.shape[0]
    xx = torch.linspace(0, 1, width, device=device).view(1, 1, -1).expand(1, width,width)
    yy = torch.linspace(0, 1, width, device=device).view(1, -1, 1).expand(1, width, width)
    xxyy  = torch.cat((xx, yy), 0).expand(depth,2,width,width)

    xx = xxyy[:,0] * params[:,0][:,None,None]
    yy = xxyy[:,1] * params[:,1][:,None,None]

    return torch.sin(xx+yy+params[:,2,None,None])[None]


def generate_coordinate_map(mode: str = "linear", spatial_dim: int = 2, height: int = 256, width: int = 256, device: torch.device = torch.device(type='cuda')):

    if mode == "linear":
        if spatial_dim ==2:
            xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height,width)
            yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
            xxyy = torch.cat((xx, yy), 0)

        elif spatial_dim >= 3:
            xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height,width)
            yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
            zz = torch.zeros_like(xx).expand(spatial_dim - 2,-1,-1)
            xxyy = torch.cat((xx, yy,zz), 0)
        else:
            xxyy = torch.zeros((spatial_dim, height, width), device=device) #NOT IMPLEMENTED - THIS IS JUST A DUMMY VALUE

    else:
        xxyy = torch.zeros((spatial_dim, height, width), device=device) #NOT IMPLEMENTED - THIS IS JUST A DUMMY VALUE


    return xxyy



class ProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width = 5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        # x is C*H*W,E+S+1 (H,W is the window of the crop used here, e.g 100x100, not the image)
      #  with torch.cuda.amp.autocast():
        x = self._relu_non_empty(self.fc1(x))
        x = self._relu_non_empty(self.fc2(x))
        x = self.fc3(x)
        return x

    def _relu_non_empty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Workaround for https://github.com/pytorch/pytorch/issues/118845 on MPS
        """
        if x.numel() == 0:
            return x
        else:
            return torch.relu_(x)



class MyBlock(nn.Sequential):
    def __init__(self, embedding_dim, width):
        super(MyBlock, self).__init__()
        self.fc1 = nn.Conv2d(embedding_dim, width, 1, padding = 0//2)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace = True)
        self.fc2 = nn.Conv2d(width, width, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace = True)
        self.fc3 = nn.Conv2d(width, 1 , 1)


class ConvProbabilityNet(nn.Module):
    def __init__(self, embedding_dim=4, width = 5, depth = 5):
        super().__init__()
        self.layer1 = MyBlock(embedding_dim + depth,width)
        self.layer2 = MyBlock(embedding_dim ,width)
        self.layer3 = MyBlock(embedding_dim + 2 ,width)
        
        self.positional_embedding_params = (nn.Parameter(torch.rand(depth,3)*10) ).to("cuda")

    

    def forward(self, x):
        # x is C*H*W,E+S+1 (H,W is the window of the crop used here, e.g 100x100, not the image)

        positional_embedding = guide_function(self.positional_embedding_params, width = 100)

        one = self.layer1(torch.cat((x,positional_embedding.expand(x.shape[0],-1,-1,-1)),dim=1))
        two = self.layer2(x)

        output = self.layer3(torch.cat((x,one,two),dim=1))

        return output
    

class MedianFilter(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int]):
        from kornia.filters import MedianBlur
        super(MedianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.MedianBlur = MedianBlur(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MedianBlur(x)


from einops import rearrange


def feature_engineering(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x = torch.cat([x,sigma])[:,mesh_grid_flat[0],mesh_grid_flat[1]]
    
    x = rearrange(x, '(E) (C H W) -> C (E) H W', E = E + S, C = C, H = 2 * window_size, W = 2 * window_size)
    c_shaped = c.view(-1, E, 1, 1)
    x[:,:E] -= c_shaped
    x = rearrange(x, 'C (E) H W-> (C H W) (E)', E = E + S)
    return x




def feature_engineering_slow(x: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped

    x = torch.cat([diff, sigma_slices], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x



def feature_engineering_2(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # EXTRA DIFF
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    norm = torch.sqrt(torch.sum(torch.pow(x_slices - c_shaped, 2) + 1e-6, dim=1, keepdim=True))  # C,1,H,W

    diff = x_slices - c_shaped


    x = torch.cat([diff, sigma_slices, norm], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x

def feature_engineering_3(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # NO SIGMA
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)

    diff = x_slices - c_shaped


    x = torch.cat([diff, sigma_slices * 0], dim=1)  # C,E+S+1,H,W

    x = x.flatten(2).permute(0, -1, 1)  # C,H*W,E+S+1
    x = x.reshape((x.shape[0] * x.shape[1]), x.shape[2])  # C*H*W,E+S+1

    return x




def feature_engineering_10(x: torch.Tensor, xxyy: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor, window_size: int,
                        mesh_grid_flat: torch.Tensor):
    
    # CONV
    E = x.shape[0]
    h, w = x.shape[-2:]
    C = c.shape[0]
    S = sigma.shape[0]

    x_slices = x[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(E, C, 2 * window_size, 2 * window_size).permute(1, 0, 2,
                                                                                                                  3)  # C,E,2*window_size,2*window_size
    sigma_slices = sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]].reshape(S, C, 2 * window_size, 2 * window_size).permute(1,
                                                                                                                          0,
                                                                                                                          2,
                                                                                                                          3)  # C,S,2*window_size,2*window_size
    c_shaped = c.reshape(-1, E, 1, 1)
    diff = x_slices - c_shaped
    x = torch.cat([diff, sigma_slices], dim=1)  # C,E+S+1,H,W

    return x




def feature_engineering_generator(feature_engineering_function):

    if feature_engineering_function == "0" or feature_engineering_function == "7":
        return feature_engineering, 2
    elif feature_engineering_function == "2":
        return feature_engineering_2, 3
    elif feature_engineering_function == "3":
        return feature_engineering_3, 2
    elif feature_engineering_function == "10":
        return feature_engineering_10, 2

    else:
        raise NotImplementedError("Feature engineering function",feature_engineering_function,"is not implemented")





class InstanSeg(nn.Module):

    def __init__(self,
                 n_sigma: int = 1, 
                 instance_weight: float = 1.5, 
                 device: str = 'cuda', 
                 binary_loss_fn_str: str = "lovasz_hinge", 
                 seed_loss_fn = "binary_xloss", 
                 cells_and_nuclei: bool = False, 
                 to_centre: bool = True, 
                 multi_centre: bool = False,
                 window_size = 256, 
                 tile_size = 256,
                 feature_engineering_function = "0",
                 dim_coords = 2):
        
        super().__init__()
        self.n_sigma = n_sigma
        self.instance_weight = instance_weight
        self.device = device
        self.dim_coords = dim_coords

        self.dim_out = self.dim_coords + self.n_sigma + 1
        self.parameters_have_been_updated = False

        if cells_and_nuclei:
            self.dim_out = self.dim_out * 2
        self.cells_and_nuclei = cells_and_nuclei

        self.to_centre = to_centre
        self.multi_centre = multi_centre
        self.window_size = window_size

        self.num_instance_cap = 50
        self.sort_by_eccentricity = False

        xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = tile_size, width = tile_size, device = device)

        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        self.feature_engineering_function = feature_engineering_function


        self.register_buffer("xxyy", xxyy)

        self.update_binary_loss(binary_loss_fn_str)

        self.update_seed_loss(seed_loss_fn)

    def update_binary_loss(self,binary_loss_fn_str):

        if binary_loss_fn_str == "lovasz_hinge":
            from instanseg.utils.loss.lovasz_losses import lovasz_hinge
            def binary_loss_fn(pred, gt, **kwargs):
               # pred = torch.sigmoid_(pred)
                return lovasz_hinge((pred.squeeze(1)), gt,per_image = True)

        elif binary_loss_fn_str == "binary_xloss":
            from instanseg.utils.loss.lovasz_losses import binary_xloss
            self.binary_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif binary_loss_fn_str == "dicefocal_loss":
            from monai.losses import DiceFocalLoss
            
            binary_loss_fn_ = DiceFocalLoss(sigmoid=True)
            def binary_loss_fn(pred, gt, **kwargs):
                l = binary_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l
        elif binary_loss_fn_str == "dice_loss":
            from monai.losses import DiceLoss
            
            binary_loss_fn_ = DiceLoss(sigmoid=True)
            def binary_loss_fn(pred, gt, **kwargs):
                l = binary_loss_fn_(pred[None,:,0], gt.unsqueeze(0)) * 1.5
                return l


        elif binary_loss_fn_str == "general_dice_loss":
            from monai.losses import GeneralizedDiceLoss
            def binary_loss_fn(pred, gt):
                return GeneralizedDiceLoss(sigmoid=True)(pred, gt.unsqueeze(1))
            

        elif binary_loss_fn_str == "cross_entropy":
            from torch.nn import NLLLoss
            assert self.window_size == 256, "Cross entropy loss only works with window size 256"
            assert self.num_instance_cap is None, "Cross entropy loss only works with num_instance_cap = None"
            
            self.l_fn = NLLLoss()
            self.m = nn.LogSoftmax(dim=1)
            
            def binary_loss_fn(pred, gt, sigma):
                pred = torch.cat([sigma[None,None],pred])
                

                gt = torch.cat(((gt.sum(0)==0)[None],gt))
                target = gt.argmax(0)[None]

                pred = pred.squeeze(1).unsqueeze(0)
              
                pred = self.m(pred)

                return self.l_fn(pred,target.long()) * 7
            
        else:
            raise NotImplementedError("Binary loss function",binary_loss_fn,"is not implemented")
        self.binary_loss_fn = binary_loss_fn

    def update_seed_loss(self,seed_loss_fn):
        if seed_loss_fn in ["binary_xloss"]:
            binary_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

            def seed_loss(x,y, mask = None):
                if mask is not None:
                    mask = mask.float()  # Ensure the mask is float for multiplication
                    loss = binary_loss(x, (y > 0).float())  # Calculate the element-wise binary loss
                    masked_loss = loss * mask  # Apply the mask to the loss
                    return masked_loss.sum() / mask.sum()
                else:
                    return binary_loss(x, (y > 0).float()).mean()
                
            self.seed_loss = seed_loss

        elif seed_loss_fn in ["l1_distance"]:
            from instanseg.utils.pytorch_utils import instance_wise_edt
            distance_loss = torch.nn.L1Loss(reduction='none')
            def seed_loss(x,y, mask = None):
                edt = (instance_wise_edt(y.float(), edt_type= 'edt') - 0.5 ) * 15 #This is to mimick the range of CELoss
                loss = distance_loss((x), (edt[None]))

                if mask is not None:
                    mask = mask.float()
                    masked_loss = loss * mask
                    return masked_loss.sum() / mask.sum()
                else:
                    return loss.mean()
                
            self.seed_loss = seed_loss
        else:
            raise NotImplementedError("Seedloss function",seed_loss_fn,"is not implemented")

    def initialize_pixel_classifier(self, model, MLP_width = 10, MLP_input_dim = None):

        if has_pixel_classifier_model(model):
            try:
                self.pixel_classifier = model.pixel_classifier
            except:
                self.pixel_classifier = model.model.pixel_classifier  # This happens when there is an adaptornet
            return model
        else:
            if MLP_input_dim is None:
                MLP_input_dim = self.feature_engineering_width + self.n_sigma -2 + self.dim_coords
            model.pixel_classifier = ProbabilityNet( MLP_input_dim, width = MLP_width)
            if self.feature_engineering_function != "10":
                model.pixel_classifier = ProbabilityNet( MLP_input_dim, width = MLP_width)
            else:
                model.pixel_classifier = ConvProbabilityNet( MLP_input_dim, width = MLP_width)
            self.pixel_classifier = model.pixel_classifier.to(self.device)
            return model

    def forward(self, prediction: torch.Tensor, instances: torch.Tensor, w_inst: float = 1.5, w_seed: float = 1.0):

        w_inst = self.instance_weight

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xxyy = self.xxyy[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0
  
        if self.cells_and_nuclei:
            dim_out = int(self.dim_out / 2)
        else:
            dim_out = self.dim_out

        for mask_channel in range(0, instances.shape[1]):

            if mask_channel == 0:
                prediction_b = prediction[:, 0: dim_out, :, :]
            else:
                prediction_b = prediction[:, dim_out:, :, :]

            instances_batch = instances

            if not self.to_centre:

                spatial_emb_batch = (torch.sigmoid((prediction_b[:, 0: self.dim_coords]))-0.5) * 8 + xxyy
            else:
                spatial_emb_batch = (prediction_b[:, 0: self.dim_coords]) + xxyy
            sigma_batch = prediction_b[:, self.dim_coords: self.dim_coords + self.n_sigma]  # n_sigma x h x w
            seed_map_batch = prediction_b[:, self.dim_coords + self.n_sigma: self.dim_coords + self.n_sigma + 1]  # 1 x h x w

            for b in range(0, batch_size):

                spatial_emb = spatial_emb_batch[b]
                sigma = sigma_batch[b]
                seed_map = seed_map_batch[b]

                instance_loss = 0
                seed_loss = 0

                instance = instances_batch[b, mask_channel].unsqueeze(0)  # 1 x h x w

                if (instance < 0).all(): #-1 means not annotated
                    continue
       

                elif instance.min() < 0: #label is sparse
                    mask = instance >=0
                    instance[instance < 0] = 0
                else:
                    mask = None

                seed_loss_tmp = self.seed_loss(seed_map,instance, mask = mask)

                seed_loss += seed_loss_tmp

                if w_inst == 0:
                    loss += w_seed * seed_loss
                    continue

                instance_ids = instance.unique()
                instance_ids = instance_ids[instance_ids != 0]

                if len(instance_ids) > 0:

                    instance = torch_fastremap(instance)

                    onehot_labels = torch_onehot(instance).squeeze(0)  # C x h x w

                    if self.num_instance_cap is not None: #This is to cap the number of objects to avoid OOM errors.
                         if self.num_instance_cap < onehot_labels.shape[0]:
                            if self.sort_by_eccentricity:
                                eccentricities = eccentricity_batch(onehot_labels.float())
                                idx = eccentricities.argsort(descending = True)[:self.num_instance_cap]
                            else:
                                idx = torch.randperm(onehot_labels.shape[0])[:self.num_instance_cap]
                            onehot_labels = onehot_labels[idx]


                    if self.multi_centre:
                        seed_map_tmp = torch.sigmoid(seed_map)
                        
                        centroids = torch_peak_local_max(seed_map_tmp.squeeze() * onehot_labels.sum(0), neighbourhood_size = 3, minimum_value = 0.5).T

                        if self.to_centre:
                            centres = xxyy[:,centroids[0],centroids[1]].detach().T
                        else:
                            centres = spatial_emb[:,centroids[0],centroids[1]].detach().T

                        idx = torch.randperm(centroids.shape[1])[:self.num_instance_cap]
                  
                        centres = centres[idx]
                        centroids = centroids[:,idx]

                        instance_labels = onehot_labels[:,centroids[0],centroids[1]].float().argmax(0)
                        onehot_labels = onehot_labels[instance_labels]

                        centroids = centroids.T
             
                    else:
                        if self.to_centre:

                            seed_map_min = seed_map.min()
           
                            seed_map_tmp = (seed_map - seed_map.min()).detach()
                            centres = xxyy.flatten(1).T[((seed_map_tmp * onehot_labels).flatten(1)).argmax(1)]  # location at max seed (used in postprocessing)
                            seed_map = seed_map_tmp + seed_map_min
                        else:
                            seed_map_tmp = seed_map - seed_map.min()
                            centres = spatial_emb.flatten(1).T[((seed_map_tmp * onehot_labels).flatten(1)).argmax(1)].detach()  # embedding at max seed (used in postprocessing)
                            
                        centroids = (torch.sum(((xxyy[:2] * onehot_labels.unsqueeze(1))).flatten(2),dim=2)/onehot_labels.flatten(1).sum(1)[:,None] )* (256 / 64) #coordinates of centre of mass
                        centroids = torch.stack((centroids[:, 1], centroids[:, 0])).T

                    if len(centroids) == 0:
                        loss += w_seed * seed_loss
                        continue
      
                    dist, coords = compute_crops(spatial_emb, 
                                                 centres, 
                                                 sigma, 
                                                 centroids, 
                                                 feature_engineering = self.feature_engineering,
                                                 pixel_classifier=self.pixel_classifier,
                                                 window_size = self.window_size)
                    

                    crop = onehot_labels.squeeze(1)[coords[0], coords[1], coords[2]].reshape(-1, self.window_size, self.window_size)

                    instance_loss = instance_loss + self.binary_loss_fn(dist,crop.float(), sigma = sigma[0])

                loss += w_inst * instance_loss + w_seed * seed_loss


        loss = loss / (b + 1)

        if self.cells_and_nuclei:
            loss = loss / 2

        return loss
    
    

    def update_hyperparameters(self,params):
        self.parameters_have_been_updated = True
        self.params = params



    #@timer
    def postprocessing(self, prediction: Union[torch.Tensor, np.ndarray],
                        mask_threshold: float = 0.53,
                        peak_distance: int = 5,
                        seed_threshold: float = 0.8,
                        overlap_threshold: float = 0.3,
                        mean_threshold: float = 0.1,
                        window_size: int = 128,
                        min_size = 10,
                       device=None,
                       classifier=None,
                       cleanup_fragments: bool = False,
                       max_seeds: int = 2000,
                       return_intermediate_objects: bool = False,
                       precomputed_crops: torch.Tensor = None,
                       precomputed_seeds: torch.Tensor = None,
                       img=None):

        if device is None:
            device = self.device
        if classifier is None:
            classifier = self.pixel_classifier

        if self.parameters_have_been_updated:
            mask_threshold = self.params['mask_threshold']
            peak_distance = self.params['peak_distance']
            seed_threshold = self.params['seed_threshold']
            overlap_threshold = self.params['overlap_threshold']
            if "min_size" in self.params:
                min_size = self.params['min_size']
            if "mean_threshold" in self.params:
                mean_threshold = self.params['mean_threshold']
            

        if isinstance(prediction, np.ndarray):
            prediction = torch.tensor(prediction, device=device)

        if self.cells_and_nuclei:
            iterations = 2
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        labels = []

        for i in range(iterations):

            if precomputed_crops is None:

                if i == 0:
                    prediction_i = prediction[0: dim_out, :, :]
                else:
                    prediction_i = prediction[dim_out:, :, :]

                height, width = prediction_i.size(1), prediction_i.size(2)

                ##torch.cuda.synchronize()

                xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = height, width = width, device = device)

                #torch.cuda.synchronize()

                if not self.to_centre:
                    fields = (torch.sigmoid(prediction_i[0:self.dim_coords])-0.5) * 8
                else:
                    fields = prediction_i[0:self.dim_coords]

                sigma = prediction_i[self.dim_coords:self.dim_coords + self.n_sigma]
            #    mask_map = torch.sigmoid(prediction_i[self.dim_coords + self.n_sigma])

                mask_map = ((prediction_i[self.dim_coords + self.n_sigma]) / 15) + 0.5

                if (mask_map > mask_threshold).max() == 0:  # no foreground pixels
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

            # local_centroids_idx = #torch.tensor([[20,21,32,32,34],[30,35,36,364,346]],device = device).long().T

                #torch.cuda.synchronize()

                if precomputed_seeds is None:
                    local_centroids_idx = torch_peak_local_max(mask_map, neighbourhood_size=int(peak_distance), minimum_value=seed_threshold)
                else:
                    local_centroids_idx = precomputed_seeds


                #torch.cuda.synchronize()

                fields = fields + xxyy
                if self.to_centre:
                    fields_at_centroids = xxyy[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]
                else:
                    fields_at_centroids = fields[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]

                if local_centroids_idx.shape[0] > max_seeds:
                    print("Too many seeds, skipping", local_centroids_idx.shape[0])
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                
                C = fields_at_centroids.shape[0]

                h, w = mask_map.shape[-2:]
                window_size = min(window_size, h, w)
                window_size = window_size - window_size % 2

                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                #torch.cuda.synchronize()
                crops, coords = compute_crops(fields, 
                                                fields_at_centroids.T, 
                                                sigma, 
                                                local_centroids_idx.int(), 
                                                feature_engineering = self.feature_engineering,
                                                pixel_classifier=self.pixel_classifier,
                                                window_size=window_size) # about 65% of the time
                #torch.cuda.synchronize()
                coords = coords[1:] # The first channel are just channel indices, not required here.

                if return_intermediate_objects:
                    return crops, coords, mask_map

                C = crops.shape[0]
                if C == 0:
                    label = torch.zeros(mask_map.shape, dtype=int, device=mask_map.device).squeeze()
                    labels.append(label)
                    continue

                

            else:
                crops,coords,mask_map = precomputed_crops
                C = crops.shape[0]



            h, w = mask_map.shape[-2:]

            label = merge_sparse_predictions(crops, coords, mask_map, size=(C,h, w), mask_threshold=mask_threshold, window_size=window_size, min_size=min_size, overlap_threshold=overlap_threshold, mean_threshold=mean_threshold).int() #about 30% of the time


            # from utils.pytorch_utils import centroids_from_lab
            # centroids, ids = centroids_from_lab(label)
            # coords = centre_crop(centroids=centroids, window_size=window_size, h=h, w=w)
            # crops = label[...,coords[0],coords[1]].view(centroids.shape[0],1,window_size,window_size)
            # clean = (crops == ids[1:][:,None,None,None]) * ids[1:][:,None,None,None]

            # clean = connected_components((clean > 0).float(),num_iterations= 128)

            # for ii, cc_map in enumerate(clean):
            #     l= torch.unique(cc_map[cc_map>0], sorted=True)
            #     clean[ii] = cc_map == l[-1]
            # clean = clean.int()
            # clean = clean * torch.arange(clean.shape[0],device = clean.device,dtype = label.dtype)[:,None,None,None]

            # label = convert(clean,coords,size = [h,w]).int()



            labels.append(label.squeeze())


        if len(labels) == 1:
            return labels[0][None]  # 1,H,W
        else:
            return torch.stack(labels)  # 2,H,W
        


    def TTA_postprocessing(self, img, model, transforms,
                        mask_threshold: float = 0.53,
                        peak_distance: int = 5,
                        seed_threshold: float = 0.8,
                        overlap_threshold: float = 0.3,
                        mean_threshold: float = 0.1,
                        window_size: int =64,
                        min_size = 10,
                       device=None,
                       classifier=None,
                       cleanup_fragments: bool = False,
                       reduction = "mean",
                       max_seeds: int = 2000,):

        

        cells_and_nuclei = self.cells_and_nuclei
        if self.cells_and_nuclei:
            iterations = 2
            assert self.dim_out % 2 == 0,  print("The model should an even number of output channels for cells and nuclei.")
            dim_out = int(self.dim_out / 2)
        else:
            iterations = 1
            dim_out = self.dim_out

        out_labels = []

        transforms = [t for t in transforms] + [IdentityTransform()]
        
        for i in range(iterations):

            all_masks_list = []
            all_predictions = []

            self.cells_and_nuclei = False

            
            for t in transforms:
                with torch.cuda.amp.autocast():
                    augmented_image = t.augment_image(img)
                    augmented_image, pad = _instanseg_padding(augmented_image, extra_pad= 0, min_dim = 32)
                    prediction = model(augmented_image)[:,i * dim_out:(i+1) * dim_out]
                    prediction = _recover_padding(prediction, pad)
                    mask_map = prediction[:,-1][None] 
                    mask_map = t.deaugment_mask(mask_map)
                #  show_images(mask_map)
                    all_masks_list.append(mask_map.cpu())
                    all_predictions.append(prediction.cpu())

          #  pdb.set_trace()

            if reduction == "local_max":

                local_maxima_maps = [torch_peak_local_max(mask.squeeze().float().to(device), int(peak_distance),seed_threshold, return_map = True) for mask in all_masks_list]
                local_maxima_map = torch_peak_local_max(torch.stack(local_maxima_maps).max(0)[0].squeeze(),int(peak_distance),seed_threshold, return_map = True)
                all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)

            elif reduction in ["mean", "median"]:
                if reduction == "mean":
                    all_masks = torch.mean(torch.stack(all_masks_list),dim=0).float().to(device)
                elif reduction == "median":
                    all_masks = torch.median(torch.stack(all_masks_list),dim=0)[0].to(device)

                local_maxima_map = torch_peak_local_max(all_masks.squeeze(), neighbourhood_size=int(peak_distance), minimum_value=seed_threshold, return_map = True)
            
            local_maxima_map = (local_maxima_map > 0).float()
            centroids = torch.stack(torch.where(local_maxima_map.squeeze())).T


            if len(centroids) == 0:
                out_labels.append(torch.zeros((1,*all_masks.shape[-2:]), dtype=int, device=device))
                continue

            local_maxima_map[...,centroids[:,0],centroids[:,1]] = torch.arange(1,centroids.shape[0]+1,device = local_maxima_map.device).float()


            all_crops = []

            for (t, prediction) in (zip(transforms, all_predictions)):
                prediction_tmp = prediction.clone().float().to(device)
                prediction_tmp[:,-1] = t.augment_image(all_masks)
                prediction_tmp = prediction_tmp.squeeze(0)

                local_maxima_map_tmp = t.augment_image( local_maxima_map )
                centroids = torch.stack(torch.where(local_maxima_map_tmp.squeeze())).T
                values = local_maxima_map_tmp[...,centroids[:,0],centroids[:,1]]
                centroids = centroids[values.sort()[1]][0,0]

                out = self.postprocessing(prediction_tmp, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier, 
                                                            cleanup_fragments, max_seeds, return_intermediate_objects = True, precomputed_seeds = centroids)
                
                if len(out)==3:
                    crops, coords, mask_map = out
                else:
                    pdb.set_trace()

                crops = t.deaugment_mask(crops)
                all_crops.append(crops.cpu())


         #   show_images(torch.cat([*torch.cat(all_crops,dim = 3)[:50]],dim = 1),colorbar= False)


            all_crops = torch.median(torch.stack(all_crops).float(),dim=0)[0].to(device)
         #   all_crops = torch.mean(torch.stack(all_crops).float(),dim=0).to(device)
         #   all_crops = torch.max(torch.stack(all_crops).float(),dim=0)[0].to(device)


            labels = self.postprocessing(prediction, mask_threshold, peak_distance, seed_threshold, overlap_threshold, mean_threshold, window_size, min_size, device, classifier,
                                        cleanup_fragments, max_seeds, precomputed_crops = (all_crops, coords, mask_map))

            
            out_labels.append(labels)
        self.cells_and_nuclei = cells_and_nuclei

        labels = torch.stack(out_labels, dim = 1).squeeze(0)
        #show_images(labels)
        return labels


        
class IdentityTransform:
    def augment_image(self, img):
        return img
    def deaugment_mask(self, mask):
        return mask

        

from instanseg.utils.biological_utils import resolve_cell_and_nucleus_boundaries
from typing import Dict, Optional
class InstanSeg_Torchscript(nn.Module):
    def __init__(self, model, 
                 cells_and_nuclei: bool = False,
                 pixel_size : float = 0, 
                 n_sigma: int = 2, 
                 dim_coords:int = 2, 
                 to_centre: bool = True,
                 backbone_dim_in: int = 3,  
                 feature_engineering_function:str  = "0",
                 params = None):
        super(InstanSeg_Torchscript, self).__init__()

        model.eval()

        use_mixed_precision = True

        with torch.amp.autocast("cuda", enabled=use_mixed_precision):
            with torch.no_grad():
                self.fcn = torch.jit.trace(model, torch.rand(1, backbone_dim_in, 256, 256))

        try:
            self.pixel_classifier = model.pixel_classifier
        except:
            self.pixel_classifier = model.model.pixel_classifier  # I think this is a pytorch version issue between 1.13.1 and 2.0.0
        self.cells_and_nuclei = cells_and_nuclei
        self.pixel_size = pixel_size
        self.dim_coords = dim_coords
        self.n_sigma = n_sigma
        self.to_centre = to_centre
        self.feature_engineering, self.feature_engineering_width = feature_engineering_generator(feature_engineering_function)
        self.params = params or {}
        self.index_dtype = torch.long #torch.int

        self.default_target_segmentation = self.params.get('target_segmentation', torch.tensor([1, 1]))
        self.default_min_size = self.params.get('min_size', 10)
        self.default_mask_threshold = self.params.get('mask_threshold', 0.53)
        self.default_peak_distance = int(self.params.get('peak_distance', 5))
        self.default_seed_threshold = self.params.get('seed_threshold', 0.7)
        self.default_overlap_threshold = self.params.get('overlap_threshold', 0.3)
        self.default_mean_threshold = self.params.get('mean_threshold', 0.0)
        self.default_window_size = self.params.get('window_size', 32)
        self.default_cleanup_fragments = self.params.get('cleanup_fragments', False)
        self.default_resolve_cell_and_nucleus = self.params.get('resolve_cell_and_nucleus', True)

    def forward(self, x: torch.Tensor,
                args: Optional[Dict[str, torch.Tensor]] = None,
                target_segmentation: torch.Tensor = torch.tensor([1, 1]), # Nuclei / Cells
                min_size: Optional[int] = None,
                mask_threshold: Optional[float] = None,
                peak_distance: Optional[int] = None,
                seed_threshold: Optional[float] = None,
                overlap_threshold: Optional[float] = None,
                mean_threshold: Optional[float] = None,
                window_size: Optional[int] = None,
                cleanup_fragments: Optional[bool] = None,
                resolve_cell_and_nucleus: Optional[bool] = None,
                precomputed_seeds: torch.Tensor = torch.tensor([]),
                ) -> torch.Tensor:
        
        min_size = int(min_size) if min_size is not None else self.default_min_size
        mask_threshold = float(mask_threshold) if mask_threshold is not None else self.default_mask_threshold
        peak_distance = int(peak_distance) if peak_distance is not None else self.default_peak_distance
        seed_threshold = float(seed_threshold) if seed_threshold is not None else self.default_seed_threshold
        overlap_threshold = float(overlap_threshold) if overlap_threshold is not None else self.default_overlap_threshold
        mean_threshold = float(mean_threshold) if mean_threshold is not None else self.default_mean_threshold
        window_size = int(window_size) if window_size is not None else self.default_window_size
        cleanup_fragments = bool(cleanup_fragments) if cleanup_fragments is not None else self.default_cleanup_fragments
        resolve_cell_and_nucleus = bool(resolve_cell_and_nucleus) if resolve_cell_and_nucleus is not None else self.default_resolve_cell_and_nucleus

        if args is None:
            args = {"None": torch.tensor([0])}

        target_segmentation = args.get('target_segmentation', target_segmentation)
        min_size = int(args.get('min_size', torch.tensor(float(min_size))).item())
        mask_threshold = args.get('mask_threshold', torch.tensor(mask_threshold)).item()
        peak_distance = args.get('peak_distance', torch.tensor(peak_distance)).item()
        seed_threshold = args.get('seed_threshold', torch.tensor(seed_threshold)).item()
        overlap_threshold = args.get('overlap_threshold', torch.tensor(overlap_threshold)).item()
        mean_threshold = args.get('mean_threshold', torch.tensor(mean_threshold)).item()
        window_size = int(args.get('window_size', torch.tensor(float(window_size))).item())
        cleanup_fragments = args.get('cleanup_fragments', torch.tensor(cleanup_fragments)).item()
        resolve_cell_and_nucleus = args.get('resolve_cell_and_nucleus', torch.tensor(resolve_cell_and_nucleus)).item()
        precomputed_seeds = args.get('precomputed_seeds', precomputed_seeds)

        torch.clamp_max_(x, 3) #Safety check, please normalize inputs properly!
        torch.clamp_min_(x, -2)

        x, pad = _instanseg_padding(x, extra_pad=0)


        with torch.no_grad():
            x_full = self.fcn(x)
    
            dim_out = x_full.shape[1]

            if self.cells_and_nuclei:
                iterations = torch.tensor([0,1]) [target_segmentation.squeeze().to("cpu") > 0 ]
                dim_out = int(dim_out / 2)

            else:
                iterations = torch.tensor([0])
                dim_out = dim_out

            output_labels_list = []

            for image_index in range(x_full.shape[0]):
                labels_list = []
                for i in iterations:
                    if i == 0:
                        x = x_full[image_index,0: dim_out, :, :]
                    else:
                        x = x_full[image_index,dim_out:, :, :]

                    x = _recover_padding(x, pad)

                    height, width = x.size(1), x.size(2)

                    xxyy = generate_coordinate_map(mode = "linear", spatial_dim = self.dim_coords, height = height, width = width, device = x.device)


                    if not self.to_centre:
                        fields = (torch.sigmoid(x[0:self.dim_coords])-0.5) * 8
                    else:
                        fields = x[0:self.dim_coords] 

                
                    sigma = x[self.dim_coords:self.dim_coords + self.n_sigma]

                    #mask_map = torch.sigmoid(x[self.dim_coords + self.n_sigma]) #legacy
                    mask_map = ((x[self.dim_coords + self.n_sigma]) / 15) + 0.5 # inverse transform applied to edt during training.

                    if precomputed_seeds is None or precomputed_seeds.shape[0] == 0:
                        centroids_idx = torch_peak_local_max(mask_map, neighbourhood_size=peak_distance,
                                                            minimum_value=seed_threshold, dtype= self.index_dtype)  # .to(prediction.device)
                    else:
                        centroids_idx = precomputed_seeds.to(mask_map.device).long()

                    fields = fields + xxyy

                    if self.to_centre:
                        fields_at_centroids = xxyy[:, centroids_idx[:, 0], centroids_idx[:, 1]]
                    else:
                        fields_at_centroids = fields[:, centroids_idx[:, 0], centroids_idx[:, 1]]

                    x = fields
                    c = fields_at_centroids.T
                    E = x.shape[0]
                    h, w = x.shape[-2:]
                    C = c.shape[0]
                    S = sigma.shape[0]

                    if C == 0:
                        label = torch.zeros(mask_map.shape, dtype= torch.float32, device=mask_map.device).squeeze()
                        labels_list.append(label)
                        continue

                    window_size = window_size
                    centroids = centroids_idx.clone().cpu()  # C,2
                    centroids[:, 0].clamp_(min=window_size, max=h - window_size)
                    centroids[:, 1].clamp_(min=window_size, max=w - window_size)
                    window_slices = centroids[:, None].to(x.device) + torch.tensor([[-1, -1], [1, 1]] , device = x.device, dtype=centroids.dtype) * window_size
                    window_slices = window_slices  # C,2,2

                    slice_size = window_size * 2

                    # Create grids of indices for slice windows
                    grid_x, grid_y = torch.meshgrid(
                        torch.arange(slice_size, device=x.device, dtype=self.index_dtype),
                        torch.arange(slice_size, device=x.device, dtype=self.index_dtype), indexing="ij")
                    mesh = torch.stack((grid_x, grid_y))

                    mesh_grid = mesh.expand(C, 2, slice_size, slice_size)  # C,2,2*window_size,2*window_size
                    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,C,2*window_size*2*window_size
                    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
                    mesh_grid_flat = mesh_grid_flat + idx
                    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)  # 2,C*2*window_size*2*window_size

                #    x = self.traced_feature_engineering(x, c, sigma, torch.tensor(window_size).int(), mesh_grid_flat)
                    x = feature_engineering_slow(x, c, sigma, torch.tensor(window_size).int(), mesh_grid_flat)

                    x = torch.sigmoid(self.pixel_classifier(x))

                    x = x.reshape(C, 1, slice_size, slice_size)

                    C = x.shape[0]

                    if C == 0:
                        label = torch.zeros(mask_map.shape, dtype= torch.float32, device=mask_map.device).squeeze()
                        labels_list.append(label)
                        continue

                    original_device = x.device

                    if x.is_mps:
                        device = 'cpu'
                        mesh_grid_flat = mesh_grid_flat.to(device)
                        x = x.to(device)
                        mask_map = mask_map.to(device)

                    coords = mesh_grid_flat.reshape(2, C, slice_size, slice_size)

                    if cleanup_fragments:

                        top_left = window_slices[:,0,:]
                        shifted_centroid = centroids_idx - top_left
                        cc = connected_components((x>mask_threshold).float(),num_iterations= 64)
                        labels_to_keep = cc[torch.arange(cc.shape[0]),0,shifted_centroid[:,0],shifted_centroid[:,1]]
                        in_mask = cc == labels_to_keep[:,None,None,None]
                        x *= in_mask

                    labels = convert(x, coords, size=(h, w), mask_threshold=mask_threshold)[None]


                    idx = torch.arange(1, C + 1, device=x.device, dtype = self.index_dtype)
                    stack_ID = torch.ones((C, slice_size, slice_size), device=x.device, dtype=self.index_dtype)
                    stack_ID = stack_ID * (idx[:, None, None] - 1)

                    iidd = torch.stack((stack_ID.flatten(), mesh_grid_flat[0] * w + mesh_grid_flat[1]))

                    fg = x.flatten() > mask_threshold
                    x = x.flatten()[fg]
                    sparse_onehot = torch.sparse_coo_tensor(
                        iidd[:, fg],
                        (x.flatten() > mask_threshold).float(),
                        size=(C, h * w),
                        dtype=x.dtype,
                        device=x.device
                    )

                    object_areas = torch.sparse.sum(sparse_onehot.to(torch.bool).float(), dim=(1,)).values()
                    sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map.flatten()[None]), dim=(1,)).values()
                    mean_mask_value = sum_mask_value / object_areas
                    objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

                    iou = fast_sparse_iou(sparse_onehot)

                    remapping = find_connected_components((iou > overlap_threshold).to(self.index_dtype))
                    
                    labels = remap_values(remapping, labels)


                    labels_to_remove = (torch.arange(0, len(objects_to_remove), device=objects_to_remove.device) + 1)[
                        objects_to_remove]
                    labels[torch.isin(labels, labels_to_remove)] = 0

                    
                    labels_list.append(labels.squeeze().to(original_device))

                if len(labels_list) == 1:
                    lab = labels_list[0][None, None]  # 1,1,H,W
                else:
                    lab = torch.stack(labels_list)[None] 

                if lab.shape[1] == 2 and resolve_cell_and_nucleus: #nuclei and cells
                    lab = resolve_cell_and_nucleus_boundaries(lab)

                output_labels_list.append(lab[0])
            
            lab = torch.stack(output_labels_list) # B,C,H,W

        
            return lab.to(torch.float32) # B,C,H,W






