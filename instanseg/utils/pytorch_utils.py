import torch
import torch.nn.functional as F

from typing import Tuple

def remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Remap the values in x according to the pairs in the remapping tensor.

    :param remapping: A tensor of shape (2, N) containing the remapping pairs.
    :param x: The input tensor to be remapped.
    :return: The remapped tensor.
    """
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


def torch_fastremap(x: torch.Tensor) -> torch.Tensor:
    """
    Fast remap the values in x to a contiguous range starting from 0.

    :param x: The input tensor to be remapped.
    :return: The remapped tensor.
    """
    if x.max() == 0:
        return x
    unique_values = torch.unique(x, sorted=True)
    new_values = torch.arange(len(unique_values), dtype=x.dtype, device=x.device)
    remapping = torch.stack((unique_values, new_values))
    return remap_values(remapping, x)

def torch_onehot(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a labeled image to a one-hot encoded tensor.

    :param x: The input labeled image tensor of shape (_, _, H, W).
    :return: The one-hot encoded tensor of shape (1, C, H, W).
    """

    if x.max() == 0:
        return torch.zeros_like(x).reshape(1, 0, *x.shape[-2:])
    H, W = x.shape[-2:]
    x = x.view(-1, 1, H, W)
    x = x.squeeze().view(1, 1, H, W)
    unique = torch.unique(x[x > 0])
    x = x.repeat(1, len(unique), 1, 1)
    return x == unique.unsqueeze(-1).unsqueeze(-1)

def fast_iou(onehot: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) for a one-hot encoded tensor.

    :param onehot: The one-hot encoded tensor of shape (C, H, W).
    :param threshold: The threshold for binarization.
    :return: The IoU tensor.
    """
    if onehot.ndim == 3:
        onehot = onehot.flatten(1)
    onehot = (onehot > threshold).float()
    intersection = onehot @ onehot.T
    union = onehot.sum(1)[None].T + onehot.sum(1)[None] - intersection
    return intersection / union

def fast_sparse_iou(sparse_onehot: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) for a sparse one-hot encoded tensor.

    :param sparse_onehot: The sparse one-hot encoded tensor.
    :return: The IoU tensor.
    """
    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,))[None].to_dense()
    union = sparse_sum.T + sparse_sum - intersection
    return intersection / union

def instance_wise_edt(x: torch.Tensor, edt_type: str = 'auto') -> torch.Tensor:
    """
    Create instance-normalized distance map from a labeled image.
    Each pixel within an instance gives the distance to the closest background pixel,
    divided by the maximum distance (so that the maximum within an instance is 1).

    :param x: The input labeled image tensor.
    :param edt_type: The type of Euclidean Distance Transform to use ('auto', 'edt', or 'monai'). 'edt' is faster for CPU computation, while 'monai' can use cucim for GPU acceleration where CUDA is available. Use 'auto' to decide automatically.
    :return: The instance-normalized distance map tensor.
    """
    if x.max() == 0:
        return torch.zeros_like(x).squeeze()
    is_mps = x.is_mps
    if is_mps:
        # Need to convert to CPU for MPS, because distance transform gives float64 result
        # and Monai's internal attempt to convert type will fail
        x = x.to('cpu')

    use_edt = edt_type == 'edt' or (edt_type != 'monai' and not x.is_cuda)
    if use_edt:
        import edt
        xedt = torch.from_numpy(edt.edt(x[0].cpu().numpy(), black_border=False))
        x = torch_onehot(x)[0] * xedt.to(x.device)
    else:
        import monai
        x = torch_onehot(x)
        x = monai.transforms.utils.distance_transform_edt(x[0])

    # Normalize instance distances to have max 1
    x = x / (x.flatten(1).max(1)[0]).view(-1, 1, 1)
    x = x.sum(0)

    if is_mps:
        x = x.type(torch.FloatTensor).to('mps')
    return x

def fast_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the intersection over union between two dense onehot encoded tensors.

    :param onehot1: The first one-hot encoded tensor of shape (C1, H, W).
    :param onehot2: The second one-hot encoded tensor of shape (C2, H, W).
    :return: The IoU tensor of shape (C1, C2).
    """

    C1 = onehot1.shape[0]
    C2 = onehot2.shape[0]

    max_C = max(C1, C2)

    onehot1 = torch.cat((onehot1, torch.zeros((max_C - C1, *onehot1.shape[1:]))), dim=0)
    onehot2 = torch.cat((onehot2, torch.zeros((max_C - C2, *onehot2.shape[1:]))), dim=0)

    onehot1 = onehot1.flatten(1)
    onehot1 = (onehot1 > 0.5).float()  # onehot should be binary

    onehot2 = onehot2.flatten(1)
    onehot2 = (onehot2 > 0.5).float()

    intersection = onehot1 @ onehot2.T
    union = (onehot1).sum(1)[None].T + (onehot2).sum(1)[None] - intersection

    return (intersection / union)[:C1, :C2]

def torch_sparse_onehot(x: torch.Tensor, flatten: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a labeled image to a sparse one-hot encoded tensor.

    :param x: The input labeled image tensor of shape (_, _, H, W).
    :param flatten: Whether to flatten the output tensor.
    :return: A tuple containing the sparse one-hot encoded tensor and the unique values tensor.
    """
    unique_values = torch.unique(x, sorted=True)
    x = torch_fastremap(x)

    H, W = x.shape[-2], x.shape[-1]

    if flatten:
        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 1, H*W)[:,:0] , unique_values

        x = x.reshape(H * W)
        xxyy = torch.nonzero(x > 0).squeeze(1)
        zz = x[xxyy] - 1
        C = x.max().int().item()
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, xxyy)).long(), (torch.ones_like(xxyy).float()),
                                                size=(int(C), int(H * W)), dtype=torch.float32)

    else:
        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 0, H,W) , unique_values

        x = x.squeeze().view(H, W)
        x_temp= torch.nonzero(x > 0).T
        zz = x[x_temp[0], x_temp[1]] - 1
        C = x.max().int().item()
        sparse_onehot = torch.sparse_coo_tensor(torch.stack((zz, x_temp[0], x_temp[1])).long(), (torch.ones_like(x_temp[0]).float()),
                                                size=(int(C), int(H), int(W)), dtype=torch.float32)

    return sparse_onehot, unique_values


def fast_sparse_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the (dense) intersection over union between two sparse onehot encoded tensors.

    :param onehot1: The first sparse one-hot encoded tensor of shape (C1, H*W).
    :param onehot2: The second sparse one-hot encoded tensor of shape (C2, H*W).
    :return: The IoU tensor of shape (C1, C2).
    """
    intersection = torch.sparse.mm(onehot1, onehot2.T).to_dense()
    sparse_sum1 = torch.sparse.sum(onehot1, dim=(1,))[None].to_dense()
    sparse_sum2 = torch.sparse.sum(onehot2, dim=(1,))[None].to_dense()
    union = sparse_sum1.T + sparse_sum2 - intersection

    return (intersection / union)


def match_labels(tile_1: torch.Tensor,tile_2: torch.Tensor,threshold: float = 0.5, strict = False):
    """
    Match the overlapping labels of tile_2 to the labels of tile_1.

    :param tile_1: The first labeled tile tensor.
    :param tile_2: The second labeled tile tensor.
    :param threshold: The IoU threshold for matching.
    :param strict: Whether to discard non-matching objects.
    :return: The matched label tensors.
    """

    
    if tile_1.max() == 0 or tile_2.max() == 0:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)
        
    old_problematic_onehot, old_unique_values = torch_sparse_onehot(tile_1, flatten=True)
    new_problematic_onehot, new_unique_values = torch_sparse_onehot(tile_2, flatten=True)

    iou = fast_sparse_dual_iou(old_problematic_onehot, new_problematic_onehot)

    onehot_remapping = torch.nonzero(iou > threshold).T# + 1

    if old_unique_values.min() == 0:
       old_unique_values = old_unique_values[old_unique_values > 0]
    if new_unique_values.min() == 0:
       new_unique_values = new_unique_values[new_unique_values > 0]

    if onehot_remapping.shape[1] > 0:
        
        onehot_remapping = torch.stack((new_unique_values[onehot_remapping[1]], old_unique_values[onehot_remapping[0]]))

        if not strict:
            mask = torch.isin(tile_2, onehot_remapping[0])
            tile_2[mask] = remap_values(onehot_remapping, tile_2[mask])

            return tile_1, tile_2
        else:
            tile_1 = tile_1 * torch.isin(tile_1, onehot_remapping[1]).int()
            tile_2 = tile_2 * torch.isin(tile_2, onehot_remapping[0]).int()

            tile_2[tile_2>0] = remap_values(onehot_remapping, tile_2[tile_2>0])

            return tile_1, tile_2
        
    else:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)
        

def connected_components(x: torch.Tensor, num_iterations: int = 32) -> torch.Tensor:
    """
    This function takes a binary image and returns the connected components.

    :param x: The input binary image tensor.
    :param num_iterations: The number of iterations for the connected components algorithm.
    :return: The tensor with connected components labeled.
    """
    mask = x == 1

    B, _, H, W = x.shape
    out = torch.arange(B * W * H, device=x.device, dtype=x.dtype).reshape((B, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out


def iou_heatmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    This function takes two labeled images and returns the intersection over union heatmap.

    :param x: The first labeled image tensor of shape (H, W).
    :param y: The second labeled image tensor of shape (H, W).
    :return: The IoU heatmap tensor.
    """
    if x.max() ==0 or y.max() == 0:
        return torch.zeros_like(x)

    x = torch_fastremap(x)
    y = torch_fastremap(y)

    x_onehot, _ = torch_sparse_onehot(x, flatten=True)
    y_onehot, _ = torch_sparse_onehot(y, flatten=True)

    iou = fast_sparse_dual_iou(x_onehot, y_onehot)
    predicted_iou = iou.sum(1)
    onehot = torch_onehot(x)
    onehot = onehot.float() * predicted_iou[:,None, None]
    map = onehot.max(1)[0]

    return map

def centroids_from_lab(lab: torch.Tensor):
    """
    Calculate the centroids of labeled objects in an image.

    :param lab: The label tensor of shape (H, W).
    :return: A tuple containing the centroids tensor and the label IDs tensor.
    """
    mesh_grid = torch.stack(torch.meshgrid(torch.arange(lab.shape[-2], device = lab.device), torch.arange(lab.shape[-1],device = lab.device), indexing="ij")).float()

    sparse_onehot, label_ids = torch_sparse_onehot(lab, flatten=True)
    sum_centroids = torch.sparse.mm(sparse_onehot, mesh_grid.flatten(1).T)
    centroids = sum_centroids / torch.sparse.sum(sparse_onehot, dim=(1,)).to_dense().unsqueeze(-1)
    return centroids, label_ids  # N,2  N


def get_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64, return_lab_ids: bool = False):
    """
    Extract patches from an image based on labeled objects.

    :param lab: The label tensor of shape (1, H, W) with N objects.
    :param image: The input image tensor of shape (C, H, W).
    :param patch_size: The size of the patches to extract.
    :param return_lab_ids: Whether to return the label IDs of the patches.
    :return: The patches tensor of shape (N, C, patch_size, patch_size), and optionally the label IDs tensor.
    """
    centroids, label_ids = centroids_from_lab(lab)
    N = centroids.shape[0]

    C, h, w = image.shape[-3:]

    window_size = patch_size // 2
    centroids = centroids.clone()  # N,2
    centroids[:, 0] = centroids[:,0].clamp(min=window_size, max=h - window_size)
    centroids[:, 1] = centroids[:,1].clamp(min=window_size, max=w - window_size)
    window_slices = centroids[:, None] + torch.tensor([[-1, -1], [1, 1]]).to(image.device) * window_size
    window_slices = window_slices.long()  # N,2,2

    slice_size = window_size * 2

    # Create grids of indices for slice windows
    grid_x, grid_y = torch.meshgrid(
        torch.arange(slice_size, device=image.device),
        torch.arange(slice_size, device=image.device), indexing="ij")
    mesh = torch.stack((grid_x, grid_y))

    mesh_grid = mesh.expand(N, 2, slice_size, slice_size)  # N,2,2*window_size,2*window_size
    mesh_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,N,2*window_size*2*window_size
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_flat = mesh_flat + idx
    mesh_flater = torch.flatten(mesh_flat, 1)  # 2,N*2*window_size*2*window_size

    out = image[:, mesh_flater[0], mesh_flater[1]].reshape(C, N, -1)
    out = out.reshape(C, N, patch_size, patch_size)
    out = out.permute(1, 0, 2, 3)

    if return_lab_ids:
        return out, label_ids

    return out,label_ids  # N,C,patch_size,patch_size


def get_masked_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64):
    """
    Extract patches from an image based on labeled objects.

    :param lab: The label tensor of shape (1, H, W) with N objects.
    :param image: The input image tensor of shape (C, H, W).
    :param patch_size: The size of the patches to extract.
    :param return_lab_ids: Whether to return the label IDs of the patches.
    :return: The patches tensor of shape (N, C, patch_size, patch_size), and optionally the label IDs tensor.
    """

    lab_patches, label_ids = get_patches(lab, lab[0], patch_size)
    mask_patches = lab_patches == label_ids[1:, None, None, None]

    image_patches,_ = get_patches(lab, image, patch_size)

    return image_patches,mask_patches

def feature_extractor():
    """
    Create a ResNet feature extractor without initial downsize.

    :return: The ResNet feature extractor model.
    """
    import torch
    from torchvision.models.resnet import ResNet
    from torchvision.models.resnet import ResNet18_Weights
    import torch.nn as nn
    from typing import Type, Union, List, Optional, Any
    from torchvision.models.resnet import BasicBlock, Bottleneck, WeightsEnum

    class ResNetNoInitialDownsize(ResNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

    def _resnet_custom(
            resnet_constructor,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            weights: Optional[WeightsEnum],
            progress: bool,
            **kwargs: Any,
    ) -> ResNet:

        model = resnet_constructor(block, layers, **kwargs)

        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=progress))

        return model

    weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
    model = _resnet_custom(ResNetNoInitialDownsize, BasicBlock, [2, 2, 2, 2], weights, progress=True)

    return model

def eccentricity_batch(mask_tensor):
    """
    Calculate the eccentricity of a batch of binary masks.

    :param mask_tensor: The input binary mask tensor of shape (B, H, W).
    :return: A tensor containing the eccentricity values for each mask in the batch.
    """
    
    # Get dimensions
    batch_size, m, n = mask_tensor.shape
    
    # Create indices grid
    y_indices, x_indices = torch.meshgrid(torch.arange(m), torch.arange(n), indexing='ij')
    y_indices = y_indices.unsqueeze(0).to(mask_tensor.device).expand(batch_size, m, n)
    x_indices = x_indices.unsqueeze(0).to(mask_tensor.device).expand(batch_size, m, n)
    
    # Find total mass and centroid
    total_mass = mask_tensor.sum(dim=(1, 2))
    centroid_y = (y_indices * mask_tensor).sum(dim=(1, 2)) / total_mass
    centroid_x = (x_indices * mask_tensor).sum(dim=(1, 2)) / total_mass
    
    # Calculate second-order moments
    y_diff = y_indices - centroid_y.view(batch_size, 1, 1)
    x_diff = x_indices - centroid_x.view(batch_size, 1, 1)
    M_yy = torch.sum(y_diff**2 * mask_tensor, dim=(1, 2))
    M_xx = torch.sum(x_diff**2 * mask_tensor, dim=(1, 2))
    M_xy = torch.sum(x_diff * y_diff * mask_tensor, dim=(1, 2))

    # Construct second-order moments tensor
    moments_tensor = torch.stack([torch.stack([M_xx, M_xy]),
                                  torch.stack([M_xy, M_yy])]).permute(2,0,1)
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(moments_tensor)

    # Get maximum eigenvalue
    lambda1 = torch.max(eigenvalues.real, dim=1).values
    # Get minimum eigenvalue
    lambda2 = torch.min(eigenvalues.real, dim=1).values
    
    # Calculate eccentricity
    eccentricity = torch.sqrt(1 - (lambda2 / lambda1))
    
    return eccentricity.squeeze(1,2)
