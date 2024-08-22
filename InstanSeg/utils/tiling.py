import torch
import numpy as np
import torch.nn.functional as F

import pdb

from InstanSeg import show_images



def edge_mask(labels, ignore=[None]):
    labels = labels.squeeze()
    first_row = labels[0, :]
    last_row = labels[-1, :]
    first_column = labels[:, 0]
    last_column = labels[:, -1]

    edges = []
    if 'top' not in ignore:
        edges.append(first_row)
    if 'bottom' not in ignore:
        edges.append(last_row)
    if 'left' not in ignore:
        edges.append(first_column)
    if 'right' not in ignore:
        edges.append(last_column)

    edges = torch.cat(edges, dim=0)
    return torch.isin(labels, edges[edges > 0])


def remove_edge_labels(labels, ignore=[None]):
    return labels * ~edge_mask(labels, ignore=ignore)


def _to_shape(a, shape):
    """Pad a tensor to a given shape."""
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    y_, x_ = shape
    y, x = a[0].shape[-2:]
    y_pad = max(0, y_ - y)
    x_pad = max(0, x_ - x)
    return torch.nn.functional.pad(a, (x_pad // 2, x_pad // 2 + x_pad % 2, y_pad // 2, y_pad // 2 + y_pad % 2))


def _to_shape_bottom_left(a, shape):
    """Pad a tensor to a given shape."""
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    y_, x_ = shape
    y, x = a[0].shape[-2:]
    y_pad = max(0, y_ - y)
    x_pad = max(0, x_ - x)
    return torch.nn.functional.pad(a, (0, x_pad, 0, y_pad))

def chops(img_shape: tuple, shape: tuple, overlap: int = 0) -> tuple:
    """This function splits an image into desired windows and returns the indices of the windows"""

    if (torch.tensor(img_shape[-2:]) < torch.tensor(shape)).any():
        return [0], [0]
    h, v = img_shape[-2:]


    stride_h = shape[0] - overlap
    stride_v = shape[1] - overlap

        
    max_v = (v - shape[1])
    max_h = (h - shape[0])

    v_index = np.array([i * stride_v for i in range(v // stride_v )] + [v - shape[1]])
    h_index = np.array([i * stride_h for i in range(h // stride_h )] + [h - shape[0]])

    v_index = v_index[v_index <= max_v]
    h_index = h_index[h_index <= max_h]

    v_index = np.unique(v_index)
    h_index = np.unique(h_index)

    return h_index, v_index


def tiles_from_chops(image: torch.Tensor, shape: tuple, tuple_index: tuple) -> list:
    """This function takes an image, a shape, and a tuple of window indices (e.g., output of the function chops)
    and returns a list of windows"""
    h_index, v_index = tuple_index

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    stride_h = shape[0]
    stride_v = shape[1]
    tile_list = []
    for i, window_i in enumerate(h_index):
        for j, window_j in enumerate(v_index):
            current_window = image[..., window_i:window_i + stride_h, window_j:window_j + stride_v]
            tile_list.append(current_window)
    return tile_list


def stitch(tiles: list, shape: tuple, chop_list: list, final_shape: tuple, offset : int):
    """This function takes a list of tiles, a shape, and a tuple of window indices (e.g., outputed by the function chops)
    and returns a stitched image"""
    from InstanSeg.utils.pytorch_utils import torch_fastremap, match_labels

    canvas = torch.zeros(final_shape, dtype=torch.int32, device = tiles[0].device)

    running_max = 0
            
    for i, window_i in enumerate(chop_list[0]):
        for j, window_j in enumerate(chop_list[1]):

            edge_window = False
            new_tile = tiles[i * len(chop_list[1]) + j][None]
            ignore_list = []
            if i == 0:
                ignore_list.append("top")
            if j == 0:
                ignore_list.append("left")
            if i == len(chop_list[0])-1:
                ignore_list.append("bottom")
            if j == len(chop_list[1])-1:
                ignore_list.append("right")

            if i == len(chop_list[0])-1 and j == len(chop_list[1])-1:
                edge_window = True
                tile1 = canvas[..., window_i + offset:window_i + shape[0], window_j + offset:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:,offset:shape[0],offset: shape[1]], ignore = ignore_list)

            elif i == len(chop_list[0])-1:
                edge_window = True
                tile1 = canvas[..., window_i + offset:window_i + shape[0], window_j + offset:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:,offset:shape[0],offset : shape[1]], ignore =ignore_list)

            elif j == len(chop_list[1])-1:
                edge_window = True
                tile1 = canvas[..., window_i + offset:window_i + shape[0], window_j + offset:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:,offset:shape[0],offset: shape[1]], ignore = ignore_list)

            if i == 0 and j == 0:
                edge_window = True
                tile1 = canvas[..., window_i  :window_i + shape[0], window_j :window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:, :shape[0], : shape[1]], ignore = ignore_list)
            elif i == 0:
                edge_window = True
                tile1 = canvas[..., window_i  :window_i + shape[0], window_j + offset :window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:, :shape[0],offset : shape[1]], ignore = ignore_list)

            elif j == 0:
                edge_window = True
                tile1 = canvas[..., window_i  + offset:window_i + shape[0], window_j:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[:,offset :shape[0],: shape[1]], ignore = ignore_list)

            if edge_window:

                tile2 = torch_fastremap(tile2)
                tile2[tile2>0] = tile2[tile2>0] + running_max

                remapped = match_labels(tile1, tile2, threshold = 0.1)[1]
                tile1[remapped>0] = remapped[remapped>0].int()

                running_max = max(running_max, tile1.max())

            else:
                
                tile1 = canvas[..., window_i + offset:window_i + shape[0] - offset, window_j + offset:window_j + shape[1] - offset]
                tile2 = remove_edge_labels(new_tile[:,offset:shape[0] -offset,offset: shape[1]-offset])


                tile2 = torch_fastremap(tile2)
                tile2[tile2>0] = tile2[tile2>0] + running_max

                remapped = match_labels(tile1, tile2, threshold = 0.1)[1]
                tile1[remapped>0] = remapped[remapped>0].int()

                running_max = max(running_max, tile1.max())
            

    return canvas


def zarr_to_json_export(path_to_zarr, cell_size = 30, size = 1024, scale = 1, n_dim = 1):

    import zarr
    import numpy as np
    import os

    z = zarr.open(path_to_zarr, mode='r')

    output_path = str(path_to_zarr).replace(".zarr",".geojson")   


    from InstanSeg.utils.tiling import chops

    chop_list = chops(z.shape, shape=(size,size), overlap=cell_size)

    from InstanSeg.utils.utils import labels_to_features
    from InstanSeg.utils.tiling import remove_edge_labels
    import json

    count = 0
    #Delete the file if it already exists
    if os.path.exists( output_path):
        os.remove( output_path)

    with open(os.path.join(output_path), "a") as outfile:
        outfile.write('[')
        outfile.write('\n')

    if n_dim == 1:
        classes = [None]
    else:
        classes = ["Nucleus","Cell"]


    for i, window_i in tqdm(enumerate(chop_list[0]), total = len(chop_list[0])):
        for j, window_j in enumerate(chop_list[1]):
            for n in range(n_dim):
                
                image = z[n,window_i:window_i+size, window_j:window_j+size]
              
                image = remove_edge_labels(torch.tensor(image)).numpy()

                features = labels_to_features(image.astype(np.int32), object_type='detection', include_labels=True,
                                        classification=classes[n],offset=[window_j*scale,window_i*scale], downsample = scale)


            if features != []:
                count+=1
                geojson = json.dumps(features)

                with open(os.path.join( output_path), "a") as outfile:
                    outfile.write(geojson[1:-1] + ",")
                    outfile.write('\n')

    with open(os.path.join( output_path), "a") as outfile:
        outfile.write(']')





def segment_image_larger_than_memory(instanseg_folder: str, 
                                     image_path: str, 
                                     shape: tuple = (512,512), 
                                     overlap = 100,
                                    cell_size = 20, 
                                    threshold: int = 200, 
                                    to_geojson = False, 
                                    driver = "AUTO", 
                                    use_torchscript = False,
                                    pixel_size: float = None):
    
    """This function uses slideio to read an image and then segments it using the instanseg model. 
    The segmentation is done in a tiled manner to avoid memory issues. 
    The function returns a zarr file with the segmentation. The zarr file is saved in the same directory as the image with the same name but with the extension .zarr. 
    The function also returns the zarr file object."""

    from InstanSeg.utils.model_loader import load_model
    from InstanSeg.utils.loss.instanseg_loss import InstanSeg
    from InstanSeg.utils.utils import read_pixel_size
    import slideio
    import zarr
    from InstanSeg.utils.augmentations import Augmentations
    from itertools import product
    from InstanSeg.utils.pytorch_utils import torch_fastremap, match_labels

    device = 'cuda'

    if not use_torchscript:
    
        model, model_dict = load_model(folder= instanseg_folder)

        
        method = InstanSeg( n_sigma=model_dict["n_sigma"],
                            cells_and_nuclei=model_dict["cells_and_nuclei"], 
                            to_centre = model_dict["to_centre"], 
                            window_size = 64, 
                            dim_coords= 2,
                            feature_engineering_function = model_dict["feature_engineering"], 
                            device = device)
        
        n_dim = 2 if model_dict["cells_and_nuclei"] else 1
        model_pixel_size = model_dict["pixel_size"]

        method.initialize_pixel_classifier(model)

        model.eval()
        model.to(device)

    else:
        instanseg = torch.jit.load("../torchscripts/" + instanseg_folder + ".pt")
        instanseg.to(device)
        
        n_dim = 2 if instanseg.cells_and_nuclei else 1
        model_pixel_size = instanseg.pixel_size

    from pathlib import Path

    
    file_with_zarr_extension = Path(image_path).with_suffix('.zarr')


    slide = slideio.open_slide(image_path, driver = driver)
    

    scene  = slide.get_scene(0)

    img_pixel_size = read_pixel_size(image_path)

    if img_pixel_size > 1 or img_pixel_size < 0.1:
        import warnings
        warnings.warn("The image pixel size {} is not in microns. Please check the slideio documentation to convert the pixel size to microns.".format(img_pixel_size))
        if pixel_size is not None:
            img_pixel_size = pixel_size
        else:
            raise ValueError("The image pixel size {} is not in microns. Please check the slideio documentation to convert the pixel size to microns.".format(img_pixel_size))
    
    scale_factor = model_pixel_size/img_pixel_size

    dims = scene.rect[2:]
    dims = (int(dims[1]/ scale_factor), int(dims[0]/scale_factor)) #The dimensions are opposite to numpy/torch/zarr dimensions.

    pad2 = overlap + cell_size
    pad = overlap
    
    chop_list = chops(dims, shape, overlap=2*pad2)

    chunk_shape = (n_dim,shape[0],shape[1])
    store = zarr.DirectoryStore(file_with_zarr_extension) 
    canvas = zarr.zeros((n_dim,dims[0],dims[1]), chunks=chunk_shape, dtype=np.int32, store=store, overwrite = True)

    Augmenter=Augmentations()

    running_max = 0

    total = len(chop_list[0]) * len(chop_list[1])
    for _, ((i, window_i), (j, window_j)) in tqdm(enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))), total=total):
        
        input_data = scene.read_block((int(window_j*scale_factor), int(window_i*scale_factor), int(shape[0]*scale_factor), int(shape[1]*scale_factor)), size = shape)

        if input_data.mean() > threshold: #image is too bright corresponds to white space
            continue
        
        
        input_tensor,_ =Augmenter.to_tensor(input_data,normalize=False) #this converts the input data to a tensor
        input_tensor = input_tensor[None]/255
        

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if not use_torchscript:
                    new_tile = method.postprocessing(model(input_tensor.to(device))[0], max_seeds = 100000).to("cpu")
                else:
                    new_tile = instanseg(input_tensor.to(device))[0].to("cpu")


        num_iter = new_tile.shape[0]

        for n in range(num_iter):
            
            ignore_list = []
            if i == 0:
                ignore_list.append("top")
            if j == 0:
                ignore_list.append("left")
            if i == len(chop_list[0])-1:
                ignore_list.append("bottom")
            if j == len(chop_list[1])-1:
                ignore_list.append("right")

            if i == len(chop_list[0])-1 and j == len(chop_list[1])-1:
                tile1 = canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n,pad:shape[0],pad: shape[1]], ignore = ignore_list)

            elif i == len(chop_list[0])-1:
                tile1 = canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n,pad:shape[0],pad : shape[1]], ignore =ignore_list)

            elif j == len(chop_list[1])-1:
                tile1 = canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n,pad:shape[0],pad: shape[1]], ignore = ignore_list)

            elif i == 0 and j == 0:
                tile1 = canvas[n, window_i  :window_i + shape[0], window_j :window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n, :shape[0], : shape[1]], ignore = ignore_list)
            elif i == 0:
                tile1 = canvas[n, window_i  :window_i + shape[0], window_j + pad :window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n, :shape[0],pad : shape[1]], ignore = ignore_list)

            elif j == 0:
                tile1 = canvas[n, window_i  + pad:window_i + shape[0], window_j:window_j + shape[1]]
                tile2 = remove_edge_labels(new_tile[n,pad :shape[0],: shape[1]], ignore = ignore_list)

            if j == 0 or i == 0 or j == len(chop_list[1])-1 or i == len(chop_list[0])-1:

                tile2 = torch_fastremap(tile2)
                tile2[tile2>0] = tile2[tile2>0] + running_max

                tile1_torch = torch.tensor(np.array(tile1), dtype = torch.int32)

                remapped = match_labels(tile1_torch, tile2, threshold = 0.1)[1]
                tile1_torch[remapped>0] = remapped[remapped>0].int()

                running_max = max(running_max, tile1_torch.max())

                if i == len(chop_list[0])-1 and j == len(chop_list[1])-1:
                    canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)
                elif i == len(chop_list[0])-1:
                    canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)
                elif j == len(chop_list[1])-1:
                    canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)
                elif i == 0 and j == 0:
                    canvas[n, window_i  :window_i + shape[0], window_j :window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)
                elif i == 0:
                    canvas[n, window_i  :window_i + shape[0], window_j + pad :window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)
                elif j == 0:
                    canvas[n, window_i  + pad:window_i + shape[0], window_j:window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)

            else:
                
                tile1 = canvas[n, window_i + pad:window_i + shape[0] - pad, window_j + pad:window_j + shape[1] - pad]
                tile2 = remove_edge_labels(new_tile[n,pad:shape[0] -pad,pad: shape[1]-pad])
                
                tile2 = torch_fastremap(tile2)

                tile2[tile2>0] = tile2[tile2>0] + running_max

                tile1_torch = torch.tensor(np.array(tile1), dtype = torch.int32)
                remapped = match_labels(tile1_torch, tile2, threshold = 0.1)[1]

                tile1_torch[remapped>0] = remapped[remapped>0].int()
                running_max = max(running_max, tile1_torch.max())
            
                canvas[n, window_i + pad:window_i + shape[0] - pad, window_j + pad:window_j + shape[1] - pad] = tile1_torch.numpy().astype(np.int32)

    if to_geojson:
        print("Exporting to geojson")
        zarr_to_json_export(file_with_zarr_extension, cell_size = cell_size, size = shape[0], scale = scale_factor, n_dim = n_dim)
            

    return canvas


def instanseg_padding(img: torch.Tensor, extra_pad: int = 0, min_dim: int = 16, ensure_square: bool = False):

    is_square = img.shape[-2] == img.shape[-1]
    original_shape = img.shape[-2:]
    bigger_dim = max(img.shape[-2], img.shape[-1])

    if ensure_square and not is_square:
        img = torch.functional.F.pad(img, [0, bigger_dim - img.shape[-1], 0, bigger_dim - img.shape[-2]], mode='constant')

    padx = min_dim * torch.ceil(torch.tensor((img.shape[-2] / min_dim))).int() - img.shape[-2] + extra_pad * 2
    pady = min_dim * torch.ceil(torch.tensor((img.shape[-1] / min_dim))).int() - img.shape[-1] + extra_pad * 2

    if padx > img.shape[-2]:
        padx = padx - extra_pad
    if pady > img.shape[-1]:
        pady = pady - extra_pad
    img = torch.functional.F.pad(img, [0, int(pady), 0, int(padx)], mode='reflect')

    if ensure_square and not is_square:
        pady = pady + bigger_dim - original_shape[-1]
        padx = padx + bigger_dim - img.shape[-2]
        print(padx, pady)
    

    return img, torch.stack((padx, pady))


def recover_padding(x: torch.Tensor, pad: torch.Tensor):
    # x must be 1,C,H,W or C,H,W
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    if pad[0] == 0:
        pad[0] = -x.shape[2]
    if pad[1] == 0:
        pad[1] = -x.shape[3]

    if squeeze:
        return x[:, :, :-pad[0], :-pad[1]].squeeze(0)
    else:
        return x[:, :, :-pad[0], :-pad[1]]


from InstanSeg.utils.utils import timer
from tqdm import tqdm


#@timer
def sliding_window_inference(input_tensor, predictor, window_size=(512, 512), overlap = 100, max_cell_size = 20, sw_device='cuda',
                             device='cpu', output_channels=1, show_progress = True, **kwargs):
    input_tensor = input_tensor.to(device)
    predictor = predictor.to(sw_device)

    tuple_index = chops(input_tensor.shape, shape=window_size, overlap=2 * (overlap + max_cell_size))
    tile_list = tiles_from_chops(input_tensor, shape=window_size, tuple_index=tuple_index)

    
        
    #print("Number of tiles: ", len(tile_list))
   # print("Shape of tiles: ", tile_list[0].shape)


    with torch.no_grad():
        with torch.cuda.amp.autocast():
            label_list = [predictor(tile[None].to(sw_device),**kwargs).squeeze(0).to(device) for tile in tqdm(tile_list, disable= not show_progress)]

    
    lab = torch.cat([stitch([lab[i] for lab in label_list],
                            shape=window_size,
                            chop_list=tuple_index,
                            offset = overlap,
                            final_shape=(1, input_tensor.shape[1], input_tensor.shape[2])) 
                            
                    for i in range(output_channels)], dim=0)
    

    return lab[None]  # 1,C,H,W


if __name__ == "__main__":

    import torch
    from InstanSeg.utils.tiling import sliding_window_inference
    from InstanSeg.utils.augmentations import Augmentations
    from InstanSeg import show_images
    Augmenter=Augmentations()

    instanseg = torch.jit.load("../torchscripts/1827669.pt")

    import tifffile as tifffile

    if "arr" not in locals():
        arr = tifffile.imread("../../Datasets/Unannotated/Tonsil/tonsil.ome.tif")

    if "input_tensor" not in locals():
        input_tensor = Augmenter.to_tensor(arr,normalize=False)[0]
        input_tensor = Augmenter.normalize(input_tensor, percentile=0.01, subsampling_factor=10)[0]

    #if "lab" not in locals():
    lab = sliding_window_inference(input_tensor[:,:700,:700], instanseg,window_size= (512,512), sw_device= "cuda", overlap= 50, output_channels= 2)



    pdb.set_trace()

    from tqdm import tqdm
    import sys

    from InstanSeg.utils.utils import show_images, _choose_device
    import torch

    from skimage import io

    import time

    instanseg = torch.jit.load("../examples/torchscript_models/instanseg_experiment.pt")
    device = 'cpu'
    instanseg.to(device)

  #  input_data=io.imread("../examples/LuCa1.tif")
    input_data = io.imread("../examples/HE_example.tif")
    from InstanSeg.utils.augmentations import Augmentations

    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=True)
    # out=instanseg(input_tensor[None,].to(device)[:,:,:500,:1000])

    input_tensor = input_tensor[:, 0:512, 0:512]

    rgb = input_tensor #Augmenter.colourize(input_tensor, c_nuclei=6, metadata={"image_modality": "Fluorescence"})[0]

    device = "cpu"
    instanseg.to(device)
    gt = instanseg(input_tensor[None].to(device))[0, 0].cpu()

    device = _choose_device()
    instanseg.to(device)

    start = time.time()

    lab = sliding_window_inference(input_tensor, instanseg, window_size=(256, 256), overlap_size= 32 / 256,
                                   sw_device=device, device='cpu', output_channels=1)
    
    
    la2 = sliding_window_inference(input_tensor,instanseg, window_size = (256,256),overlap_size = 128/1024,sw_device = device,device = 'cpu', output_channels = 1)

    show_images(lab, gt, (lab > 0).int() - (gt > 0).int(), la2, gt, (la2 > 0).int() - (gt > 0).int(),
                labels=[0, 1, 3, 4], titles=["Tiled", "One Pass", "Difference", "Tiled", "One Pass", "Difference"],
                n_cols=3)

    end = time.time()

    print("Time for dual channel output sliding window: ", end - start)

    start = time.time()

    tuple_index = chops(input_tensor, shape=(256, 256), overlap=0.2)
    tile_list = tiles_from_chops(input_tensor, shape=(256, 256), tuple_index=tuple_index)
    label_list = [instanseg(tile[None].to(device))[0, 0].cpu() for tile in tile_list]
    lab = stitch(label_list, shape=(256, 256), chop_list=tuple_index,
                 final_shape=(1, input_tensor.shape[1], input_tensor.shape[2]))

    end = time.time()

    print("Time for 256x256 tiling: ", end - start)

    start = time.time()

    tuple_index = chops(input_tensor, shape=(512, 512), overlap=0.2)
    tile_list = tiles_from_chops(input_tensor, shape=(512, 512), tuple_index=tuple_index)
    label_list = [instanseg(tile[None].to(device))[0, 0].cpu() for tile in tile_list]
    lab2 = stitch(label_list, shape=(512, 512), chop_list=tuple_index,
                  final_shape=(1, input_tensor.shape[1], input_tensor.shape[2]))

    end = time.time()
    print("Time for 512x512 tiling: ", end - start)

    show_images(lab, gt, (lab > 0).int() - (gt > 0).int(), labels=[0, 1], titles=["Tiled", "One Pass", "Difference"])
    show_images(lab, lab2, (lab > 0).int() - (lab2 > 0).int(), rgb, labels=[0, 1],
                titles=["Tile 256", "Tile 512", "Difference", "original"])

    start = time.time()

    lab = sliding_window_inference(input_tensor, instanseg, window_size=(512, 512), overlap_size=0.2, sw_device='cpu',
                                   device='cpu', output_channels=1)

    end = time.time()

    print("Time for dual channel output sliding window: ", end - start)

# show_images([i for i in lab]+[i for i in lab],labels=[0,1])


# pdb.set_trace()
