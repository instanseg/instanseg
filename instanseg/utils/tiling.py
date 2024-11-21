import torch
import numpy as np
from tqdm import tqdm

def _edge_mask(labels, ignore=[None]):
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

    if len(edges) == 0:
        return torch.zeros_like(labels).bool()

    edges = torch.cat(edges, dim=0)
    return torch.isin(labels, edges[edges > 0])


def _remove_edge_labels(labels, ignore=[None]):
    return labels * ~_edge_mask(labels, ignore=ignore)



def _chops(img_shape: tuple, shape: tuple, overlap: int = 0) -> tuple:
    """This function splits an image into desired windows and returns the indices of the windows"""

    if (torch.tensor(img_shape[-2:]) < torch.tensor(shape)).any():
        return [0], [0]
    h, v = img_shape[-2:]

    assert shape[0] > overlap and shape[1] > overlap, f"The overlap {overlap} must be smaller than the window size {shape}"

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


def _tiles_from_chops(image: torch.Tensor, shape: tuple, tuple_index: tuple) -> list:
    """This function takes an image, a shape, and a tuple of window indices (e.g., output of the function _chops)
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


def _stitch(tiles: list, shape: tuple, chop_list: list, final_shape: tuple, offset : int):
    """This function takes a list of tiles, a shape, and a tuple of window indices (e.g., outputed by the function _chops)
    and returns a stitched image"""
    from instanseg.utils.pytorch_utils import torch_fastremap, match_labels

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
                tile2 = _remove_edge_labels(new_tile[:,offset:shape[0],offset: shape[1]], ignore = ignore_list)

            elif i == len(chop_list[0])-1:
                edge_window = True
                tile1 = canvas[..., window_i + offset:window_i + shape[0], window_j + offset:window_j + shape[1]]
                tile2 = _remove_edge_labels(new_tile[:,offset:shape[0],offset : shape[1]], ignore =ignore_list)

            elif j == len(chop_list[1])-1:
                edge_window = True
                tile1 = canvas[..., window_i + offset:window_i + shape[0], window_j + offset:window_j + shape[1]]
                tile2 = _remove_edge_labels(new_tile[:,offset:shape[0],offset: shape[1]], ignore = ignore_list)

            if i == 0 and j == 0:
                edge_window = True
                tile1 = canvas[..., window_i  :window_i + shape[0], window_j :window_j + shape[1]]
                tile2 = _remove_edge_labels(new_tile[:, :shape[0], : shape[1]], ignore = ignore_list)
            elif i == 0:
                edge_window = True
                tile1 = canvas[..., window_i  :window_i + shape[0], window_j + offset :window_j + shape[1]]
                tile2 = _remove_edge_labels(new_tile[:, :shape[0],offset : shape[1]], ignore = ignore_list)

            elif j == 0:
                edge_window = True
                tile1 = canvas[..., window_i  + offset:window_i + shape[0], window_j:window_j + shape[1]]
                tile2 = _remove_edge_labels(new_tile[:,offset :shape[0],: shape[1]], ignore = ignore_list)

            if edge_window:

                tile2 = torch_fastremap(tile2)
                tile2[tile2>0] = tile2[tile2>0] + running_max

                remapped = match_labels(tile1, tile2, threshold = 0.1)[1]
                tile1[remapped>0] = remapped[remapped>0].int()

                running_max = max(running_max, tile1.max())

            else:
                
                tile1 = canvas[..., window_i + offset:window_i + shape[0] - offset, window_j + offset:window_j + shape[1] - offset]
                tile2 = _remove_edge_labels(new_tile[:,offset:shape[0] -offset,offset: shape[1]-offset])


                tile2 = torch_fastremap(tile2)
                tile2[tile2>0] = tile2[tile2>0] + running_max

                remapped = match_labels(tile1, tile2, threshold = 0.1)[1]
                tile1[remapped>0] = remapped[remapped>0].int()

                running_max = max(running_max, tile1.max())
            

    return canvas


def _zarr_to_json_export(path_to_zarr, detection_size = 30, size = 1024, scale = 1, n_dim = 1):

    import zarr
    import numpy as np
    import os

    z = zarr.open(path_to_zarr, mode='r')

    output_path = str(path_to_zarr).replace(".zarr",".geojson")   


    from instanseg.utils.tiling import _chops

    chop_list = _chops(z.shape, shape=(size,size), overlap=detection_size)

    from instanseg.utils.utils import labels_to_features
    from instanseg.utils.tiling import _remove_edge_labels
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
              
                image = _remove_edge_labels(torch.tensor(image)).numpy()

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


def _instanseg_padding(img: torch.Tensor, extra_pad: int = 0, min_dim: int = 16, ensure_square: bool = False):

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


def _recover_padding(x: torch.Tensor, pad: torch.Tensor):
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



def _sliding_window_inference(input_tensor, 
                             predictor, 
                             window_size=(512, 512), 
                             overlap = 80, 
                             max_cell_size = 20, 
                             sw_device='cuda',
                             device='cpu', 
                             output_channels=1, 
                             show_progress = True, 
                             batch_size = 1,
                             **kwargs):
    
    h,w = input_tensor.shape[-2:]
    window_size = (min(window_size[0], h), min(window_size[1], w))
    
    input_tensor = input_tensor.to(device)
    predictor = predictor.to(sw_device)
 
    tuple_index = _chops(input_tensor.shape, shape=window_size, overlap=2 * (overlap + max_cell_size))
    tile_list = _tiles_from_chops(input_tensor, shape=window_size, tuple_index=tuple_index)
 
 
    assert len(tile_list) > 0, "No tiles generated"
    # print("Number of tiles: ", len(tile_list))
    # print("Shape of tiles: ", tile_list[0].shape)
    # print("window size: ", window_size)
    # print("input tensor shape: ", input_tensor.shape)
 
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            batch_list = [torch.stack(tile_list[batch_size * i:batch_size * (i+1)]) for i in range(int(np.ceil(len(tile_list)/batch_size)))]
            label_list = torch.cat([predictor(tile.to(sw_device),**kwargs).to(device) for tile in tqdm(batch_list, disable= not show_progress,leave = False, colour = "blue")])
 
   
    lab = torch.cat([_stitch([lab[i] for lab in label_list],
                            shape=window_size,
                            chop_list=tuple_index,
                            offset = overlap,
                            final_shape=(1, input_tensor.shape[1], input_tensor.shape[2])) 
 
                    for i in range(output_channels)], dim=0)
 
 
    return lab[None]  # 1,C,H,W

