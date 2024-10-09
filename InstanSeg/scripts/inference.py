import os
import pandas as pd
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse
import numpy as np
import pdb
from skimage import io
from aicsimageio import AICSImage
import warnings
import torchvision
from torchvision.transforms import InterpolationMode


parser = argparse.ArgumentParser()
parser.add_argument("-i_p", "--image_path", type=str, default=r"../examples")
parser.add_argument("-m_f", "--model_folder", type=str)
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("-exclude", "--exclude_str", type=str, default= ["mask","prediction", "geojson", "zip"], help="Exclude files with this string in their name")
parser.add_argument("-pixel_size", "--pixel_size", type=float, default= None, help="Pixel size of the input image in microns")
parser.add_argument("-recursive", "--recursive",default=False, type=lambda x: (str(x).lower() == 'true'),help="Look for images recursively at the image path")
parser.add_argument("-ignore_segmented", "--ignore_segmented",default=False, type=lambda x: (str(x).lower() == 'true'),help="Whether to ignore previously segmented images in the image path")

#advanced usage
parser.add_argument("-tile_size", "--tile_size", type=int, default= 512, help="tile size in pixels given to the model, only used for large images.")
parser.add_argument("-ram_tile_size", "--ram_tile_size", type=int, default= 5000, help="tile size in pixels to be read in ram, only useful for WSIs")

parser.add_argument("-batch_size", "--batch_size", type=int, default= 3, help="batch size, only useful for large images")
parser.add_argument("-output_geojson", "--output_geojson", type=lambda x: (str(x).lower() == 'true'), default= False, help="Output geojson files of the segmentation")
parser.add_argument("-overlap", "--overlap_size",default=60, type=int,help="Overlap size in pixels for the sliding window inference")
parser.add_argument("-max_cell_size", "--max_cell_size",default=30, type=int,help="Diameter (in pixels at the model's pixel size) of the largest cell in the image. Used for tiling.")

def file_matches_requirement(root,file, exclude_str):
    if not os.path.isfile(os.path.join(root,file)):
        return False
    for e_str in exclude_str:
        if e_str in file:
            return False
        if parser.ignore_segmented:
            if os.path.isfile(os.path.join(root,str(Path(file).stem) + prediction_tag + ".tiff")):
                return False
    return True

prediction_tag = "_instanseg_prediction"


if __name__ == "__main__":
    from InstanSeg.utils.utils import show_images, save_image_with_label_overlay, _choose_device
    from InstanSeg.utils.model_loader import load_model
    from InstanSeg.utils.utils import export_to_torchscript
    from InstanSeg.utils.augmentations import Augmentations


    parser = parser.parse_args()

    if parser.image_path is None or not os.path.exists(parser.image_path):
        from InstanSeg.utils.utils import drag_and_drop_file
        parser.image_path = drag_and_drop_file()
        print("Using image path: ", parser.image_path)


    if parser.model_folder is None:
        raise ValueError("Please provide a model name")

    device = _choose_device(parser.device)

    if parser.model_folder in ["brightfield_nuclei","fluorescence_nuclei_and_cells"]:
        from InstanSeg.utils.utils import download_model
        instanseg = download_model(parser.model_folder, return_model=True)
        output_dimension =  2 if instanseg.cells_and_nuclei else 1
    else:
        if not os.path.exists("../torchscripts/{}.pt".format(parser.model_folder)):
            print("Exporting model to torchscript")
            export_to_torchscript(parser.model_folder)
        instanseg = torch.jit.load("../torchscripts/" + parser.model_folder + ".pt").to(device)
        output_dimension =  2 if instanseg.cells_and_nuclei else 1

    if not parser.recursive:
        print("Loading files from: ", parser.image_path)
        files = os.listdir(parser.image_path)
        files = [os.path.join(parser.image_path, file) for file in files if file_matches_requirement(parser.image_path, file, parser.exclude_str)]
    else:
        print("Loading files recursively from: ", parser.image_path)
        files = []
        for root, dirs, filenames in os.walk(parser.image_path):
            for filename in filenames:
                if file_matches_requirement(root , filename, parser.exclude_str):
                    files.append(os.path.join(root, filename))


    assert len(files) > 0, "No files found in the specified directory"

    if __name__ == "__main__":
        augmenter = Augmentations()
        for file in tqdm(files):
            stem = Path(file).stem

            print("Processing file: ", file)

            img = AICSImage(file)
            if parser.pixel_size is None and img.physical_pixel_sizes.X is None:

                warnings.warn("Pixel size was not found in the metadata, please set the pixel size of the input image in microns manually")
            elif parser.pixel_size is None and img.physical_pixel_sizes.X is not None:
                pixel_size = img.physical_pixel_sizes.X
                if pixel_size < 0.1 or pixel_size > 0.9:
                    warnings.warn("Pixel size {} doesn't seem to be in microns, - ignoring the metadata pixel size".format(pixel_size))
                    pixel_size = parser.pixel_size
            else:
                pixel_size = parser.pixel_size

            channel_number = img.dims.C
            num_pixels = np.cumprod(img.shape)[-1]

            if num_pixels < 3 * 15000 * 15000:       
                if "S" in img.dims.order and img.dims.S > img.dims.C:
                    channel_number = img.dims.S
                    input_data = img.get_image_data("SYX")
                else:
                    input_data = img.get_image_data("CYX")

                input_tensor = augmenter.to_tensor(input_data, normalize=True)[0].to(device)
                original_shape = input_tensor.shape[1:]

                import math
                if pixel_size is not None and not math.isnan(instanseg.pixel_size):
                   # print("Rescaling image {} to match the model's pixel {} size".format(pixel_size, instanseg.pixel_size))
                    input_tensor,_ = augmenter.torch_rescale(input_tensor,labels=None,current_pixel_size=pixel_size,requested_pixel_size=instanseg.pixel_size,crop = False)


                if num_pixels > 3 * 1500 * 1500:
                    from InstanSeg.utils.tiling import sliding_window_inference
                    labels = sliding_window_inference(input_tensor,
                                instanseg, 
                                window_size = (parser.tile_size,parser.tile_size),
                                overlap= parser.overlap_size, 
                                max_cell_size= parser.max_cell_size,
                                sw_device = device,
                                device = 'cpu', 
                                batch_size= parser.batch_size,
                                output_channels = output_dimension,
                                resolve_cell_and_nucleus = False)
                    
                    

                else:
                    with torch.cuda.amp.autocast():
                        labels = instanseg(input_tensor[None])

            else:
                print("Image {} is too large, attempting using a zarr array".format(stem))
                from InstanSeg.utils.tiling import segment_image_larger_than_memory
                                 

                segment_image_larger_than_memory(instanseg_folder= parser.model_folder, 
                                                image_path= file, 
                                                memory_block_size = (parser.ram_tile_size,parser.ram_tile_size),  #this is the size of the image that will be read in memory
                                                inference_tile_size = (parser.tile_size,parser.tile_size), #this is the size of the image that will be passed to the model
                                                threshold= 255, 
                                                cell_size = parser.max_cell_size, 
                                                overlap= parser.overlap_size,
                                                batch_size= parser.batch_size,
                                                to_geojson= parser.output_geojson, 
                                                torchscript = instanseg,
                                                pixel_size = parser.pixel_size,
                                                prediction_tag = prediction_tag)

                continue

            labels = torchvision.transforms.Resize(original_shape,interpolation = InterpolationMode.NEAREST)(labels)

            labels = labels.cpu().detach().numpy()

            new_stem = stem + prediction_tag

            io.imsave(Path(file).parent / (new_stem + ".tiff"), labels.squeeze().astype(np.int32), check_contrast=False)

            if parser.output_geojson:
                from InstanSeg.utils.utils import labels_to_features
                import json
                if output_dimension == 1:
                    features = labels_to_features(labels[0,0],object_type = "detection")

                elif output_dimension == 2:
                    features = labels_to_features(labels[0,0],object_type = "detection",classification="Nuclei") + labels_to_features(labels[0,1],object_type = "detection",classification = "Cells")
                geojson = json.dumps(features)

                geojson_path = Path(file).parent / (new_stem + ".geojson")
                with open(os.path.join(geojson_path), "w") as outfile:
                    outfile.write(geojson)









