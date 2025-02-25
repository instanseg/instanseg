import os
import pandas as pd
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse
import pdb

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
parser.add_argument("-batch_size", "--batch_size", type=int, default= 3, help="batch size, only useful for large images")
parser.add_argument("-save_geojson", "--save_geojson", type=lambda x: (str(x).lower() == 'true'), default= False, help="Output geojson files of the segmentation")

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
    from instanseg.utils.utils import show_images
    from instanseg import InstanSeg

    parser = parser.parse_args()

    if parser.image_path is None or not os.path.exists(parser.image_path):
        from instanseg.utils.utils import drag_and_drop_file
        parser.image_path = drag_and_drop_file()
        print("Using image path: ", parser.image_path)

    if parser.model_folder is None:
        raise ValueError("Please provide a model name")
    
    instanseg = InstanSeg(model_type=parser.model_folder, device= parser.device)
    instanseg.prediction_tag = prediction_tag

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

    
    for file in tqdm(files):

        print("Processing: ", file)

        _ = instanseg.eval(image=file,
                        pixel_size = parser.pixel_size,
                        save_output = True,
                        save_overlay = True,
                        save_geojson = parser.save_geojson,
                        batch_size = parser.batch_size,
                        tile_size = parser.tile_size,)












