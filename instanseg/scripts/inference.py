import os
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse

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
parser.add_argument("-image_reader", "--image_reader", type=str, default= "auto", help='The image reader to use. Options are "auto", "tiffslide", "skimage.io", "bioio", "AICSImageIO". If "auto", will use the first available reader.')
parser.add_argument("-use_otsu", "--use_otsu_threshold", type=lambda x: (str(x).lower() == 'true'), default= True, help="Use an Otsu Threshold on the WSI thumbnail to determine which channels to segment(ignored for images that are not WSIs)")

parser.add_argument("-kwargs", "--kwargs", nargs="*", type=str, default={}, help="Additional keyword arguments in the form key=value", dest="kwargs_raw")

def smart_cast(value):
    """Try to convert string to int, float, bool, or leave as string."""
    value_lower = value.lower()
    if value[0] == "[" and value[-1] == "]":
        value = value.replace("[","").replace("]","").split(",")
        value = [smart_cast(v) for v in value]
        return value
    if value_lower == "true":
        return True
    elif value_lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def parse_key_value(arg_list):
    kwargs = {}
    for arg in arg_list:
        if "=" not in arg:
            raise argparse.ArgumentTypeError(f"Invalid format for argument '{arg}'. Use key=value.")
        key, value = arg.split("=", 1)
        kwargs[key] = smart_cast(value)
    return kwargs


def file_matches_requirement(root,file, exclude_str):
    if not os.path.isfile(os.path.join(root,file)):
        return False
    for e_str in exclude_str:
        if e_str in file:
            return False
        if parser.ignore_segmented:
            for extension in [".tiff",".zarr"]:
                if os.path.exists(os.path.join(root,str(Path(file).stem) + prediction_tag + extension)):
                    return False
    return True

prediction_tag = "_instanseg_prediction"

def main():
    global parser
    from instanseg import InstanSeg

    parser = parser.parse_args()

# Convert the list of key=value strings into a dictionary
    kwargs = parse_key_value(parser.kwargs_raw)
    del parser.kwargs_raw


    if parser.image_path is None or not os.path.exists(parser.image_path):
        from instanseg.utils.utils import drag_and_drop_file
        parser.image_path = drag_and_drop_file()
        print("Using image path: ", parser.image_path)

    if parser.model_folder is None:
        raise ValueError("Please provide a model name")
    
    instanseg = InstanSeg(model_type=parser.model_folder, device= parser.device, image_reader= parser.image_reader)
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
                        tile_size = parser.tile_size,use_otsu_threshold = parser.use_otsu_threshold,
                        **kwargs,
                        )


if __name__ == "__main__":
    main()









