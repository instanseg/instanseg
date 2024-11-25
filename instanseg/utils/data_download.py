
from pathlib import Path
from instanseg.utils.utils import _move_channel_axis
import fastremap
from tqdm import tqdm
from skimage import io
from pathlib import Path
from scipy import ndimage
import skimage
import numpy as np
import os

import requests
import zipfile


def get_raw_datasets_dir(*others) -> Path:

    if os.environ.get('INSTANSEG_RAW_DATASETS'):
        path = Path(os.environ['INSTANSEG_RAW_DATASETS'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../../Raw_Datasets/"))
        os.environ['INSTANSEG_RAW_DATASETS'] = str(path)
    
    if others:
        path = path.joinpath(*others)

    return path

def get_processed_datasets_dir(*others) -> Path:

    if os.environ.get('INSTANSEG_DATASET_PATH'):
        path = Path(os.environ['INSTANSEG_DATASET_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../datasets/"))
        os.environ['INSTANSEG_DATASET_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if others:
        path = path.joinpath(*others)

    return path
    

def create_raw_datasets_dir(*others) -> Path:
    path = get_raw_datasets_dir(*others)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def create_processed_datasets_dir(*others) -> Path:
    path = get_processed_datasets_dir(*others)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

    
def load_Cellpose(Segmentation_Dataset: dict, verbose: bool = True) -> dict:

    from instanseg.utils.augmentations import Augmentations
    from instanseg.utils.pytorch_utils import torch_fastremap

    cellpose_dir = create_raw_datasets_dir("Cell_Segmentation", "Cellpose")

    # Process Train Data
    train_file_path = Path(cellpose_dir) / "train"
    train_folders = sorted(list(train_file_path.iterdir()))
    train_items = []

    for file in tqdm(train_folders):
        if "masks" not in file.name:
            item = {}
            image = io.imread(file)
            labels = io.imread(str(file).replace("_img", "_masks"))
            _, area = np.unique(labels[labels > 0], return_counts=True)
            augmenter = Augmentations(shape=(2, 2))
            input_tensor, labels = augmenter.to_tensor(image, labels, normalize=False)
            tensor, labels = augmenter.torch_rescale(input_tensor, labels, current_pixel_size=0.5 / ((np.median(area) ** 0.5) / (300 ** 0.5)), requested_pixel_size=0.5, crop=False)

            item['cell_masks'] = torch_fastremap(labels).squeeze()
            item['image'] = fastremap.refit(np.array(tensor.byte()))
            item["parent_dataset"] = "cellpose"
            item['licence'] = "Non-Commercial"
            item['image_modality'] = "Fluorescence"
            item['file_name'] = file.name
            item['original_size'] = image.shape

            train_items.append(item)

    np.random.seed(42)
    np.random.shuffle(train_items)
    Segmentation_Dataset['Train'] += train_items[:int(len(train_items) * 0.8)]
    Segmentation_Dataset['Validation'] += train_items[int(len(train_items) * 0.8):]
    Segmentation_Dataset['Test'] += train_items[int(len(train_items) * 0.9):]

    # Process Test Data
    test_file_path = Path(cellpose_dir) / "test"
    test_folders = sorted(list(test_file_path.iterdir()))
    test_items = []

    for file in tqdm(test_folders):
        if "masks" not in file.name:
            item = {}
            image = io.imread(file)
            labels = io.imread(str(file).replace("_img", "_masks"))
            _, area = np.unique(labels[labels > 0], return_counts=True)
            augmenter = Augmentations(shape=(2, 2))
            input_tensor, labels = augmenter.to_tensor(image, labels, normalize=False)
            tensor, labels = augmenter.torch_rescale(input_tensor, labels, current_pixel_size=0.5 / ((np.median(area) ** 0.5) / (300 ** 0.5)), requested_pixel_size=0.5, crop=False)
            item['cell_masks'] = torch_fastremap(labels).squeeze()
            item['image'] = fastremap.refit(np.array(tensor.byte()))
            item["parent_dataset"] = "cellpose"
            item['licence'] = "Non-Commercial"
            item['image_modality'] = "Fluorescence"
            item['file_name'] = file.name
            item['original_size'] = image.shape

            test_items.append(item)


    Segmentation_Dataset['Test'] += test_items

    return Segmentation_Dataset

def load_TNBC_2018(Segmentation_Dataset: dict, verbose: bool = True) -> dict:

    tnbc_dir = create_raw_datasets_dir("Nucleus_Segmentation", "TNBC_NucleiSegmentation")
    zip_file_path = tnbc_dir / "TNBC_NucleiSegmentation.zip"
    download_url = "https://zenodo.org/record/3552674/files/TNBC_and_Brain_dataset.zip?download=1"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {tnbc_dir}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tnbc_dir)
        if verbose:
            print("Unzipping completed.")

    file_path = tnbc_dir / "TNBC_and_Brain_dataset"
    folders = sorted(list(file_path.iterdir()))

    items=[]

    for i,folder in enumerate(tqdm(folders)):
        if "Slide_" in str(folder):
            for file in sorted(os.listdir(folder)):
                file = (folder / file)
                if ".DS_Store" not in str(file):
                    item={}
                    image = io.imread(str(file))
                    if _move_channel_axis(image).shape[0] == 4:  #Remove alpha channel
                        image = _move_channel_axis(image)[:3]

                    masks = io.imread(str(file).replace("Slide_","GT_"))
                    lab, n_labels = ndimage.label(masks > 0)
                    labels, remapping = fastremap.renumber(skimage.morphology.label(lab), in_place=True)

                    item['nucleus_masks']=fastremap.refit(labels)
                    item['image']=image
                    item["parent_dataset"]="TNBC_2018"
                    item['licence']="CC BY 4.0"
                    item['pixel_size']=0.25    #In microns per pixel
                    item['image_modality']="Brightfield"
                    item['stain']="H&E"
                    items.append(item)
                    

    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train']+=items[:int(len(items)*0.8)]
    Segmentation_Dataset['Validation']+=items[int(len(items)*0.8):int(len(items)*0.9)]
    Segmentation_Dataset['Test']+=items[int(len(items)*0.9):]

    return Segmentation_Dataset

def load_LyNSeC(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    lynsec_dir = create_raw_datasets_dir("Nucleus_Segmentation", "LyNSeC")
    zip_file_path = lynsec_dir / "LyNSeC.zip"
    download_url = "https://zenodo.org/record/8065174/files/lynsec.zip?download=1"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {lynsec_dir / 'LyNSeC'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(lynsec_dir / 'LyNSeC')
        if verbose:
            print("Unzipping completed.")

    file_path = Path(zip_file_path).parent / "LyNSeC"
    folders = sorted(list(file_path.iterdir()))

    items = []

    for i, folder in enumerate(tqdm(folders)):
        if folder.stem == "lynsec 2":  # Skip incorrectly annotated folder
            continue


        for j, file in enumerate(sorted(os.listdir(folder))):
            file_ = folder / file
            if ".DS_Store" not in str(file_):
                item = {}
                data = np.load(str(file_))
                image = data[:, :, :3].astype(np.uint8)
                masks = data[:, :, 3].copy()
                masks, remapping = fastremap.renumber(masks, in_place=True)
                item['nucleus_masks'] = fastremap.refit(masks)
                item['image'] = image
                item["parent_dataset"] = "LyNSeC"
                item['licence'] = "CC BY 4.0"
                item['pixel_size'] = 0.25  # In microns per pixel
                item['image_modality'] = "Brightfield"
                item['stain'] = "H&E" if i > 1 else "IHC"
                items.append(item)

    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):int(len(items) * 0.9)]
    Segmentation_Dataset['Test'] += items[int(len(items) * 0.9):]

    return Segmentation_Dataset

def load_NuInsSeg(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    nuinsseg_dir = create_raw_datasets_dir("Nucleus_Segmentation", "NuInsSeg")
    zip_file_path = nuinsseg_dir / "NuInsSeg.zip"
    download_url = "https://zenodo.org/record/10518968/files/NuInsSeg.zip?download=1"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {nuinsseg_dir / 'NuInsSeg'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(nuinsseg_dir / 'NuInsSeg')
        if verbose:
            print("Unzipping completed.")


    file_path = nuinsseg_dir / "NuInsSeg"
    folders = sorted(list(Path(file_path).iterdir()))

    items=[]

    for i,folder in enumerate(tqdm(folders)):

        img_folder = folder / "tissue images"

        for j,file in enumerate(sorted(os.listdir(img_folder))):
            file = (img_folder / file)

            if ".DS_Store" not in str(file):
                item={}
                data = io.imread(str(file))
                image = data[:,:,:3].astype(np.uint8)

                masks = io.imread(str(file).replace("tissue images","label masks modify").replace(".png",".tif"))

                masks, remapping = fastremap.renumber(masks ,in_place=True)
                item['nucleus_masks']=fastremap.refit(masks)
                item['image']=image
                item["parent_dataset"]="NuInsSeg"
                item['licence']="CC BY 4.0"
                item['pixel_size']=0.25    #In microns per pixel
                item['image_modality']="Brightfield"
                item['stain']="H&E"
                item["tissue_type"]= folder.stem
                items.append(item)

    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train']+=items[:int(len(items)*0.8)]
    Segmentation_Dataset['Validation']+=items[int(len(items)*0.8):int(len(items)*0.9)]
    Segmentation_Dataset['Test']+=items[int(len(items)*0.9):]

    return Segmentation_Dataset
            

def load_IHC_TMA(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    ihc_tma_dir = create_raw_datasets_dir("Nucleus_Segmentation", "IHC_TMA_dataset")
    zip_file_path = ihc_tma_dir / "IHC_TMA_dataset.zip"
    download_url = "https://zenodo.org/record/7647846/files/IHC_TMA_dataset.zip?download=1"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {ihc_tma_dir}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(ihc_tma_dir)
        if verbose:
            print("Unzipping completed.")


    file_path = ihc_tma_dir / "IHC_TMA_dataset" / "images"
    files = sorted(list(file_path.iterdir()))

    
    def separate_touching_objects(lab):
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed
        from skimage.measure import label

        mask = lab > 0
        distance_map = distance_transform_edt(mask) 
        markers = distance_map > 2
        markers = label(markers)
        labels = watershed(-distance_map, markers, mask=mask)
        labels = labels + lab
        return labels


    items = []

    for i, file in enumerate(tqdm(files)):
        item = {}

        image = io.imread(str(file))
        masks = np.load(str(file).replace("images", "masks").replace(".png", ".npy"))

        n_masks = np.max(masks[0:2], axis=0)

        masks = separate_touching_objects(n_masks)
        masks, remapping = fastremap.renumber(masks, in_place=True)
        masks = fastremap.refit(masks)

        item['nucleus_masks'] = fastremap.refit(masks)
        item['image'] = image
        item["parent_dataset"] = "IHC_TMA"
        item['licence'] = "CC BY 4.0"
        item['pixel_size'] = 0.25  # In microns per pixel
        item['stain'] = "IHC"
        item['image_modality'] = "Brightfield"

        items.append(item)

    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):int(len(items) * 0.9)]
    Segmentation_Dataset['Test'] += items[int(len(items) * 0.9):]

    return Segmentation_Dataset

def load_CoNSeP(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    consep_dir = create_raw_datasets_dir("Nucleus_Segmentation", "CoNSeP")
    zip_file_path = consep_dir / "CoNSeP.zip"
    download_url = "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {consep_dir / 'CoNSeP'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(consep_dir / 'CoNSeP')
        if verbose:
            print("Unzipping completed.")

    # Process Training Data
    training_path = consep_dir / "CoNSeP" / "Training"
    training_files = sorted(list(training_path.iterdir()))

    items = []

    for i, file in enumerate(tqdm(training_files)):
        if ".DS_Store" not in str(file) and "_masks" not in str(file):
            item = {}
            img = io.imread(file)

            masks = io.imread(str(file.parent / (file.stem + "_masks.tif")))
            masks, remapping = fastremap.renumber(masks, in_place=True)
            masks = fastremap.refit(masks)

            item['nucleus_masks'] = masks
            item['image'] = img
            item["parent_dataset"] = "CoNSeP"
            item['image_modality'] = "Brightfield"
            item['stain'] = "H&E"
            item['licence'] = "Apache 2.0"
            item['pixel_size'] = 0.275
            items.append(item)

    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):]

    # Process Testing Data
    testing_path = consep_dir / "CoNSeP" / "Testing"
    testing_files = sorted(list(testing_path.iterdir()))

    items = []

    for i, file in enumerate(tqdm(testing_files)):
        if ".DS_Store" not in str(file) and "_masks" not in str(file):
            item = {}
            img = io.imread(file)
            masks = io.imread(str(file.parent / (file.stem + "_masks.tif")))
            masks, remapping = fastremap.renumber(masks, in_place=True)
            masks = fastremap.refit(masks)
            item['nucleus_masks'] = masks
            item['image'] = img
            item["parent_dataset"] = "CoNSeP"
            item['image_modality'] = "Brightfield"
            item['stain'] = "H&E"
            item['licence'] = "Apache 2.0"
            item['pixel_size'] = 0.275
            items.append(item)

    Segmentation_Dataset['Test'] += items

    return Segmentation_Dataset

def load_MoNuSeg(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    monuseg_dir = create_raw_datasets_dir("Nucleus_Segmentation", "MoNuSeg")
    zip_file_path = monuseg_dir / "MoNuSeg.zip"
    download_url = "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/monuseg-2018.zip"

    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")

        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {monuseg_dir / 'MoNuSeg'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(monuseg_dir / 'MoNuSeg')
        if verbose:
            print("Unzipping completed.")

    file_path = Path(f"{monuseg_dir}/MoNuSeg/monuseg-2018/download")
    
    # Process Test Data
    items = []
    set_type = "test"
    test_files = sorted(list((file_path / f"{set_type}/images").iterdir()))

    for file in tqdm(test_files):
        if ".DS_Store" not in str(file):
            item = {}
            data = io.imread(str(file))
            image = data[:, :, :3].astype(np.uint8)
            masks = io.imread(str(file).replace("images", "masks"))
            masks, remapping = fastremap.renumber(masks, in_place=True)
            item['nucleus_masks'] = fastremap.refit(masks)
            item['image'] = image
            item["parent_dataset"] = "MoNuSeg"
            item['licence'] = "CC BY NC 4.0"
            item['pixel_size'] = 0.25  # In microns per pixel
            item['image_modality'] = "Brightfield"
            item['stain'] = "H&E"
            items.append(item)

    Segmentation_Dataset['Test'] += items

    # Process Train Data
    items = []
    set_type = "train"
    train_files = sorted(list((file_path / f"{set_type}/images").iterdir()))

    for file in tqdm(train_files):
        if ".DS_Store" not in str(file):
            item = {}
            data = io.imread(str(file))
            image = data[:, :, :3].astype(np.uint8)
            masks = io.imread(str(file).replace("images", "masks"))
            masks, remapping = fastremap.renumber(masks, in_place=True)
            item['nucleus_masks'] = fastremap.refit(masks)
            item['image'] = image
            item["parent_dataset"] = "MoNuSeg"
            item['licence'] = "CC BY NC 4.0"
            item['pixel_size'] = 0.25  # In microns per pixel
            item['image_modality'] = "Brightfield"
            item['stain'] = "H&E"
            items.append(item)

    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):]

    return Segmentation_Dataset



def load_CIL(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    CIL_dir = create_raw_datasets_dir("Cell_Segmentation", "CIL")
    image_zip_path = CIL_dir / "CIL_images.zip"
    label_zip_path = CIL_dir / "CIL_labels.zip"

    import numpy as np
    import torch
    from skimage import io
    from tqdm import tqdm
    from instanseg.utils.augmentations import Augmentations
    from instanseg.utils.pytorch_utils import torch_fastremap

    if not image_zip_path.exists():
        # Define the URLs and paths
        image_url = "https://cildata.crbs.ucsd.edu/ccdb//telescience/home/CCDB_DATA_USER.portal/P2043/Experiment_6835/Subject_6837/Tissue_6840/Microscopy_6843/MP6843_img_full.zip"
        label_url = "https://cildata.crbs.ucsd.edu/ccdb//telescience/home/CCDB_DATA_USER.portal/P2043/Experiment_6835/Subject_6837/Tissue_6840/Microscopy_6843/MP6843_seg.zip"
        image_extract_path = CIL_dir / "CIL_images"
        label_extract_path = CIL_dir / "CIL_labels"

        # Download the image dataset using requests
        if verbose:
            print(f"Downloading image dataset from {image_url} to {image_zip_path}...")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(image_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Image dataset download completed.")

        # Download the label dataset using requests
        if verbose:
            print(f"Downloading label dataset from {label_url} to {label_zip_path}...")
        response = requests.get(label_url, stream=True)
        response.raise_for_status()
        with open(label_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Label dataset download completed.")

        # Unzip the image dataset
        if verbose:
            print(f"Unzipping image dataset to {image_extract_path}...")
        with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_extract_path)
        if verbose:
            print("Image dataset unzipping completed.")

        # Unzip the label dataset
        if verbose:
            print(f"Unzipping label dataset to {label_extract_path}...")
        with zipfile.ZipFile(label_zip_path, 'r') as zip_ref:
            zip_ref.extractall(label_extract_path)
        if verbose:
            print("Label dataset unzipping completed.")
    train_file_path = Path(CIL_dir) / "CIL_images"
    train_folders = sorted(list(train_file_path.iterdir()))
    items = []

    for file in tqdm(train_folders):
        if "w2" not in file.name:
            item = {}
            image = io.imread(file)
            image2 = io.imread(str(file).replace("w1","w2"))
            img = np.stack([image,image2],axis = 0)

            labels = io.imread(str(file).replace("CIL_images", "CIL_labels").replace("w1.TIF", "_GT_01.tif"))[:,:,0]
            from skimage.measure import label
            labels = label(labels)
            item['cell_masks'] =  labels
            item['image'] = img[:,::2,::2]
            item["parent_dataset"] = "CIL"
            item['licence'] = "CC BY 3.0"
            item['pixel_size'] = 0.3118 * 2
            item['image_modality'] = "Fluorescence"
            item['file_name'] = file.name

            items.append(item)

    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):int(len(items) * 0.9)]
    Segmentation_Dataset['Test'] += items[int(len(items) * 0.9):]

    return Segmentation_Dataset



def download_and_extract(url, dest_path, extract_to, verbose=True):
    if verbose:
        print(f"Downloading dataset from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    if verbose:
        print("Download completed.")

    if verbose:
        print(f"Unzipping dataset to {extract_to}...")
    with zipfile.ZipFile(dest_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    if verbose:
        print("Unzipping completed.")



def load_pannuke(Segmentation_Dataset: dict, verbose: bool = True, no_zip = False) -> dict:
    from pathlib import Path
    import numpy as np
    import fastremap
    import os

    pannuke_dir = create_raw_datasets_dir("Nucleus_Segmentation", "PanNuke")

    if not (pannuke_dir / "fold_1.zip").exists():
        for fold in ["1", "2", "3"]:
            zip_file_path = pannuke_dir / f"fold_{fold}.zip"
            download_url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip"

            # Download and unzip the dataset
            download_and_extract(download_url, zip_file_path, pannuke_dir, verbose)


    processed_pannuke_dir = create_processed_datasets_dir("pannuke_data")


    def get_data(dataset):

        if dataset == "train":
            fold = "1"
        elif dataset == "val":
            fold = "2"
        elif dataset == "test":
            fold = "3"

        import gc
        from tqdm import tqdm
        import tifffile

        out_path = create_processed_datasets_dir("pannuke_data",dataset)

        import fastremap

        items =[]
        masks = np.load(Path(pannuke_dir)/ "Fold {}/masks/fold{}/masks.npy".format(fold,fold), allow_pickle = True).astype(np.int16)
        images = np.load(Path(pannuke_dir)/ "Fold {}/images/fold{}/images.npy".format(fold,fold), allow_pickle = True).astype(np.uint8)
        types = np.load(Path(pannuke_dir)/ "Fold {}/images/fold{}/types.npy".format(fold,fold), allow_pickle = True)

        for i in tqdm(range(len(images))):
            item={}

            assert (np.unique(masks[i,:,:,:-1]) == np.unique(masks[i,:,:,:-1].max(-1))).all()
            label = masks[i,:,:,:-1].max(-1)
            label,_ = fastremap.renumber(label, in_place=True)
            classes = (np.argmax(masks[i,:,:,:-1],axis = -1) + 1) * (label > 0).astype(np.int8)

            image = images[i]

            tifffile.imwrite(out_path / ("image_"+str(i)+".tif"),image)
            tifffile.imwrite(out_path / ("nucleus_masks_"+str(i)+".tif"),fastremap.refit(fastremap.renumber(label)[0]))
            tifffile.imwrite(out_path / ("class_masks_"+str(i)+".tif"),classes)

            relative_path_img = os.path.relpath(str(out_path / ("image_"+str(i)+".tif")),os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_nucleus = os.path.relpath(str(out_path / ("nucleus_masks_"+str(i)+".tif")),os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_class = os.path.relpath(str(out_path / ("class_masks_"+str(i)+".tif")),os.environ['INSTANSEG_DATASET_PATH'])

            item['image']=relative_path_img
            item['nucleus_masks']= relative_path_nucleus
            item['class_masks']= relative_path_class

            item["parent_dataset"]="PanNuke"
            item['licence']="Attribution-NonCommercial-ShareAlike 4.0 International"
            item['image_modality']="Brightfield"
            item['pixel_size']= 0.25   #In microns per pixel
            item['tissue_type']= types[i]
            
            items.append(item)

        return items
    
    Segmentation_Dataset['Train']+=get_data("train")
    Segmentation_Dataset['Validation']+=get_data("val")
    Segmentation_Dataset['Test']+=get_data("test")

    if no_zip:
        return Segmentation_Dataset

    import shutil
    print("Zipping...")#Zip the processed data
    shutil.make_archive(processed_pannuke_dir, 'zip', processed_pannuke_dir)

    return Segmentation_Dataset


def load_tissuenet(Segmentation_Dataset: dict, verbose: bool = True, no_zip = False) -> dict:
    from pathlib import Path
    import numpy as np
    import tifffile
    import gc
    import fastremap
    from tqdm import tqdm
    import os
    import shutil
    import subprocess

    # Create raw and processed dataset directories
    tissuenet_dir = create_raw_datasets_dir("Cell_Segmentation", "TissueNet")
    processed_tissuenet_dir = create_processed_datasets_dir("tissuenet_data")

    def get_data(dataset):
        out_path = processed_tissuenet_dir / dataset
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)

        file_path = tissuenet_dir / f"tissuenet_v1.1/tissuenet_v1.1_{dataset}.npz"
        data = np.load(file_path, allow_pickle=True, mmap_mode='r')

        items = []
        imgs = data["X"]
        labels = data["y"]
        metas = data["meta"]

        for i in tqdm(range(len(imgs))):
            item = {}
            image = imgs[i]
            label = labels[i]
            meta = metas[i]

            tifffile.imwrite(out_path / f"image_{i}.tif", fastremap.refit(image))
            tifffile.imwrite(out_path / f"cell_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 0])[0]))
            tifffile.imwrite(out_path / f"nucleus_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 1])[0]))

            relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_cell = os.path.relpath(str(out_path / f"cell_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_nucleus = os.path.relpath(str(out_path / f"nucleus_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])

            item['image'] = relative_path_img
            item['cell_masks'] = relative_path_cell
            item['nucleus_masks'] = relative_path_nucleus

            item["parent_dataset"] = "TissueNet"
            item['licence'] = "Modified Apache, Non-Commercial"
            item['image_modality'] = "Fluorescence"
            item['pixel_size'] = meta[2]  # In microns per pixel
            item['nuclei_channels'] = [0]

            items.append(item)

        data.close()
        del data, imgs, labels, metas
        gc.collect()

        return items

    Segmentation_Dataset['Train'] += get_data("train")
    Segmentation_Dataset['Validation'] += get_data("val")
    Segmentation_Dataset['Test'] += get_data("test")

    if no_zip:
        return Segmentation_Dataset

    print("Zipping...")  # Zip the processed data
    shutil.make_archive(processed_tissuenet_dir, 'zip', processed_tissuenet_dir)

    return Segmentation_Dataset


def load_CPDMI_Vectra(Segmentation_Dataset: dict):

    from pathlib import Path
    from tifffile import TiffFile
    from skimage import io
    import kornia as K
    import torch
    import numpy as np
    import pdb
    import os

    base_path= create_raw_datasets_dir("Cell_Segmentation","CPDMI_2023")

    import pandas as pd
    df=pd.read_excel(base_path/"Annotation Panel Table.xlsx",sheet_name="Vectra")

    VECTRA_dir = create_raw_datasets_dir("Cell_Segmentation","CPDMI_2023", "Vectra")

    subfolders = sorted([path for path in Path(VECTRA_dir).iterdir() if path.is_dir()])
    items=[]
    for subfolder in subfolders:
        subsubfolders = sorted([path for path in Path(subfolder).iterdir() if path.is_dir()])

        for subsubfolder in subsubfolders:
            
            files=sorted([path for path in Path(subsubfolder).iterdir() if path.is_file()])
            cell_mask_path=[path for path in files if "Crop_Cell_Mask_Png" in str(path)]
            nuclei_mask_path=[path for path in files if "Crop_Dapi_Mask_Png" in str(path)]
            img_path=[path for path in files if "Crop_Cell_Tif" in str(path)] + [path for path in files if "Crop_Tif" in str(path)]

            if len(cell_mask_path)==0:
                print("No image found in",subsubfolder)
                continue

            image=io.imread(img_path[0])
            mask=io.imread(cell_mask_path[0])

            if len(nuclei_mask_path)>0:
                nuclei_mask=io.imread(nuclei_mask_path[0])
                connected_nuclei = K.contrib.connected_components(torch.tensor(nuclei_mask[None]).bool().float(), num_iterations=150).int().numpy()
                connected_nuclei = fastremap.renumber(connected_nuclei)[0].squeeze()
                

            if image.shape[-2:]!=mask.shape[-2:]:
                print(image.shape,mask.shape, subfolder,subsubfolder)
                
                continue

            connected = K.contrib.connected_components(torch.tensor(mask[None]).bool().float(), num_iterations=150).int().numpy()
            connected = fastremap.renumber(connected)[0].squeeze()
            connected = fastremap.refit(connected)
            item={}
            item['image']=image
            
            item['cell_masks']=connected

            item["parent_dataset"]="CPDMI_2023"
            item['licence']="CC BY 4.0"
            item['image_modality']="Fluorescence"
            item["platform"]="Vectra"

            if str(subsubfolder).split("/")[-1] == "P13-10002(45819.10401)2250,500":
                continue #this one is corrupted

    
            with TiffFile(img_path[0]) as tif:
                info_string=tif.imagej_metadata['Info']
                info_string=info_string.split("MetaDataPhotometricInterpretation = Monochrome")
                out_channels=info_string[1].split("NewSubfileType = 0")[0].replace("Name #","").split("\n")
                out_channels=[i for i in out_channels if i!=""]
                out_channels=[i.split("=")[1].strip() for i in out_channels]

            with TiffFile(img_path[0]) as tif:
                info_string=tif.imagej_metadata['Info']
                info_list=info_string.split("\n")
                magnification=[i for i in info_list if "XResolution" in i]
                item['resolution']=float(magnification[0].split("=")[1].strip())/1000
                if item['resolution'] > 30:
                    item["pixel_size"] = 0.25
                else:
                    item["pixel_size"] = 0.5
            

            assert len(out_channels)==image.shape[0], print(len(out_channels),image.shape[0])

            item['channel_names']=out_channels
            item['nuclei_channels']=[i for i,val in enumerate(out_channels) if "dapi" in val.lower()]


            if len(nuclei_mask_path)>0:
                connected_nuclei = fastremap.refit(connected_nuclei)

                item['nucleus_masks']=connected_nuclei

            annotation=str(subsubfolder).split("/")[-1].replace("(","[").replace(")","]").replace(".",",") 

            item['filename']=annotation

            if annotation not in df["Annotation:"].unique(): #some annotations are missing
                if annotation == "P07-10004[53024,13505]250,600":
                    item["tissue_type"]="Colon"
                    item["tumor_type"]="Adenocarcinoma"
                elif annotation == "P05-10008[53049,13410]2000,400":
                    item["tissue_type"]="Pancreas"
                    item["tumor_type"]="Pancreatic Ductal Adenocarcinoma"
                elif annotation == "P03-10009[53725,9272]570,10":
                    item["tissue_type"]="Lung"
                    item["tumor_type"]="Adenocarcinoma"
                elif annotation == "P03-10005[46905,11217]800,600":
                    item["tissue_type"]="Lung"
                    item["tumor_type"]="Adenocarcinoma"
                elif annotation == "P03-10005[46905,11217]0,600":
                    item["tissue_type"]="Lung"
                    item["tumor_type"]="Adenocarcinoma"
                elif annotation == "P02-10009[53829,17312]943,604":
                    item["tissue_type"]="Lung"
                    item["tumor_type"]="Adenocarcinoma"

                continue

            item["tissue_type"]=df[df["Annotation:"]==annotation]["Organ:"].values[0]
            item["tumor_type"]=df[df["Annotation:"]==annotation]["Tissue Type:"].values[0]

            items.append(item)

    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train']+=items[:int(len(items)*0.8)]
    Segmentation_Dataset['Validation']+=items[int(len(items)*0.8):]

    return Segmentation_Dataset





def load_CPDMI_Zeiss(Segmentation_Dataset: dict):

    from pathlib import Path
    from tifffile import TiffFile
    from skimage import io
    import kornia as K
    import torch
    import numpy as np

    Zeiss = create_raw_datasets_dir("Cell_Segmentation","CPDMI_2023", "Zeiss")

    subfolders = sorted([path for path in Path(Zeiss).iterdir() if path.is_dir()])
    items=[]
    for subfolder in subfolders:

        files=sorted([path for path in Path(subfolder).iterdir() if path.is_file()])
        cell_mask_path=[path for path in files if "Crop_Cell_Mask_Png" in str(path)]
        img_path=[path for path in files if "Crop_Cell_Tif" in str(path)] + [path for path in files if "Crop_Tif" in str(path)]

        if len(cell_mask_path)==0:
            print("No image found in",subfolder)
            continue

        
        image=io.imread(img_path[0])
        mask=io.imread(cell_mask_path[0])

        if image.shape[-2:]!=mask.shape[-2:]:
            print(image.shape,mask.shape, subfolder,subfolder)
            continue

        connected = K.contrib.connected_components(torch.tensor(mask[None]).bool().float(), num_iterations=150).int().numpy()
        connected = fastremap.renumber(connected)[0].squeeze().astype(np.int16)

        item={}
        item['image']=image
        item['cell_masks']=connected

        item["parent_dataset"]="CPDMI_2023"
        item['licence']="CC BY 4.0"
        item['image_modality']="Fluorescence"
        item["platform"]="Zeiss"
        item["pixel_size"]=0.325

        if "ZP-9999" in subfolder.name:
            item["tissue_type"]="Skin"
            item["tumor_type"]="Basal_Cell"
            item['channel_names']=['DAPI', 'CD8', 'CD4', 'FoxP3', 'PanCK']
            item['nuclei_channels']=[0]
        elif "ZP-10002" in subfolder.name:
            item["tissue_type"]="Skin"
            item["tumor_type"]="CTCL"
            item['channel_names']=['PanCK', 'PD-L1', 'DAPI', 'CD68', 'CD3', 'FoxP3', 'CD8']
            item['nuclei_channels']=[2]
        elif "ZP-10001" in subfolder.name:
            item["tissue_type"]="Skin"
            item["tumor_type"]="CTCL"
            item['channel_names']=['DAPI', 'CD3', 'PD-L1', 'FoxP3', 'PanCK']
            item['nuclei_channels']=[0]
        elif "Spleen" in subfolder.name:
            item["tissue_type"]="Spleen"
            item["tumor_type"]="Melanoma"
            item['channel_names']=['DAPI', 'CD8', 'CD68', 'PD-L1', 'PanCK']
            item['nuclei_channels']=[0]
        elif "PDAC" in subfolder.name:
            item["tissue_type"]="Pancreas"
            item["tumor_type"]="PDAC"
            item['channel_names']=['DAPI', 'CD8', 'CD68', 'PD-L1', 'PanCK']
            item['nuclei_channels']=[0]

        item['filename']=img_path[0].stem

        items.append(item)

    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train']+=items[:int(len(items)*0.8)]
    Segmentation_Dataset['Validation']+=items[int(len(items)*0.8):]

    return Segmentation_Dataset


def load_CPDMI_CODEX(Segmentation_Dataset: dict):

    from pathlib import Path
    from tifffile import TiffFile
    from skimage import io
    import kornia as K
    import torch
    import numpy as np

    CODEX = create_raw_datasets_dir("Cell_Segmentation","CPDMI_2023", "CODEX")

    subfolders = sorted([path for path in Path(CODEX).iterdir() if path.is_dir()])
    items=[]
    for subfolder in subfolders:

        files=sorted([path for path in Path(subfolder).iterdir() if path.is_file()])
        cell_mask_path=[path for path in files if "Crop_Cell_Mask_Png" in str(path)]
        nucleus_mask_path=[path for path in files if "Crop_Dapi_Mask_Png" in str(path)]
        img_path=[path for path in files if "Crop_Cell_Tif" in str(path)] + [path for path in files if "Crop_Tif" in str(path)]

        if len(cell_mask_path)==0:
            print("No image found in",subfolder)
            continue

        image=io.imread(img_path[0])
        mask=io.imread(cell_mask_path[0])

        if len(nucleus_mask_path)>0:
            nuclei_mask=io.imread(nucleus_mask_path[0])
            connected_nuclei = K.contrib.connected_components(torch.tensor(nuclei_mask[None]).bool().float(), num_iterations=150).int().numpy()
            connected_nuclei = fastremap.renumber(connected_nuclei)[0].squeeze().astype(np.int16)

        im=image.transpose(0,3,1,2) 
        image=im.reshape(-1,im.shape[-2],im.shape[-1]) #This was checked by eye and looks correct. The channel ordering was not recorded properly in the metadata.

        if image.shape[-2:]!=mask.shape[-2:]:
            print("Dimension inconsistency",image.shape,mask.shape, subfolder,subfolder)
            continue

        connected = K.contrib.connected_components(torch.tensor(mask[None]).bool().float(), num_iterations=150).int().numpy()
        connected = fastremap.renumber(connected)[0].squeeze().astype(np.int16)
        item={}
        item['image']=image
        item['cell_masks']=connected

        if len(nucleus_mask_path)>0:
            item['nucleus_masks']=connected_nuclei


        item["parent_dataset"]="CPDMI_2023"
        item['licence']="CC BY 4.0"
        item['image_modality']="Fluorescence"
        item["platform"]="CODEX"
        item["pixel_size"]=0.377

        if "LN" in subfolder.name:
            item["tissue_type"]="Lymph_Node"
            item["tumor_type"]="Normal_Tissue"
            item['channel_names']=["DAPI1","NA","NA","NA","DAPI2","CD8","CD3","CD20","DAPI3","Ki67","CD68","PanCK","DAPI4","CD21","CD4","CD31","DAPI5","CD45RO","CD11c","NA","DAPI6","NA","HLA-DR","NA","DAPI7","NA","NA","NA"]
            item['nuclei_channels']=[0, 4, 8, 12, 16, 20, 24]
        elif "Tnsl" in subfolder.name:
            item["tissue_type"]="Tonsil"
            item["tumor_type"]="Normal_Tissue"
            item['channel_names']=['DAPI1', 'NA', 'NA', 'NA', 'DAPI2', 'NA', 'CD4', 'NA', 'API3', 'CD8', 'CD3e', 'CD20', 'DAPI4', 'Ki67', 'HLA-DR', 'DAPI5', 'NA', 'CD68', 'CD31', 'DAPI6', 'CD45RO', 'CD11c', 'NA', 'DAPI7', 'CD21', 'NA', 'NA', 'DAPI8', 'NA', 'NA', 'NA']
            item['nuclei_channels']=[0, 4, 12, 15, 19, 23, 27]

        item['filename']=img_path[0].stem

        items.append(item)

    Segmentation_Dataset['Test']+=items

    return Segmentation_Dataset



def load_BSST265(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    bsst265_dir = create_raw_datasets_dir("Nucleus_Segmentation", "BSST265")
    zip_file_path = bsst265_dir / "BSST265.zip"
    download_url = "https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip"
    if not zip_file_path.exists():
        # Download the dataset using requests
        if verbose:
            print(f"Downloading dataset from {download_url} to {zip_file_path}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Download completed.")
        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {bsst265_dir / 'BSST265'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(bsst265_dir / 'BSST265')
        if verbose:
            print("Unzipping completed.")
    import pandas as pd
    metadata = pd.read_csv(bsst265_dir / "BSST265" / "image_description.csv", sep=";")
    
    file_path = Path(f"{bsst265_dir}/BSST265/rawimages")
    files = sorted(list(file_path.iterdir()))
    items = []
    for i, file in enumerate(tqdm(files)):
        metadata_row = metadata[metadata["Image_Name"] == file.stem]
        magnification = metadata_row["Magnification"].values[0]
        if magnification == "20x":
            pixel_size = 0.323
        elif magnification == "40x":
            pixel_size = 0.161
        elif magnification == "63x":
            pixel_size = 0.102
        img_file = str(file)
        mask_file = img_file.replace("rawimages", "groundtruth")
        item = {}
        image = io.imread(img_file)
        masks = io.imread(mask_file)
        item['nucleus_masks'] = fastremap.refit(masks)
        item['image'] = image
        item["parent_dataset"] = "BSST265"
        item['licence'] = "CC0"
        item['pixel_size'] = pixel_size
        item['nuclei_channels'] = [0]  
        items.append(item)
    np.random.seed(42) 
    np.random.shuffle(items)
    Segmentation_Dataset['Train']+=items[:int(len(items)*0.8)]
    Segmentation_Dataset['Validation']+=items[int(len(items)*0.8):int(len(items)*0.9)]
    Segmentation_Dataset['Test']+=items[int(len(items)*0.9):]
    return Segmentation_Dataset
