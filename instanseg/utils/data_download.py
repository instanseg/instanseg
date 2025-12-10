
from pathlib import Path
from instanseg.utils.utils import _move_channel_axis
import fastremap
from tqdm import tqdm
from skimage import io
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

    if not image_zip_path.exists(): #https://www.cellimagelibrary.org/images/40217
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

            image=io.imread(img_path[0])

            item={}
            item['image']=image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            if len(cell_mask_path)>0:
                cell_mask=io.imread(cell_mask_path[0])
                connected_cells = K.contrib.connected_components(torch.tensor(cell_mask[None]).bool().float().to(device), num_iterations=150).int().cpu().numpy()
                connected_cells = fastremap.renumber(connected_cells)[0].squeeze()
                connected_cells = fastremap.refit(connected_cells)
                item['cell_masks']=connected_cells

                if image.shape[-2:]!=cell_mask.shape[-2:]:
                   # print(image.shape,cell_mask.shape, subfolder,subsubfolder)
                    #get offset from folder name
                    (x,y) = [int(i) for i in str(cell_mask_path[0].parents[0]).split(")")[-1].split(",")]
                  #  from instanseg.utils.utils import show_images
                  #  show_images([image[-2,y : y + cell_mask.shape[0],x:x + cell_mask.shape[1]],cell_mask])
                    item["image"] = image[:,y : y + cell_mask.shape[0],x:x + cell_mask.shape[1]]

                
            if len(nuclei_mask_path)>0:
                nuclei_mask=io.imread(nuclei_mask_path[0])
                connected_nuclei = K.contrib.connected_components(torch.tensor(nuclei_mask[None]).bool().float().to(device), num_iterations=150).int().cpu().numpy()
                connected_nuclei = fastremap.renumber(connected_nuclei)[0].squeeze()
                connected_nuclei = fastremap.refit(connected_nuclei)
                item['nucleus_masks']=connected_nuclei
                
                if image.shape[-2:]!=nuclei_mask.shape[-2:]:
                    print(image.shape,nuclei_mask.shape, subfolder,subsubfolder)

                    from instanseg.utils.utils import show_images

                    show_images([image[-2],nuclei_mask])
                    1/0
                    continue

            assert "nucleus_masks" in item or "cell_masks" in item, print("No masks found in",subsubfolder)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    for subfolder in subfolders:

        files=sorted([path for path in Path(subfolder).iterdir() if path.is_file()])
        cell_mask_path=[path for path in files if "Crop_Cell_Mask_Png" in str(path)]
        nuclei_mask_path=[path for path in files if "Crop_Dapi_Mask_Png" in str(path)]

        img_path = sorted(
            [path for path in files if "Crop_Cell_Tif" in str(path)] + 
            [path for path in files if "Crop_Tif" in str(path)],
            key=os.path.getsize
        ) #sort by size to get the smallest image

        if len(img_path) > 1:
            print("deleting redundant image:",img_path[-1])
            os.remove(img_path[-1])

        cell_mask = io.imread(cell_mask_path[0])

        if len(nuclei_mask_path)>0:
            print("nuclei_mask_path",nuclei_mask_path)

        if len(img_path) > 1:
            for img in img_path:
                image = io.imread(img)
                if image.shape[-2:] == cell_mask.shape[-2:]:
                    img_path = [img]
                    break

        if len(cell_mask_path)==0:
            print("No image found in",subfolder)
            continue
        
        image=io.imread(img_path[0])
        cell_mask=io.imread(cell_mask_path[0])

        if image.shape[-2:]!=cell_mask.shape[-2:]:
            print(image.shape,cell_mask.shape, subfolder)
    
        connected = K.contrib.connected_components(torch.tensor(cell_mask[None]).bool().float().to(device), num_iterations=150).cpu().int().numpy()
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        connected = K.contrib.connected_components(torch.tensor(mask[None]).bool().float().to(device), num_iterations=150).int().cpu().numpy()
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
            item['channel_names']=['DAPI1', 'NA', 'NA', 'NA', 'DAPI2', 'NA', 'CD4', 'NA', 'DAPI3', 'CD8', 'CD3e', 'CD20', 'DAPI4', 'Ki67', 'HLA-DR', 'DAPI5', 'NA', 'CD68', 'CD31', 'DAPI6', 'CD45RO', 'CD11c', 'NA', 'DAPI7', 'CD21', 'NA', 'NA', 'DAPI8', 'NA', 'NA', 'NA']
            item['nuclei_channels']=[0, 4, 12, 15, 19, 23, 27]

        item['filename']=img_path[0].stem

        items.append(item)

    Segmentation_Dataset['Train']+=items

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


def load_CellBinDB(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
    cellbindb_dir = create_raw_datasets_dir("Nucleus_Segmentation", "CellBinDB")



    zip_file_path = cellbindb_dir / "CellBinDB.zip"

    download_url = "https://zenodo.org/records/14312044/files/CellBinDB.zip?download=1"

    if not zip_file_path.exists():
        print("Downloading CellBinDB dataset...")
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
            print(f"Unzipping dataset to {cellbindb_dir}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(cellbindb_dir)
        if verbose:
            print("Unzipping completed.")

    cellbindb_dir = os.path.join(cellbindb_dir, "CellBinDB")

    dataset_path = sorted(Path(f"{cellbindb_dir}").iterdir())

    print(f"Found {len(dataset_path)} tissue directories in CellBinDB dataset.")

    # Iterate over the dataset folders and files
    items = []
    for tissue_dir in dataset_path:
        if not tissue_dir.is_dir():
            continue

        for sample_dir in sorted(tissue_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            img_file = sample_dir / f"{sample_dir.name}-img.tif"
            instance_mask_file = sample_dir / f"{sample_dir.name}-instancemask.tif"

            if img_file.exists() and instance_mask_file.exists():
                item = {}
                image = io.imread(img_file)
                instance_mask = io.imread(instance_mask_file)

                from instanseg.utils.utils import show_images

                #check if we are in mIF folder

                if sample_dir.parents[0].name == "mIF":
                    item['cell_masks'] = instance_mask
                else:
                    item['nucleus_masks'] = instance_mask

                item['image'] = image
                item['parent_dataset'] = "CellBinDB"
                item['licence'] = "CC bY 4.0"
                item['pixel_size'] = 0.5  # Assuming a default pixel size, update if known

                items.append(item)

    # Shuffle and split the dataset into Train, Validation, and Test
    np.random.seed(42)
    np.random.shuffle(items)
    num_items = len(items)

    Segmentation_Dataset['Train'] += items[:int(num_items * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(num_items * 0.8):int(num_items * 0.9)]
    Segmentation_Dataset['Test'] += items[int(num_items * 0.9):]

    return Segmentation_Dataset





def load_HPA_Segmentation(Segmentation_Dataset: dict, verbose: bool = True) -> dict:

    def geojson_to_label_image(geojson_data: dict, image_shape: tuple):

        from shapely.geometry import shape
        from rasterio.features import rasterize

        # Prepare an empty label image
        label_image = np.zeros(image_shape, dtype=np.uint16)
        
        # Initialize label counter
        label_id = 1
        
        # Iterate over each feature in the GeoJSON data
        for feature in geojson_data['features']:
            # Parse the geometry to get a polygon
            polygon_geom = shape(feature['geometry'])
            
            # Rasterize the polygon onto the label image
            mask = rasterize(
                [(polygon_geom, label_id)],
                out_shape=image_shape,
                fill=0,
                default_value=label_id,
                dtype=np.uint16
            )
            
            # Add the mask to the label image
            label_image[mask > 0] = mask[mask > 0]
            label_id += 1  # Increment label ID for the next polygon

        return label_image
    

    import json
    hpa_dir = create_raw_datasets_dir("Cell_Segmentation", "HPA")
    zip_file_path = hpa_dir / "HPA_Segmentation.zip"
    download_url = "https://zenodo.org/records/4430893/files/hpa_cell_segmentation_dataset_v2_512x512_4train_159test.zip?download=1"
    
    
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
            print("Download completed.")
        
        # Unzip the dataset
        if verbose:
            print(f"Unzipping dataset to {hpa_dir / 'HPA'}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(hpa_dir / 'HPA')
        if verbose:
            print("Unzipping completed.")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Traverse through each sample directory in the test folder
    test_dir = hpa_dir / "HPA/hpa_dataset_v2" / "test"
    train_dir = hpa_dir / "HPA/hpa_dataset_v2" / "train"
    sample_dirs = [d for d in test_dir.iterdir() if d.is_dir()] + [d for d in train_dir.iterdir() if d.is_dir()]
    
    items = []
    for sample_dir in tqdm(sample_dirs):
        sample_name = sample_dir.name
        # Load each image channel
        channel_images = []
        
        # Define paths for each image channel
        channels = [ "er.png", "microtubules.png", "nuclei.png", "protein.png"]
        for channel in channels:
            channel_path = sample_dir / channel
            if channel_path.exists():
                image = io.imread(channel_path)
                channel_images.append(image)

        # Stack channels into a single multi-channel image
        multi_channel_image = np.stack(channel_images)
        
        # Load annotation.json and convert to label image
        annotation_path = sample_dir / "annotation.json"
        with open(annotation_path, "r",encoding="utf-8-sig") as f:
            annotation_data = json.load(f)
        
        # Create label image from JSON data
        # Assuming JSON contains a list of labeled regions, e.g., {"cells": [{"label": 1, "coordinates": [...]}, ...]}
        label_image = geojson_to_label_image(annotation_data, multi_channel_image.shape[-2:])

        item = {}

        susbsammpling = 2
        image = multi_channel_image[:,::susbsammpling,::susbsammpling]
        masks = label_image[::-susbsammpling,::susbsammpling]
        item['cell_masks'] = masks
        item['image'] = image
        item["parent_dataset"] = "HPA"
        item['licence'] = "CC BY 4.0"
        item['pixel_size'] = 0.08 * susbsammpling
        item['nuclei_channels'] = [2]  
        item['channel_names'] = ["ER", "Microtubules", "Nuclei", "Protein"]
        item['image_modality'] = "Fluorescence"
        items.append(item)
    
    # Split dataset into training, validation, and test sets
    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):int(len(items) * 0.9)]
    Segmentation_Dataset['Test'] += items[int(len(items) * 0.9):]
    
    return Segmentation_Dataset


def load_cellseg(Segmentation_Dataset: dict, verbose: bool = True, no_zip = False) -> dict:
    import requests
    import zipfile
    from skimage import io
    from pathlib import Path
    import numpy as np
    from skimage.transform import rescale
    import tifffile
    from tqdm import tqdm

    cellseg_dir = create_raw_datasets_dir("Cell_Segmentation", "NeurIPS_CellSeg")

    processed_cellseg_dir = create_processed_datasets_dir("cellseg_data")


    def get_data(split):
        out_path = processed_cellseg_dir / split

        if split == "train":
            zip_file_path = cellseg_dir / "Training-labeled.zip"
            download_url = "https://zenodo.org/records/10719375/files/Training-labeled.zip?download=1"
            neurips_path = cellseg_dir / "Training-labeled"
        
        elif split == "val":
            zip_file_path = cellseg_dir / "Tuning.zip"
            download_url = "https://zenodo.org/records/10719375/files/Tuning.zip?download=1"
            neurips_path = cellseg_dir / "Tuning"

        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)


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
                print(f"Unzipping dataset to {cellseg_dir}...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(cellseg_dir)
            if verbose:
                print("Unzipping completed.")


        

        files = sorted(os.listdir(os.path.join(neurips_path,"images")))

        items = []

        for i,file in enumerate(tqdm(files)):
            image_path = os.path.join(neurips_path, "images", file)
            mask_path = os.path.join(neurips_path, "labels", Path(file).stem + "_label.tiff")

            image = io.imread(image_path)
            mask = io.imread(mask_path)

            if image.ndim == 2:
                image = image[..., np.newaxis]

            median_area = np.median(np.unique(mask[mask > 0],return_counts = True)[1])
            median_diameter = np.sqrt(median_area / np.pi) * 2

            target_diameter = 30

            #resize image to target diameter
            scale_factor = target_diameter / median_diameter

            image_rescaled = rescale(image, scale=scale_factor, preserve_range=True, anti_aliasing=True, channel_axis=np.argmin(image.shape)).astype(image.dtype)
            mask_rescaled = rescale(mask, scale=scale_factor, order = 0).astype(mask.dtype)

        #  show_images(image_rescaled, mask_rescaled,)

            item = {}

            tifffile.imwrite(out_path / f"image_{i}.tif", image_rescaled)
            tifffile.imwrite(out_path / f"cell_masks_{i}.tif", mask_rescaled)

            relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_cell = os.path.relpath(str(out_path / f"cell_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])

            item['image'] = relative_path_img
            item['cell_masks'] = relative_path_cell

            item["parent_dataset"] = "cellseg"
            item['licence'] = "cc by nc nd"
            item['image_modality'] = "Fluorescence"
            item['pixel_size'] = None  # In microns per pixel

            items.append(item)
        
        return items

    Segmentation_Dataset['Train'] += get_data("train")
    Segmentation_Dataset['Validation'] += get_data("val")
   

    return Segmentation_Dataset


 
#### FROM CELLPOSE !!! ####
 
def remove_overlaps(masks, medians, overlap_threshold=0.75):
    """ replace overlapping mask pixels with mask id of closest mask
        if mask fully within another mask, remove it
        masks = Nmasks x Ly x Lx
    """
    cellpix = masks.sum(axis=0)
    igood = np.ones(masks.shape[0], 'bool')
    for i in masks.sum(axis=(1, 2)).argsort():
        npix = float(masks[i].sum())
        noverlap = float(masks[i][cellpix > 1].sum())
        if noverlap / npix >= overlap_threshold:
            igood[i] = False
            cellpix[masks[i] > 0] -= 1
            #print(cellpix.min())
   # print(f'removing {(~igood).sum()} masks')
    masks = masks[igood]
    medians = medians[igood]
    cellpix = masks.sum(axis=0)
    overlaps = np.array(np.nonzero(cellpix > 1.0)).T
    dists = ((overlaps[:, :, np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:, 0], overlaps[:, 1]] = 0
    masks[tocell, overlaps[:, 0], overlaps[:, 1]] = 1
 
    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1, masks.shape[0] + 1, 1, int)[:, np.newaxis,
                                                                         np.newaxis]
    masks = masks.sum(axis=0)
    return masks
 
def ann_to_masks(annotations, anns, overlap_threshold=0.75):
    """ list of coco-format annotations with masks to single image"""
    masks = []
    k = 0
    medians = []
    for ann in anns:
        mask = annotations.annToMask(ann)
        masks.append(mask)
        ypix, xpix = mask.nonzero()
        medians.append(np.array([ypix.mean(), xpix.mean()]))
        k += 1
    masks = np.array(masks).astype('int')
    medians = np.array(medians)
    masks = remove_overlaps(masks, medians, overlap_threshold=overlap_threshold)
    return masks
 
def livecell_ann_to_masks(img_dir, annotation_file):
    from pycocotools.coco import COCO
 
    from glob import glob
 
    from tifffile import imsave
    img_dir_classes = glob(img_dir + '*/')
    classes = [img_dir_class.split(os.sep)[-2] for img_dir_class in img_dir_classes]
    #print(classes)
 
    train_files = []
    train_class_files = []
    for cclass, img_dir_class in zip(classes, img_dir_classes):
        train_files.extend(glob(img_dir_class + '*.tif'))
        train_class_files.append(glob(img_dir_class + '*.tif'))
 
    annotations = COCO(annotation_file)
    imgIds = list(annotations.imgs.keys())
 
    for train_class_file in train_class_files:
        for i in tqdm(range(len(train_class_file))):
            filename = train_class_file[i]
 
            if not os.path.exists(os.path.splitext(filename)[0] + '_masks.tif'):
                fname = os.path.split(filename)[-1]
                loc = np.array([
                    annotations.imgs[imgId]['file_name'] == fname for imgId in imgIds
                ]).nonzero()[0]
                if len(loc) > 0:
                    imgId = imgIds[loc[0]]
                    annIds = annotations.getAnnIds(imgIds=[imgId], iscrowd=None)
                    anns = annotations.loadAnns(annIds)
                    masks = ann_to_masks(annotations, anns, overlap_threshold=0.75)
                    from instanseg.utils.utils import show_images
                  #  show_images(masks,labels = [0])
                    masks = masks.astype(np.uint16)
                    maskname = os.path.splitext(filename)[0] + '_masks.tif'
                    imsave(maskname, masks)
                   # print(f'saved masks at {maskname}')
#end of cellpose code ######
 
def load_LIVECELL(Segmentation_Dataset: dict, verbose: bool = True) -> dict:
 
    import tifffile
 
    livecell_dir = create_raw_datasets_dir("Cell_Segmentation", "LIVECELL")
 
 
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
 
        if extract_to.suffix == ".zip":
            if verbose:
                print(f"Unzipping dataset to {extract_to}...")
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            if verbose:
                print("Unzipping completed.")
 
 
    if not (livecell_dir / "images.zip").exists():
        download_and_extract("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip", livecell_dir / "images.zip", livecell_dir)
    if not (livecell_dir / "livecell_coco_train.json").exists():
        download_and_extract("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json", livecell_dir / "livecell_coco_train.json", livecell_dir)
    if not (livecell_dir / "livecell_coco_test.json").exists():
        download_and_extract("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json", livecell_dir / "livecell_coco_test.json", livecell_dir)
    if not (livecell_dir / "livecell_coco_val.json").exists():
        download_and_extract("http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json", livecell_dir / "livecell_coco_val.json", livecell_dir)
 
 
    livecell_ann_to_masks(str(livecell_dir / "images/livecell_train_val_images"), livecell_dir / "livecell_coco_train.json")
    livecell_ann_to_masks(str(livecell_dir / "images/livecell_test_images"), livecell_dir / "livecell_coco_test.json")
    livecell_ann_to_masks(str(livecell_dir / "images/livecell_train_val_images"), livecell_dir / "livecell_coco_val.json")
 
 
    processed_livecell_dir = create_processed_datasets_dir("livecell_data")
    train_files = []
    test_files = []
    for root, dirs, filenames in os.walk(livecell_dir):
        for filename in filenames:
            if filename.endswith(".tif"):
              #  print(root,filename)
                if str(root).endswith("livecell_train_val_images") and "masks" not in filename:
                    train_files.append(os.path.join(root, filename))
                elif str(root).endswith("livecell_test_images") and "masks" not in filename:
                    test_files.append(os.path.join(root, filename))
 
    print(f"Found {len(train_files)} training images and {len(test_files)} test images.")
 
    def get_data(dataset):
        out_path = processed_livecell_dir / dataset
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
 
        if dataset == "train":
            imgs = train_files
        elif dataset == "test":
            imgs = test_files
 
        items = []
 
        for i,file in tqdm(enumerate(imgs)):
            item = {}
            image = io.imread(file)
 
            from instanseg.utils.utils import show_images
            mask = io.imread(file.replace(".tif", "_masks.tif"))
 
        
            tifffile.imwrite(out_path / f"image_{i}.tif", image)
            tifffile.imwrite(out_path / f"cell_masks_{i}.tif", mask)
 
            relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_cell = os.path.relpath(str(out_path / f"cell_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            
            image = io.imread(file)
            mask = io.imread(file.replace("input", "output"))
            item['cell_masks'] = relative_path_cell
            item['image'] = relative_path_img
            item["parent_dataset"] = "LIVECELL"
            item['licence'] = "CC BY-NC 4.0"
            item['pixel_size'] = 1.2429  # In microns per pixel
            item['image_modality'] = "Brightfield"
            items.append(item)
        return items
 
    items = get_data("train")
 
    np.random.seed(42)
    np.random.shuffle(items)
    Segmentation_Dataset['Train'] += items[:int(len(items) * 0.8)]
    Segmentation_Dataset['Validation'] += items[int(len(items) * 0.8):]
 
    Segmentation_Dataset['Test'] += get_data("test")
    return Segmentation_Dataset
 