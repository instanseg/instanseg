import os
import bioimageio.core
import numpy as np
import torch
from bioio import BioImage

from instanseg.utils.augmentations import Augmentations
from instanseg.utils.utils import _choose_device, show_images
from instanseg.utils.model_loader import load_model


def set_export_paths():
    from pathlib import Path
    if os.environ.get('INSTANSEG_BIOIMAGEIO_PATH'):
        path = Path(os.environ['INSTANSEG_BIOIMAGEIO_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../bioimageio_models/"))
        os.environ['INSTANSEG_BIOIMAGEIO_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if os.environ.get('INSTANSEG_TORCHSCRIPT_PATH'):
        path = Path(os.environ['INSTANSEG_TORCHSCRIPT_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../torchscripts/"))
        os.environ['INSTANSEG_TORCHSCRIPT_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if os.environ.get('INSTANSEG_MODEL_PATH'):
        path = Path(os.environ['INSTANSEG_MODEL_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../models/"))
        os.environ['INSTANSEG_MODEL_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)


dataset_dict = {
    "DSB_2018": ["CC 0","https://bbbc.broadinstitute.org/BBBC038"],
    "CoNSeP": ["Apache 2.0","https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet"],
    "TNBC_2018": ["CC BY 4.0","https://zenodo.org/records/3552674"],
    "MoNuSeg": ["CC BY NC 4.0","https://monuseg.grand-challenge.org/"],
    "LyNSec": ["CC BY 4.0","https://zenodo.org/records/8065174"],
    "LyNSeC": ["CC BY 4.0","https://zenodo.org/records/8065174"],
    "NuInsSeg": ["CC BY 4.0","https://zenodo.org/records/10518968"],
    "IHC_TMA": ["CC BY 4.0","https://zenodo.org/records/7647846"],
    "CPDMI_2023": ["CC BY 4.0","https://www.nature.com/articles/s41597-023-02108-z"],
    "cellpose": ["NC","https://www.cellpose.org/dataset"],
    "TissueNet": ["Modified Apache, Non-Commercial", "https://datasets.deepcell.org/"],
    "CIL": ["CC BY 3.0", "https://www.cellimagelibrary.org/images/CCDB_6843"],
    "BSST265": ["CC0","https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265"],
}



# from bioimageio.core.build_spec import build_model
# from bioimageio.core.resource_tests import test_model

from bioimageio.spec.model.v0_5 import ModelDescr

def readme(model_name: str, model_dict: dict = None):
    # create markdown documentation for your model
    # this should describe how the model was trained, (and on which data)
    # and also what to take into consideration when running the model, especially how to validate the model
    # here, we just create a stub documentation

    with open(os.path.join(model_name, model_name + "_README.md"), "w") as f:
        f.write("# This is an InstanSeg model. \n")
        f.write("The InstanSeg method is shared with an Apache-2.0 license.\n\n")
        f.write("""For an introduction & comparison to other approaches for nucleus segmentation in brightfield histology images, see: \n > Goldsborough, T. et al. (2024) InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation. _arXiv_. Available at: https://doi.org/10.48550/arXiv.2408.15954. \n\n To read about InstanSeg's extension to nucleus + full cell segmentation and support for fluorescence & multiplexed images, see: \n > Goldsborough, T. et al. (2024) A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg. _bioRxiv_, p. 2024.09.04.611150. Available at: https://doi.org/10.1101/2024.09.04.611150. \n""")
        f.write("\n This model was trained on the following datasets: \n")
        
        if model_dict is not None and "source_dataset" in model_dict.keys():
            for dataset in (model_dict["source_dataset"]).replace("[","").replace("]","").replace("'","").split(", "):
                if dataset in dataset_dict.keys():
                    f.write(f"- {dataset} \n")
                    f.write(f"  - License: {dataset_dict[dataset][0]} \n")
                    f.write(f"  - URL: {dataset_dict[dataset][1]} \n")
                else:
                    f.write(f"- {dataset} \n")
                    f.write(f"  - License: Not specified \n")
                    f.write(f"  - URL: Not specified \n")
        

        f.write("\n The user is responsible for ensuring that the model is used in accordance with the licenses of the source datasets. \n")

          #  f.write(str(model_dict["source_dataset"]))



import os, shutil
def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, root_dir=archive_from, base_dir=archive_to)
        shutil.move('%s.%s'%(name,format), destination)



def export_bioimageio(torchsript: torch.jit._script.RecursiveScriptModule, 
                      model_name: str, 
                      test_img_path: str, 
                      model_dict: dict = None, 
                      output_name = None,
                      output_channel_names = None,
                      output_types = ["instance_segmentation"],
                      version: str = None):
    
    set_export_paths()

    output_path = os.environ['INSTANSEG_BIOIMAGEIO_PATH']

    if output_name is None:
        output_name = model_name
    # create a directory to store bioimage.io model files
    os.makedirs(output_name, exist_ok=True)
    # save the model weights
    torchsript.save(os.path.join(output_name, "instanseg.pt"))


    try:
        model,model_dict = load_model(model_name, path = os.environ['INSTANSEG_MODEL_PATH'])
    except:
        raise Exception("Model configuration files could not be loaded")
    
    print("Model pixel size: ", model_dict["pixel_size"])
    model_pixel_size = model_dict["pixel_size"]

    torchsript.eval()
    device = _choose_device()
    torchsript.to(device)

    img = BioImage(test_img_path)
    if "S" in img.dims.order and img.dims.S > img.dims.C:
        input_data = img.get_image_data("SYX")
    else:
        input_data = img.get_image_data("CYX")
        
    if img.physical_pixel_sizes.X is not None:
        pixel_size = img.physical_pixel_sizes.X
        print("Pixel size was found in the metadata, pixel size is set to: ", pixel_size)
    else:
        pixel_size = 0.5
        print("Pixel size was not found in the metadata, please set the pixel size of the input image in microns manually")

    if model_dict["channel_invariant"]:
        dim_in = 1
        step = 1
    else:
        dim_in = model_dict["dim_in"]
        step = 0


    target_segmentation = model_dict["target_segmentation"]
    print("Cells and nuclei: ", target_segmentation)
    if output_channel_names is None:
        if target_segmentation == "N":
            output_channel_names = ["nuclei"]
        elif target_segmentation == "C":
            output_channel_names = ["cells"]
        elif target_segmentation == "NC":
            output_channel_names = ["nucei", "cells"]
        else:
            output_channel_names = ["nuclei"]
        print("Assuming output channel names: ", output_channel_names)

    Augmenter=Augmentations()

    input_tensor,_ = Augmenter.to_tensor(input_data,normalize=False) #this converts the input data to a tensor and does percentile normalization (no clipping)
    import math
    if math.isnan(model_pixel_size):
        model_pixel_size_tmp = pixel_size
    else:
        model_pixel_size_tmp = model_pixel_size
    input_crop,_ = Augmenter.torch_rescale(input_tensor,labels=None,current_pixel_size=pixel_size,requested_pixel_size=model_pixel_size_tmp,crop = True, random_seed=1)
    input_crop = input_crop.unsqueeze(0) # add batch dimension
    if input_crop.shape[1] != dim_in and not model_dict["channel_invariant"]:
        input_crop = torch.zeros((1,dim_in,input_crop.shape[2],input_crop.shape[3]),dtype=torch.float32, device = input_crop.device)

    print("Input tensor shape: ", input_crop.shape)

    np.save(os.path.join(output_name, "test-input.npy"), input_crop.float().numpy())

    input_crop,_ = Augmenter.to_tensor(input_crop[0],normalize=True)
    input_crop = input_crop.unsqueeze(0)

    with torch.no_grad():
        output = torchsript(input_crop.to(device))

        if isinstance(output, tuple):
            assert len(output) == len(output_types)
            for i, out in enumerate(output):
                np.save(os.path.join(output_name, f"test-output_{output_types[i]}.npy"), out.cpu().numpy())
        else:
            np.save(os.path.join(output_name, f"test-output_{output_types[0]}.npy"), output.cpu().numpy())

    from instanseg.utils.utils import _display_overlay

    if isinstance(output, tuple):
        output = output[0]

    cover = _display_overlay(input_crop[0], output)
    show_images(cover, colorbar=False)
    show_images(cover, colorbar=False, save_str= os.path.join(output_name, "cover"))

    if model_dict is not None and "source_dataset" in model_dict.keys():
        train_data = str(model_dict["source_dataset"])
    else:
        train_data = "Not specified"

    # create readme
    readme(output_name, model_dict)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    if os.path.exists(os.path.join(output_path, output_name + ".zip")):
        print("Removing existing model zip")
        os.remove(os.path.join(output_path, output_name + ".zip"))

    from bioimageio.spec.model.v0_5 import (
        AxisId,
        BatchAxis,
        ChannelAxis,
        FileDescr,
        Identifier,
        InputTensorDescr,
        IntervalOrRatioDataDescr,
        ParameterizedSize,
        SpaceInputAxis,
        SpaceOutputAxis,
        IndexInputAxis,
        TensorId,
        WeightsDescr,
        ScaleRangeDescr,
    )

    if model_dict["channel_invariant"]:
        input_axes = [BatchAxis(), IndexInputAxis( description = "Channel axis for channel invariant models", size=ParameterizedSize(min=1, step=1))]
    else:
        input_axes = [BatchAxis(), ChannelAxis(id = AxisId("channel"),channel_names=[Identifier(i) for i in ["C1","C2","C3"]])]

    if input_crop.ndim == 5: 
        input_axes += [
            SpaceInputAxis(id=AxisId("z"), size=ParameterizedSize(min=32, step=1)),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=32, step=1)),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=32, step=1)),
        ]
        data_descr = IntervalOrRatioDataDescr(type="float32")
    elif input_crop.ndim == 4: 
        input_axes += [
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=32, step=1),scale = model_pixel_size_tmp,unit = "micrometer",),

            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=32, step=1),scale = model_pixel_size_tmp,unit = "micrometer",),
        ]
        data_descr = IntervalOrRatioDataDescr(type="float32")
    else:
        raise NotImplementedError(
            f"Recreating inputs for {input_crop.shape} is not implemented"
        )
    
    preprocessing = [ScaleRangeDescr(kwargs = {"min_percentile": 0.1, "max_percentile": 99.9, "eps": 1e-6, "axes": ['x','y']})]

    input_descr = InputTensorDescr(
        id=TensorId("raw"),
        axes=input_axes,
        test_tensor=FileDescr(source=os.path.join(output_name, "test-input.npy"),),
        data=data_descr,
        preprocessing=preprocessing,

    )

    def get_instance_segmentation_descriptor():
        from bioimageio.spec.model.v0_5 import OutputTensorDescr, SizeReference

        output_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(n) for n in output_channel_names]
            ),
        ]

        if output.ndim == 5: 
            output_axes += [
                SpaceOutputAxis(
                    id=AxisId("z"),
                    size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("z")),
                    scale=model_pixel_size_tmp,
                    unit="micrometer",
                ),  # same size as input (tensor `raw`) axis `z`
                SpaceOutputAxis(
                    id=AxisId("y"),
                    size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")),
                    scale=model_pixel_size_tmp,
                    unit="micrometer",
                ),
                SpaceOutputAxis(
                    id=AxisId("x"),
                    size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")),
                    scale=model_pixel_size_tmp,
                    unit="micrometer",
                ),
            ]
        elif output.ndim == 4:
            output_axes += [
                SpaceOutputAxis(
                    id=AxisId("y"),
                    size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")),
                    scale=model_pixel_size_tmp,
                    unit="micrometer",
                ),  # same size as input (tensor `raw`) axis `y`
                SpaceOutputAxis(
                    id=AxisId("x"),
                    size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")),
                    scale=model_pixel_size_tmp,
                    unit="micrometer",
                ),
            ]
        else:
            raise NotImplementedError(
                f"Recreating outputs for { output.ndim} is not implemented"
            )

        output_descr = OutputTensorDescr(
            id=TensorId("instance_segmentation"),
            axes=output_axes,
            test_tensor=FileDescr(source=os.path.join(output_name, "test-output_instance_segmentation.npy"),),
        )

        return output_descr
    

    def get_detection_classes_descriptor():

        from bioimageio.spec.model.v0_5 import IndexOutputAxis, DataDependentSize, OutputTensorDescr,NominalOrOrdinalDataDescr

        output_axes = [
           # BatchAxis(),
            IndexOutputAxis(id = "n_objects", 
                            description = "Number of detected objects",
                            size = DataDependentSize()),
            IndexOutputAxis(id = "n_classes",
                            description = "Number of classes",
                            size = 1,),
            
        ]

        output_descr = OutputTensorDescr(
            id=TensorId("detection_classes"),
            axes=output_axes,
            test_tensor=FileDescr(source=os.path.join(output_name, "test-output_detection_classes.npy"),),
            data = NominalOrOrdinalDataDescr(type = "uint8",values = ["fancy_class_" + str(i) for i in range(19)]),
        )

        return output_descr
    

    def get_detection_logits_descriptor():

        from bioimageio.spec.model.v0_5 import IndexOutputAxis, DataDependentSize, OutputTensorDescr,NominalOrOrdinalDataDescr

        output_axes = [
           # BatchAxis(),
            IndexOutputAxis(id = "n_objects", 
                            description = "Number of detected objects",
                            size = DataDependentSize()),
            IndexOutputAxis(id = "logits",
                            description = "Logits for classes",
                            size = 19,),
            
        ]

        output_descr = OutputTensorDescr(
            id=TensorId("detection_logits"),
            axes=output_axes,
            test_tensor=FileDescr(source=os.path.join(output_name, "test-output_detection_classes.npy"),),
            data = NominalOrOrdinalDataDescr(type = "float32", values = [0]),
        )

        return output_descr
    

    outputs = []
    for output_type in output_types:
        if output_type == "instance_segmentation":
            outputs.append(get_instance_segmentation_descriptor())
        elif output_type == "detection_classes":
            outputs.append(get_detection_classes_descriptor())
        elif output_type == "detection_logits":
            outputs.append(get_detection_logits_descriptor())
        else:
            raise NotImplementedError(f"Output type {output_type} is not implemented")



    from bioimageio.spec.model.v0_5 import (
        Author,
        CiteEntry,
        Doi,
        HttpUrl,
        LicenseId,
        TorchscriptWeightsDescr,
    )

    my_model_descr = ModelDescr(
            name=output_name,
            version =version,
            description="An InstanSeg model trained for instance segmentation",
            authors=[
                Author(name="Thibaut Goldsborough", affiliation="School of Informatics, University of Edinburgh", github_user="ThibautGoldsborough")
            ],
            cite=[
                CiteEntry(text="If you use InstanSeg for nucleus segmentation of brightfield histology images, please cite:", doi=Doi("10.48550/arXiv.2408.15954")),
                CiteEntry(text="If you use InstanSeg for nucleus and/or cell segmentation in fluorescence images, please cite:", doi=Doi("10.1101/2024.09.04.611150"))
            ],
            license=LicenseId("Apache-2.0"),
            documentation=HttpUrl(
                "https://github.com/instanseg/instanseg/blob/main/README.md"
            ),
            git_repo=HttpUrl(
                "https://github.com/instanseg/instanseg"
            ), 
            inputs=[input_descr], 
            outputs=outputs, 
            weights=WeightsDescr(
            torchscript=TorchscriptWeightsDescr(
                source=os.path.join(output_name, "instanseg.pt"),
                pytorch_version="2.0.0",
            ),

            
        ),

        )
    print(f"created '{my_model_descr.name}'")

    from pathlib import Path

    from bioimageio.spec import save_bioimageio_package

    print(
        "package path:",
        save_bioimageio_package(my_model_descr, output_path=Path(os.path.join(output_path, output_name + ".zip"))),
    )

    from bioimageio.core import test_model

    summary = test_model(my_model_descr)
    summary.display()

    # #Cleanup 

    # files_to_remove = [
    #     "cover.png",
    #     "test-output.npy",
    #     "test-input.npy",
    #     "instanseg.pt",
    #     output_name + "_README.md",
    #     "sample_output_0.tif",
    #     "sample_input_0.tif"
    # ]

    # for file in files_to_remove:
    #     if os.path.exists(file):
    #         os.remove(file)

    # #unzip the folder

    # import zipfile
    # import shutil

    # input = os.path.join(output_path, output_name + ".zip")
    # destination = os.path.join(output_path, output_name)
    # with zipfile.ZipFile(input, 'r') as zip_ref:
    #     zip_ref.extractall(destination)
    
    # yaml_path = os.path.join(destination, 'rdf.yaml')
    # modify_yaml_for_qupath_config(yaml_path, pixel_size=model_pixel_size, dim_in=dim_in, dim_out=dim_out, version=version)

    
    # make_archive(destination, input)

    # shutil.rmtree(output_name)
