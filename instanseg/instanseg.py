from typing import Union, List, Optional
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from pathlib import Path, PosixPath

import sys
import pdb


def _to_tensor_float32(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, np.ndarray):      
        if image.dtype == np.uint16:
            image = image.astype(np.int32)
        image = torch.from_numpy(image).float()
    
    image = image.squeeze()

    assert image.dim() <= 3 and image.dim() >= 2, f"Input image shape {image.shape()} is not supported."

    image = torch.atleast_3d(image)
    channel_index = np.argmin(image.shape) #Note, this could break for small, highly multiplexed images.
    if channel_index != 0:
        image = image.movedim(channel_index, 0)

    return image

def _rescale_to_pixel_size(image: torch.Tensor, 
                           requested_pixel_size: float, 
                           model_pixel_size: float) -> torch.Tensor:

    if image.squeeze().dim() == 3:
        not_batched = True
        image = image[None]
    else:
        not_batched = False

    scale_factor = requested_pixel_size / model_pixel_size

    if not np.allclose(scale_factor,1, 0.01):
        image = interpolate(image, scale_factor=scale_factor, mode="bilinear")
    
    if not_batched:
        image = image[0]

    return image

def display_colourized(mIF):
    from instanseg.utils.utils import _move_channel_axis, generate_colors

    mIF = _to_tensor_float32(mIF)
    mIF = mIF / (mIF.max() + 1e-6)
    if mIF.shape[0]!=3:
        colours = generate_colors(num_colors=mIF.shape[0])
        colour_render = (mIF.flatten(1).T @ torch.tensor(colours)).reshape(mIF.shape[1],mIF.shape[2],3)
    else:
        colour_render = mIF
    colour_render = torch.clamp_(colour_render, 0, 1)
    colour_render = _move_channel_axis(colour_render,to_back = True).detach().numpy()*255
    return colour_render.astype(np.uint8)


class InstanSeg():
    """
    Main class for running InstanSeg.
    """
    def __init__(self, 
                 model_type: Union[str,nn.Module] = "brightfield_nuclei", 
                 device: Optional[str] = None, 
                 image_reader: str = "tiffslide",
                verbosity = 1, #0,1,2
                ):
        
        """
        Parameters:
            model_type (Union[str,nn.Module])
                The type of model to use. If a string is provided, the model will be downloaded. 
                If the model is not public, it will look for a model in your bioimageio folder. 
                If an nn.Module is provided, this model will be used.

            device (Optional[str]): 
                The device to run the model on. If None, the device will be chosen automatically.

            image_reader (str): 
                The image reader to use. Options are "tiffslide", "skimage.io", "bioio", "AICSImageIO".

            verbosity (int): 
                The verbosity level. 0 is silent, 1 is normal, 2 is verbose.
        
        """
        from instanseg.utils.utils import download_model, _choose_device

        self.verbosity = verbosity
        self.verbose = verbosity != 0

        if isinstance(model_type, nn.Module):
            self.instanseg = model_type
        else:
            self.instanseg = download_model(model_type, verbose = self.verbose)
        self.inference_device = _choose_device(device, verbose= self.verbose)
        self.instanseg = self.instanseg.to(self.inference_device)

        self.prefered_image_reader = image_reader
        self.small_image_threshold = 3 * 1500 * 1500 #max number of image pixels to be processed on GPU.
        self.medium_image_threshold = 10000 * 10000 #max number of image pixels that could be loaded in RAM.
        self.prediction_tag = "_instanseg_prediction"

    def read_image(self, image_str: str):
        if self.prefered_image_reader == "tiffslide":
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            width,height = slide.dimensions[0], slide.dimensions[1]
            num_pixels = width * height
            if num_pixels < self.medium_image_threshold:
                image_array = slide.read_region((0, 0), 0, (width, height), as_array=True)
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "skimage.io":
            from skimage.io import imread
            image_array = imread(image_str)
            img_pixel_size = None

        elif self.prefered_image_reader == "bioio":
            from bioio import BioImage
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            if num_pixels < self.medium_image_threshold:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size

        elif self.prefered_image_reader == "AICSImageIO":
            from aicsimageio import AICSImage
            slide = AICSImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            if num_pixels < self.medium_image_threshold:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size
        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented.")
        
        if img_pixel_size is None or float(img_pixel_size) < 0 or float(img_pixel_size) > 2:
            img_pixel_size = self.read_pixel_size(image_str)

        if img_pixel_size is not None:
            import warnings
            if float(img_pixel_size) <= 0 or float(img_pixel_size) > 2:
                warnings.warn(f"Pixel size {img_pixel_size} microns per pixel is invalid.")
                img_pixel_size = None

        return image_array, img_pixel_size
    
    def read_pixel_size(self,image_str: str):
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        from aicsimageio import AICSImage 
        try:
            
            slide = AICSImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        from bioio import BioImage
        try:
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        import slideio
        try:
            slide = slideio.open_slide(image_str, driver = "AUTO")
            scene  = slide.get_scene(0)
            img_pixel_size = scene.resolution[0] * 10**6

            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                    
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        print("Could not read pixel size from image metadata.")
        
        return None

    
    def read_slide(self, image_str: str):      
        if self.prefered_image_reader == "tiffslide":
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
        # elif self.prefered_image_reader == "AICSImageIO":
        #     from aicsimageio import AICSImage
        #     slide = AICSImage(image_str)
        # elif self.prefered_image_reader == "bioio":
        #     from bioio import BioImage
        #     slide = BioImage(image_str)
        # elif self.prefered_image_reader == "slideio":
        #     import slideio
        #     slide = slideio.open_slide(image_str, driver = "AUTO")

        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented for whole slide images.")
        return slide
    
    def _to_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return _to_tensor_float32(image)
    
    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        from instanseg.utils.utils import percentile_normalize, _move_channel_axis
        assert image.ndim == 3 or image.ndim == 4, f"Input image shape {image.shape} is not supported."
        if image.dim() == 3:
            image = percentile_normalize(image)
            image = image[None]
        else:
            image = torch.stack([percentile_normalize(i) for i in image])

        return image

    def eval(self, image: Union[str, List[str]], 
            pixel_size: Optional[float] = None,
            save_output: bool = False,
            save_overlay: bool = False,
            save_geojson: bool = False,
            **kwargs):
        """
        Evaluate the input image or list of images using the InstanSeg model.
        """

        if isinstance(image, PosixPath):
            image = str(image)
        if isinstance(image, str):
            initial_type = "not_list"
            image_list = [image]
        else:
            initial_type = "list"
            image_list = image
        
        output_list = []
    
        for image in image_list:
            image_array, img_pixel_size = self.read_image(image)

            if img_pixel_size is None and pixel_size is not None:
                img_pixel_size = pixel_size
            if img_pixel_size is None:
                import warnings
                warnings.warn("Pixel size not provided and could not be read from image metadata, this may lead to innacurate results.")
                
            if not isinstance(image_array, str):
                
                num_pixels = np.cumprod(image_array.shape)[-1]
                if num_pixels < self.small_image_threshold:
                    instances = self.eval_small_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)

                else:
                    instances = self.eval_medium_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)

                output_list.append(instances)

                if save_output:
                    self.save_output(image, instances, image_array = image_array, save_overlay = save_overlay, save_geojson = save_geojson)
     
                    
            else:
                self.eval_whole_slide_image(image_array, pixel_size, **kwargs)

        if initial_type == "not_list":
            output = output_list[0]
        else:
            output = output_list

        
        return output
    
    def save_output(self,image_path: str, 
                    labels: torch.Tensor,
                    image_array: Optional[np.ndarray] = None,
                    save_overlay = False,
                    save_geojson = False,):

        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        new_stem = image_path.stem + self.prediction_tag

        from skimage import io

        if self.verbose:

            out_path = Path(image_path).parent / (new_stem + ".tiff")
            print(f"Saving output to {out_path}")
            io.imsave(out_path, labels.squeeze().astype(np.int32), check_contrast=False)

        if save_geojson:
            if labels.ndim == 3:
                labels = labels[None]

            output_dimension = labels.shape[1]
            from instanseg.utils.utils import labels_to_features
            import json
            if output_dimension == 1:
                features = labels_to_features(labels[0,0],object_type = "detection")

            elif output_dimension == 2:
                features = labels_to_features(labels[0,0],object_type = "detection",classification="Nuclei") + labels_to_features(labels[0,1],object_type = "detection",classification = "Cells")
            geojson = json.dumps(features)

            geojson_path = Path(image_path).parent / (new_stem + ".geojson")
            with open(os.path.join(geojson_path), "w") as outfile:
                outfile.write(geojson)
        
        if save_overlay:

            if self.verbose:
                out_path = Path(image_path).parent / (new_stem + "_overlay.tiff")
                print(f"Saving overlay to {out_path}")
            assert image_array is not None, "Image array must be provided to save overlay."
            display = self.display(image_array, labels)
            
            io.imsave(out_path, display, check_contrast=False)


            

    def eval_small_image(self,image: torch.Tensor, pixel_size: Optional[float] = None,
                          normalise: bool = True,
                          return_image_tensor: bool = True,
                          target = "all_outputs", #or "nuclei" or "cells"
                          rescale_output: bool = True,
                          **kwargs):
        """
        Evaluate the input image using the InstanSeg model.
        
        Args:
            image (torch.Tensor): The input image(s) to be evaluated.
            pixel_size (Optional[float]): The pixel size of the image. If not provided, it will be read from the image metadata.
        """
        from instanseg.utils.utils import percentile_normalize, _move_channel_axis

        image = _to_tensor_float32(image)

        original_shape = image.shape

        if pixel_size is not None:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False

        image = image.to(self.inference_device)

        assert image.dim() ==3 or image.dim() == 4, f"Input image shape {image.shape} is not supported."

        if normalise:
            if image.dim() == 3:
                image = percentile_normalize(image)
                image = image[None]
            else:
                image = torch.stack([percentile_normalize(i) for i in image])

        
        if target != "all_outputs" and self.instanseg.cells_and_nuclei:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0])
            else:
                target_segmentation = torch.tensor([0,1])
        else:
            target_segmentation = torch.tensor([1,1])

        with torch.amp.autocast('cuda'):

        
            instanseg_kwargs = {k: v for k, v in kwargs.items() if k not in ["batch_size", "tile_size", "normalisation_subsampling_factor"]}
            instances = self.instanseg(image,target_segmentation = target_segmentation, **instanseg_kwargs)

        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")
        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()

    def eval_medium_image(self,image: torch.Tensor, 
                          pixel_size: Optional[float] = None, 
                          normalise: bool = True,
                          tile_size: int = 512,
                          batch_size: int = 1,
                        return_image_tensor: bool = True,
                        normalisation_subsampling_factor: int = 1,
                        target = "all_outputs", #or "nuclei" or "cells"
                        rescale_output: bool = True,
                          **kwargs):
        
        from instanseg.utils.utils import percentile_normalize, _move_channel_axis

        image = _to_tensor_float32(image)
        
        from instanseg.utils.tiling import sliding_window_inference
        original_shape = image.shape

        if pixel_size is None:
            import warnings
            warnings.warn("Pixel size not provided, this may lead to innacurate results.")
        else:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False
        assert image.dim() ==3, f"Input image shape {image.shape} is not supported."
        
        if normalise:
            image = percentile_normalize(image, subsampling_factor=normalisation_subsampling_factor)
            image = image

        output_dimension = 2 if self.instanseg.cells_and_nuclei else 1

        if target != "all_outputs" and output_dimension == 2:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0])
            else:
                target_segmentation = torch.tensor([0,1])
            output_dimension = 1
        else:
            target_segmentation = torch.tensor([1,1])

        instances = sliding_window_inference(image,
                    self.instanseg, 
                    window_size = (tile_size,tile_size),
                    sw_device = self.inference_device,
                    device = 'cpu', 
                    batch_size= batch_size,
                    output_channels = output_dimension,
                    show_progress= self.verbose,
                    target_segmentation = target_segmentation,
                    **kwargs).float()
        
        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()
        
    def eval_whole_slide_image(self, image: str, pixel_size: Optional[float] = None, 
                               normalise: bool = True, 
                               batch_size: int = 1,
                               tile_size: int = 512,
                               output_geojson: bool = False,
                               **kwargs):
            from instanseg.utils.tiling import segment_image_larger_than_memory
                                 
            segment_image_larger_than_memory(instanseg_class= self,
                                            image_path= image, 
                                            memory_block_size = (int(self.medium_image_threshold**0.5),int(self.medium_image_threshold **0.5)),  #this is the size of the image that will be read in memory
                                            inference_tile_size = (tile_size,tile_size), #this is the size of the image that will be passed to the model
                                            batch_size= batch_size,
                                            to_geojson= output_geojson, 
                                            pixel_size =pixel_size,
                                            prediction_tag = self.prediction_tag, 
                                            sw_device = self.inference_device,
                                            normalisation_subsling_factor=10,
                                            **kwargs)
            
    
    def display(self, image: torch.tensor,
               instances: torch.Tensor):
        
        from instanseg.utils.utils import save_image_with_label_overlay

        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        im_for_display = display_colourized(image.squeeze())

        output_dimension = instances.shape[1]

        if output_dimension ==1: #Nucleus or cell mask
            labels_for_display = instances[0,0] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=labels_for_display,return_image=True, label_boundary_mode="thick", label_colors=None,thickness=10,alpha=0.9)
        elif output_dimension ==2: #Nucleus and cell mask
            nuclei_labels_for_display = instances[0,0]
            cell_labels_for_display = instances[0,1] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=nuclei_labels_for_display,return_image=True, label_boundary_mode="thick", label_colors="red",thickness=10)
            image_overlay = save_image_with_label_overlay(image_overlay,lab=cell_labels_for_display,return_image=True, label_boundary_mode="inner", label_colors="green",thickness=1)

        return image_overlay
    

    def cluster_instances_by_mean_channel_intensity(self, image_tensor: torch.Tensor, labeled_output: torch.Tensor):

        #This is experimental code that is not yet implemented. You'll need to install rapids_singlecell, cuml and scanpy to run this code.

        from instanseg.utils.biological_utils import get_mean_object_features
        import fastremap
        import numpy as np
        from instanseg.utils.utils import apply_cmap
        from instanseg.utils.pytorch_utils import torch_fastremap
        import rapids_singlecell as rsc
        import scanpy as sc
        import matplotlib.pyplot as plt

        X_features = get_mean_object_features( image_tensor.to("cuda"), labeled_output.to("cuda"),)

        adata = sc.AnnData(X_features.cpu().numpy())
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.scale(adata)
        rsc.pp.neighbors(adata, n_neighbors=8, n_pcs=50)
        rsc.tl.umap(adata)
        rsc.tl.leiden(adata, resolution=0.1)

        # Create the UMAP plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        mapping = fastremap.component_map(np.arange(1, len(adata.obs["leiden"]) + 1), adata.obs["leiden"].astype(np.int64) + 1)
        labs = torch_fastremap(labeled_output[0, 0])
        labels = fastremap.remap(labs.numpy(), mapping, preserve_missing_labels=True)

        labels_disp = apply_cmap(labels, labels > 0, cmap="tab10")

        # Show the labeled image
        axes[0].imshow(labels_disp)
        axes[0].set_title('Labeled Image')
        axes[0].axis('off')

        sc.pl.umap(adata, color="leiden", legend_loc='on data', cmap="tab10", title='UMAP with Leiden Clustering', s=30, ax=axes[1], show = False)
        axes[1].axis('off')
        plt.subplots_adjust(wspace=0., hspace=0)
        plt.show()


            
if __name__ == "__main__":


    import os
    import pdb
    sys.path = sys.path[1:]
    from instanseg.utils.utils import show_images

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"./examples/"))

    instanseg_brightfield = InstanSeg("brightfield_nuclei")


    image_array, pixel_size = instanseg_brightfield.read_image(example_image_folder/"HE_example.tif")

    labeled_output, image_tensor  = instanseg_brightfield.eval_small_image(image_array, pixel_size)

    display = instanseg_brightfield.display(image_tensor, labeled_output)
    show_images(image_tensor,display, colorbar=False)


    instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells")
    image_array, pixel_size = instanseg_fluorescence.read_image("../instanseg/examples/LuCa1.tif")

    labeled_output, image_tensor  = instanseg_fluorescence.eval_small_image(image_array, pixel_size)

    display = instanseg_fluorescence.display(image_tensor, labeled_output)

    show_images(display, colorbar=False)

   # instanseg = InstanSeg("fluorescence_nuclei_and_cells")
   # image_array,pixel_size = instanseg.read_image(example_image_folder / "adam.ome.tif")

   # labeled_output, image_tensor  = instanseg.eval_medium_image(image_array, pixel_size, normalisation_subsampling_factor=10, batch_size=3)

   # instanseg.save_output(example_image_folder / "adam.ome.tif", labeled_output, image_array, output_overlay=True, output_geojson=True)

    # instanseg = InstanSeg("brightfield_nuclei")
    # # instances = instanseg.eval_whole_slide_image(example_image_folder / "HE_Hamamatsu.tiff")

    # instanseg = InstanSeg("brightfield_nuclei")
    # instances = instanseg.eval(example_image_folder / "HE_example.tif")

    # instanseg.prefered_image_reader = "skimage.io"
    # instances = instanseg.eval(example_image_folder / "HE_example.tif")

    # instanseg.prefered_image_reader = "AICSImageIO"
    # instances = instanseg.eval(example_image_folder / "HE_example.tif")

    # instanseg = InstanSeg("fluorescence_nuclei_and_cells")
    # instances = instanseg.eval(example_image_folder / "LuCa1.tif", batch_size=3)

    # show_images(instances)

        



        














 

