from typing import Union, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from pathlib import Path, PosixPath
from tiffslide import TiffSlide
import zarr
import os


def _to_ndim(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Ensure that the input tensor has the desired number of dimensions.
    If the input tensor has fewer dimensions, it will be unsqueezed.
    If the input tensor has more dimensions, it will be squeezed.
    If the input tensor has the desired number of dimensions, it will be returned as is.
    
    Args:
        x (torch.Tensor): The input tensor.
        n (int): The desired number of dimensions.
        
    Returns:
        torch.Tensor: The input tensor with the desired number of dimensions.
    """
    if x.dim() == n:
        return x
    if x.dim() > n:
        x = x.squeeze()
    x = x[(None,) * (n - x.dim())]
    if x.dim() != n:
        raise ValueError(f"Input tensor has shape {x.shape}, which is not compatible with the desired dimension {n}.")
    return x

def _to_tensor_float32(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert the input image to a PyTorch tensor with float32 data type.
    If the input is a NumPy array, it will be converted to a PyTorch tensor.
    The tensor will be squeezed to remove any singleton dimensions.
    The channel dimension will be moved to the first position if it is not already there.
    
    Args:
        image (Union[np.ndarray, torch.Tensor]): The input image, which can be either a NumPy array or a PyTorch tensor.
        
    Returns:
        torch.Tensor: The input image as a PyTorch tensor with float32 data type and the channel dimension in the first position.
    """

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
    
    original_dim = image.dim()

    image = _to_ndim(image, 4)

    scale_factor = requested_pixel_size / model_pixel_size

    if not np.allclose(scale_factor,1, 0.01):
        image = interpolate(image, scale_factor=scale_factor, mode="bilinear")

    return _to_ndim(image, original_dim)
    

def _display_colourized(mIF):
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
                 verbosity: int = 1 #0,1,2
                 ):
        
        """
        :param model_type: The type of model to use. If a string is provided, the model will be downloaded. If the model is not public, it will look for a model in your bioimageio folder. If an nn.Module is provided, this model will be used.
        :param device: The device to run the model on. If None, the device will be chosen automatically.
        :param image_reader: The image reader to use. Options are "tiffslide", "skimage.io", "bioio", "AICSImageIO".
        :param verbosity: The verbosity level. 0 is silent, 1 is normal, 2 is verbose.
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

    def read_image(self, image_str: str) -> Union[Tuple[str, float], Tuple[np.ndarray, float]]:
        """
        Read an image file from disk.
        :param image_str: The path to the image.
        :return: The image array if it can be safely read (or the path to the image if it cannot) and the pixel size in microns.
        """
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
    
    def read_pixel_size(self,image_str: str) -> float:
        """
        Read the pixel size from an image on disk.
        :param image_str: The path to the image.
        :return: The pixel size in microns.
        """
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
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
        """
        Read a whole slide image from disk.
        :param image_str: The path to the image.
        """
        if self.prefered_image_reader == "tiffslide":
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

    def eval(self,
             image: Union[str, List[str]], 
             pixel_size: Optional[float] = None,
             save_output: bool = False,
             save_overlay: bool = False,
             save_geojson: bool = False,
             **kwargs) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """
        Evaluate the input image or list of images using the InstanSeg model.
        :param image: The path to the image, or a list of such paths.
        :param pixel_size: The pixel size in microns.
        :param save_output: Controls whether the output is saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_overlay: Controls whether the output is saved to disk as an overlay (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_geojson: Controls whether the geojson output labels are saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param kwargs: Passed to other eval methods, eg :func:`save_output <instanseg.Instanseg.eval_small_image>`, :func:`save_output <instanseg.Instanseg.eval_medium_image>`, :func:`save_output <instanseg.Instanseg.eval_whole_slide_image>` 
        :return: A torch.Tensor of outputs if the input is a path to a single image, or a list of such outputs if the input is a list of paths, or None if the input is a whole slide image.
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
    
    def save_output(self,
                    image_path: str, 
                    labels: torch.Tensor,
                    image_array: Optional[np.ndarray] = None,
                    save_overlay = False,
                    save_geojson = False) -> None:
        """
        Save the output of InstanSeg to disk.
        :param image_path: The path to the image, and where outputs will be saved.
        :param labels: The output labels.
        :param image_array: The image in array format. Required to save overlay.
        :param save_overlay: Save the labels overlaid on the image.
        :param save_geojson: Save the labels as a GeoJSON feature collection.
        """
        import os

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

            labels = _to_ndim(labels, 4)
        
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



    def eval_small_image(self,
                         image: torch.Tensor,
                         pixel_size: Optional[float] = None,
                         normalise: bool = True,
                         return_image_tensor: bool = True,
                         target: str = "all_outputs", #or "nuclei" or "cells"
                         rescale_output: bool = True,
                         **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a small input image using the InstanSeg model.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """
        from instanseg.utils.utils import percentile_normalize

        image = _to_tensor_float32(image)

        image = _to_ndim(image, 4)

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
                image = _to_ndim(image, 4)
                image = torch.stack([percentile_normalize(i) for i in image]) #over the batch dimension

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

    def eval_medium_image(self,
                          image: torch.Tensor, 
                          pixel_size: Optional[float] = None, 
                          normalise: bool = True,
                          tile_size: int = 512,
                          batch_size: int = 1,
                          return_image_tensor: bool = True,
                          normalisation_subsampling_factor: int = 1,
                          target: str = "all_outputs", #or "nuclei" or "cells"
                          rescale_output: bool = True,
                          **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a medium input image using the InstanSeg model. The image will be split into tiles, and then inference and object merging will be handled internally.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param tile_size: The width/height of the tiles that the image will be split into.
        :param batch_size: The number of tiles to be run simultaneously.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """

        from instanseg.utils.utils import percentile_normalize, _move_channel_axis

    
        image = _to_tensor_float32(image)
        
        from instanseg.utils.tiling import _sliding_window_inference
        original_shape = image.shape
        original_ndim = image.dim()

        if pixel_size is None:
            import warnings
            warnings.warn("Pixel size not provided, this may lead to innacurate results.")
        else:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False
        

        image = _to_ndim(image, 3)
        
        if normalise:
            image = percentile_normalize(image, subsampling_factor=normalisation_subsampling_factor)
            

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

        instances = _sliding_window_inference(image,
                                              self.instanseg,
                                              window_size = (tile_size,tile_size),sw_device = self.inference_device,
                                              device = 'cpu', 
                                              batch_size= batch_size,
                                              output_channels = output_dimension,
                                              show_progress= self.verbose,
                                              target_segmentation = target_segmentation,
                                              **kwargs).float()

        instances = _to_ndim(instances, 4)
        image = _to_ndim(image, 4)
        
        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")
            instances = _to_ndim(instances, 4)

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        image = _to_ndim(image, original_ndim)

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()
        
    def eval_whole_slide_image(self,
                               image: str,
                               pixel_size: Optional[float] = None, 
                               normalise: bool = True,
                               normalisation_subsampling_factor: int = 10,
                               tile_size: int = 512,
                               overlap: int = 100,
                               detection_size: int = 20, 
                               batch_size: int = 1,
                               output_geojson: bool = False,
                               **kwargs):
            """
            Evaluate a whole slide input image using the InstanSeg model. This function uses slideio to read an image and then segments it using the instanseg model. The segmentation is done in a tiled manner to avoid memory issues. 
            
            :param image: The input image to be evaluated.
            :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
            :param normalise: Controls whether the image is normalised.
            :param tile_size: The width/height of the tiles that the image will be split into.
            :param overlap: The overlap (in pixels) betwene tiles.
            :param detection_size: The expected maximum size of detection objects.
            :param batch_size: The number of tiles to be run simultaneously.
            :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
            :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
            :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
            :param kwargs: Passed to pytorch.
            :return: Returns a zarr file with the segmentation. The zarr file is saved in the same directory as the image with the same name but with the extension .zarr.
            """

            memory_block_size =  (int(self.medium_image_threshold**0.5), int(self.medium_image_threshold **0.5))
            inference_tile_size = (tile_size,tile_size)

            from itertools import product
            from instanseg.utils.pytorch_utils import torch_fastremap, match_labels
            from pathlib import Path
            from tqdm import tqdm
            from instanseg.utils.tiling import _chops, _remove_edge_labels, _zarr_to_json_export
            
            instanseg = self.instanseg

            image, img_pixel_size = self.read_image(image)
            slide = self.read_slide(image)

            n_dim = 2 if instanseg.cells_and_nuclei else 1
            model_pixel_size = instanseg.pixel_size

            new_stem = Path(image).stem + self.prediction_tag
            file_with_zarr_extension = Path(image).parent / (new_stem + ".zarr")


            if img_pixel_size > 1 or img_pixel_size < 0.1:
                import warnings
                warnings.warn("The image pixel size {} is not in microns.".format(img_pixel_size))
                if pixel_size is not None:
                    img_pixel_size = pixel_size
                else:
                    raise ValueError("The image pixel size {} is not in microns.".format(img_pixel_size))
            
            scale_factor = model_pixel_size/img_pixel_size

            dims = slide.dimensions
            dims = (int(dims[1]/ scale_factor), int(dims[0]/scale_factor)) #The dimensions are opposite to numpy/torch/zarr dimensions.

            pad2 = overlap + detection_size
            pad = overlap

            shape = memory_block_size
            
            chop_list = _chops(dims, shape, overlap=2*pad2)

            chunk_shape = (n_dim,shape[0],shape[1])
            store = zarr.DirectoryStore(file_with_zarr_extension) 
            canvas = zarr.zeros((n_dim,dims[0],dims[1]), chunks=chunk_shape, dtype=np.int32, store=store, overwrite = True)

            running_max = 0

            total = len(chop_list[0]) * len(chop_list[1])
            for _, ((i, window_i), (j, window_j)) in tqdm(enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))), total=total, colour = "green", desc = "Slide progress: "):

            #  input_data = scene.read_block((int(window_j*scale_factor), int(window_i*scale_factor), int(shape[0]*scale_factor), int(shape[1]*scale_factor)), size = shape)

                best_level = slide.get_best_level_for_downsample(scale_factor)
                downsample_factor = slide.level_downsamples[best_level]

                initial_pixel_size = img_pixel_size
                itermediate_pixel_size = initial_pixel_size * downsample_factor
                final_pixel_size = model_pixel_size

                intermediate_to_final = final_pixel_size/itermediate_pixel_size
                
                # Calculate the size of the region needed at the base level to get the desired output size
                intermediate_shape = (int(shape[0] * intermediate_to_final), int(shape[1] * intermediate_to_final))

                input_data = slide.read_region((int(window_j*scale_factor), int(window_i*scale_factor)), best_level, (int(intermediate_shape[0]) , int(intermediate_shape[1])), as_array=True)
            
                input_tensor = self._to_tensor(input_data)

                new_tile = self.eval_medium_image(input_tensor,
                                                  pixel_size = itermediate_pixel_size,
                                                  tile_size = inference_tile_size[0],
                                                  batch_size = batch_size,
                                                  return_image_tensor = False,
                                                  normalise = normalise,
                                                  normalisation_subsampling_factor = normalisation_subsampling_factor,
                                                )
                
            

                from torch.nn.functional import interpolate
                new_tile = interpolate(new_tile, size=shape[-2:], mode="nearest").int()[0]

            # pdb.set_trace()

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
                        tile2 = _remove_edge_labels(new_tile[n,pad:shape[0],pad: shape[1]], ignore = ignore_list)

                    elif i == len(chop_list[0])-1:
                        tile1 = canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]]
                        tile2 = _remove_edge_labels(new_tile[n,pad:shape[0],pad : shape[1]], ignore =ignore_list)

                    elif j == len(chop_list[1])-1:
                        tile1 = canvas[n, window_i + pad:window_i + shape[0], window_j + pad:window_j + shape[1]]
                        tile2 = _remove_edge_labels(new_tile[n,pad:shape[0],pad: shape[1]], ignore = ignore_list)

                    elif i == 0 and j == 0:
                        tile1 = canvas[n, window_i  :window_i + shape[0], window_j :window_j + shape[1]]
                        tile2 = _remove_edge_labels(new_tile[n, :shape[0], : shape[1]], ignore = ignore_list)
                    elif i == 0:
                        tile1 = canvas[n, window_i  :window_i + shape[0], window_j + pad :window_j + shape[1]]
                        tile2 = _remove_edge_labels(new_tile[n, :shape[0],pad : shape[1]], ignore = ignore_list)

                    elif j == 0:
                        tile1 = canvas[n, window_i  + pad:window_i + shape[0], window_j:window_j + shape[1]]
                        tile2 = _remove_edge_labels(new_tile[n,pad :shape[0],: shape[1]], ignore = ignore_list)

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
                        tile2 = _remove_edge_labels(new_tile[n,pad:shape[0] -pad,pad: shape[1]-pad])
                        
                        tile2 = torch_fastremap(tile2)

                        tile2[tile2>0] = tile2[tile2>0] + running_max

                        tile1_torch = torch.tensor(np.array(tile1), dtype = torch.int32)
                        remapped = match_labels(tile1_torch, tile2, threshold = 0.1)[1]

                        tile1_torch[remapped>0] = remapped[remapped>0].int()
                        running_max = max(running_max, tile1_torch.max())
                    
                        canvas[n, window_i + pad:window_i + shape[0] - pad, window_j + pad:window_j + shape[1] - pad] = tile1_torch.numpy().astype(np.int32)

            if output_geojson:
                print("Exporting to geojson")
                _zarr_to_json_export(file_with_zarr_extension, detection_size = detection_size, size = shape[0], scale = scale_factor, n_dim = n_dim)
                    


    
    def display(self,
                image: torch.tensor,
                instances: torch.Tensor) -> np.ndarray:
        """
        Save the output of an InstanSeg model overlaid on the input.
        See :func:`save_image_with_label_overlay <instanseg.utils.save_image_with_label_overlay>` for more details and return types.
        :param image: The input image.
        :param instances: The output labels.
        """
        from instanseg.utils.utils import save_image_with_label_overlay

        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        im_for_display = _display_colourized(image.squeeze())

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
    

    def _cluster_instances_by_mean_channel_intensity(self, image_tensor: torch.Tensor, labeled_output: torch.Tensor):

        #This is experimental code that is not yet implemented. You'll need to install rapids_singlecell, cuml and scanpy to run this code.

        from instanseg.utils.biological_utils import get_mean_object_features
        import fastremap
        import numpy as np
        from instanseg.utils.utils import apply_cmap
        from instanseg.utils.pytorch_utils import torch_fastremap
        import rapids_singlecell as rsc
        import scanpy as sc
        import matplotlib.pyplot as plt

        labeled_output = _to_ndim(labeled_output, 4)
        image_tensor = _to_ndim(image_tensor, 3)

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

