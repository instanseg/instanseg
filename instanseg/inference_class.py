from typing import Union, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from pathlib import Path, PosixPath
from tiffslide import TiffSlide
import zarr
import os
from instanseg.utils.pytorch_utils import _to_tensor_float32
pixel_size_precision = 0.01
def _to_ndim(x, *args, **kwargs):
    from instanseg.utils.pytorch_utils import _to_ndim as _to_ndim_pytorch
    from instanseg.utils.pytorch_utils import _to_ndim_numpy
    if isinstance(x, torch.Tensor):
        return _to_ndim_pytorch(x, *args, **kwargs)
    elif isinstance(x, np.ndarray):
        return _to_ndim_numpy(x, *args, **kwargs)


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

    def read_image(self, image_str: str, processing_method = "auto") -> Union[Tuple[str, float], Tuple[np.ndarray, float]]:
        """
        Read an image file from disk.
        :param image_str: The path to the image.
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
        :return: The image array if it can be safely read (or the path to the image if it cannot) and the pixel size in microns.
        """
        if self.prefered_image_reader == "tiffslide":

            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            width,height = slide.dimensions[0], slide.dimensions[1]
            num_pixels = width * height

            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

            if eval_function_str in ["small","medium"]:
                image_array = slide.read_region((0, 0), 0, (width, height), as_array=True)
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "skimage.io":
            from skimage.io import imread
            assert processing_method != "wsi", "skimage.io does not support whole slide images."
            image_array = imread(image_str)
            img_pixel_size = None

        elif self.prefered_image_reader == "bioio":
            from bioio import BioImage
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "bioformats":
            from bioio import BioImage
            import bioio_bioformats
            slide = BioImage(image_str, reader=bioio_bioformats.Reader)
            channel_names = slide.channel_names
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]

            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.data.squeeze()
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
    
    def _get_eval_function_to_use(self,num_pixels, processing_method = "auto") -> str:

        if processing_method != "auto":
            assert processing_method in ["small", "medium", "wsi"], f"Processing method {processing_method} is not supported."
            return processing_method
        if num_pixels < self.small_image_threshold:
            return "small"
        elif num_pixels < self.medium_image_threshold:
            return "medium"
        else:
            return "wsi"

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
             processing_method: str = "auto", #auto, small, medium, wsi
             **kwargs) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """
        Evaluate the input image or list of images using the InstanSeg model.
        :param image: The path to the image, or a list of such paths.
        :param pixel_size: The pixel size in microns.
        :param save_output: Controls whether the output is saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_overlay: Controls whether the output is saved to disk as an overlay (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_geojson: Controls whether the geojson output labels are saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
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
            image_array, img_pixel_size = self.read_image(image, processing_method = processing_method)

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            if img_pixel_size is None and pixel_size is not None:
                img_pixel_size = pixel_size
            if img_pixel_size is None:
                import warnings
                warnings.warn("Pixel size not provided and could not be read from image metadata, this may lead to innacurate results.")
                
            if not isinstance(image_array, str):
                
                num_pixels = np.cumprod(image_array.shape)[-1]

                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

                if eval_function_str == "small":
                    instances = self.eval_small_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                    
                
                elif eval_function_str == "medium":
                    instances = self.eval_medium_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                
                else:
                    raise NotImplementedError(f"Processing method {eval_function_str} is not implemented for image array inputs.")


                if save_output or save_overlay or save_geojson:
                    self.save_output(image, instances, image_array = image_array, save_output = save_output, save_overlay = save_overlay, save_geojson = save_geojson)
       
            else:
                self.eval_whole_slide_image(image_array, pixel_size, save_geojson = save_geojson, **kwargs)
                output_list.append(None)

        if initial_type == "not_list":
            output = output_list[0]
        else:
            output = output_list
        
        return output
    
    def save_output(self,
                    image_path: str, 
                    labels: torch.Tensor,
                    image_array: Optional[np.ndarray] = None,
                    save_output: bool = True,
                    save_overlay = False,
                    save_geojson = False) -> None:
        """
        Save the output of InstanSeg to disk.
        :param image_path: The path to the image, and where outputs will be saved.
        :param labels: The output labels.
        :param image_array: The image in array format. Required to save overlay.
        :param save_output: Save the labels to disk.
        :param save_overlay: Save the labels overlaid on the image.
        :param save_geojson: Save the labels as a GeoJSON feature collection.
        """
        import os
        from skimage import io

        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        new_stem = image_path.stem + self.prediction_tag

        out_path = Path(image_path).parent / (new_stem + ".tiff")

        if save_output:
            if self.verbose:
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
                features = labels_to_features(labels[0,0],object_type = "detection",classification="Nuclei")["features"] + labels_to_features(labels[0,1],object_type = "detection",classification = "Cells")["features"]
            
            geojson = json.dumps(features)

            geojson_path = Path(image_path).parent / (new_stem + ".geojson")
            with open(os.path.join(geojson_path), "w") as outfile:
                if self.verbose:
                    print(f"Saving geojson to {geojson_path}")
                outfile.write(geojson)
        
        if save_overlay:

            out_path = Path(image_path).parent / (new_stem + "_overlay.tiff")

            if self.verbose:
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
        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

        image = _to_tensor_float32(image)

        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]

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
            instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
            instanseg_kwargs["target_segmentation"] = target_segmentation

            instances = self.instanseg(image, **instanseg_kwargs)

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

        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

    
        image = _to_tensor_float32(image)
        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]


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

        instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
        instanseg_kwargs["target_segmentation"] = target_segmentation


        instances = _sliding_window_inference(image,
                                              self.instanseg,
                                              window_size = (tile_size,tile_size),sw_device = self.inference_device,
                                              device = 'cpu', 
                                              batch_size= batch_size,
                                              output_channels = output_dimension,
                                              show_progress= self.verbose,
                                              instanseg_kwargs = instanseg_kwargs).float()

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
                               normalisation_subsampling_factor: int = 1,
                               tile_size: int = 512,
                               overlap: int = 80,
                               detection_size: int = 20, 
                               save_geojson: bool = False,
                               use_otsu_threshold: bool = False,
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
            :param use_otsu_threshold: bool = False. Whether to use an otsu threshold on the image thumbnail to find the tissue region.
            :param kwargs: Passed to pytorch.
            :return: Returns a zarr file with the segmentation. The zarr file is saved in the same directory as the image with the same name but with the extension .zarr.
            """

            memory_block_size = tile_size,tile_size

            from itertools import product
            from instanseg.utils.pytorch_utils import torch_fastremap, match_labels
            from pathlib import Path
            from tqdm import tqdm
            from instanseg.utils.tiling import _chops, _remove_edge_labels, _zarr_to_json_export
    
            instanseg = self.instanseg

            image, img_pixel_size = self.read_image(image, processing_method= "wsi")

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            slide = self.read_slide(image)

            n_dim = 2 if instanseg.cells_and_nuclei else 1
            model_pixel_size = instanseg.pixel_size

            new_stem = Path(image).stem + self.prediction_tag
            file_with_zarr_extension = Path(image).parent / (new_stem + ".zarr")

            if img_pixel_size is None or img_pixel_size > 1 or img_pixel_size < 0.1:
                import warnings
                warnings.warn("The image pixel size {} is not in microns.".format(img_pixel_size))
                if pixel_size is not None:
                    img_pixel_size = pixel_size
                else:
                    raise ValueError("The image pixel size {} is not in microns.".format(img_pixel_size))
            
            scale_factor = model_pixel_size/img_pixel_size

            dims = slide.dimensions
            dims = (round(dims[1]/ scale_factor), round(dims[0]/scale_factor)) #The dimensions are opposite to numpy/torch/zarr dimensions.

            pad2 = overlap + detection_size
            pad = overlap

            shape = memory_block_size
            chop_list = _chops(dims, shape, overlap=2*pad2)

            if use_otsu_threshold:
                mask,_ = _threshold_thumbnail(slide)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
            else:
                valid_positions = np.ones((len(chop_list[0])* len(chop_list[1])))

            chunk_shape = (n_dim,shape[0],shape[1])
            store = zarr.DirectoryStore(file_with_zarr_extension) 
            canvas = zarr.zeros((n_dim,dims[0],dims[1]), chunks=chunk_shape, dtype=np.int32, store=store, overwrite = True)

            running_max = 0

            total = len(chop_list[0]) * len(chop_list[1])
            counter = -1
            for _, ((i, window_i), (j, window_j)) in tqdm(enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))), total=total, colour = "green", desc = "Slide progress: "):
                counter += 1
                if valid_positions[counter] == 0:
                    continue

            #  input_data = scene.read_block((round(window_j*scale_factor), round(window_i*scale_factor), round(shape[0]*scale_factor), round(shape[1]*scale_factor)), size = shape)

                best_level = slide.get_best_level_for_downsample(scale_factor)
                downsample_factor = slide.level_downsamples[best_level]

                initial_pixel_size = img_pixel_size
                intermediate_pixel_size = initial_pixel_size * downsample_factor
                final_pixel_size = model_pixel_size

                intermediate_to_final = intermediate_pixel_size / final_pixel_size
     
                # Calculate the size of the region needed at the base level to get the desired output size
                intermediate_shape = (round(shape[0] / intermediate_to_final), round(shape[1] / intermediate_to_final))

                input_data = slide.read_region((round(window_j*scale_factor), round(window_i*scale_factor)), best_level, (round(intermediate_shape[0]) , round(intermediate_shape[1])), as_array=True)
            
                input_tensor = self._to_tensor(input_data)

                new_tile = self.eval_small_image(input_tensor,
                                                  pixel_size = intermediate_pixel_size,
                                                  return_image_tensor= False,
                                                  rescale_output=False,
                                                  normalise = normalise,
                                                    **kwargs)
                                
                if new_tile.shape[-2:] != shape: #this only happens when the pixel size is close but not exactly the model pixel size.
               #     print(new_tile.shape, shape[-2:])
                    from torch.nn.functional import interpolate
                    new_tile = interpolate(new_tile, size=shape[-2:], mode="nearest").int()[0]

                new_tile = _to_ndim(new_tile, 3)

                num_iter = new_tile.shape[0]

                edge_window = True

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
                
                    if j == len(chop_list[0]) - 1:
                        window_j_start = window_j + pad
                    else:
                        window_j_start = window_j

                    if i == len(chop_list[1]) - 1:
                        window_i_start = window_i + pad
                    else:
                        window_i_start = window_i

                    if j == 0:
                        window_j_start = window_j

                    if i == 0:
                        window_i_start = window_i

                    if len(ignore_list) == 0:
                        edge_window = False

                    tile1 = canvas[n, window_i_start:window_i + shape[0], window_j_start:window_j + shape[1]]
                    tile2 = _remove_edge_labels(new_tile[n, window_i_start - window_i:shape[0], window_j_start - window_j:shape[1]], ignore=ignore_list)
                    if edge_window:

                        tile2 = torch_fastremap(tile2)
                        tile2[tile2>0] = tile2[tile2>0] + running_max

                        tile1_torch = torch.tensor(np.array(tile1), dtype = torch.int32)

                        remapped = match_labels(tile1_torch, tile2, threshold = 0.1)[1]
                        tile1_torch[remapped>0] = remapped[remapped>0].int()

                        running_max = max(running_max, tile1_torch.max())

                        canvas[n, window_i_start:window_i + shape[0], window_j_start:window_j + shape[1]] = tile1_torch.numpy().astype(np.int32)  

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

            if save_geojson:
                print("Exporting to geojson")
                _zarr_to_json_export(file_with_zarr_extension, 
                                     detection_size = detection_size, size = shape[0], scale = scale_factor, n_dim = n_dim)
                    
    def display(self,
                image: torch.tensor,
                instances: torch.Tensor,
                normalise: bool = True) -> np.ndarray:
        """
        Save the output of an InstanSeg model overlaid on the input.
        See :func:`save_image_with_label_overlay <instanseg.utils.save_image_with_label_overlay>` for more details and return types.
        :param image: The input image.
        :param instances: The output labels.
        """
        from instanseg.utils.utils import save_image_with_label_overlay

        instances = _to_ndim(instances, 4)
 
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        im_for_display = _display_colourized(image.squeeze(),normalise = normalise)
 
        output_dimension = instances.shape[1]
 
        if output_dimension ==1: #Nucleus or cell mask
            labels_for_display = instances[0,0] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=labels_for_display,return_image=True, label_boundary_mode="thick", label_colors=None,thickness=10,alpha=0.9)
        elif output_dimension ==2: #Nucleus and cell mask
            nuclei_labels_for_display = instances[0,0]
            cell_labels_for_display = instances[0,1] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=nuclei_labels_for_display,return_image=True, label_boundary_mode="thick", label_colors="red",thickness=10)
            image_overlay = save_image_with_label_overlay(image_overlay,lab=cell_labels_for_display,return_image=True, label_boundary_mode="inner", label_colors="green",thickness=1)
 
        else:
            raise ValueError(f"Output dimension {instances.shape} not supported")
        return image_overlay

    def _cluster_instances_by_mean_channel_intensity(self, image_tensor: torch.Tensor, 
                                                     labeled_output: torch.Tensor,
                                                     features: Optional[torch.Tensor] = None,
                                                      n_neighbors = 50,
                                                      n_pcs = 100,
                                                    resolution = 0.1,
                                                     device = "cuda",
                                                     channel_names = None,
                                                     normalise = True):

        #This is experimental code that is not yet implemented. You'll need to install rapids_singlecell, cuml and scanpy to run this code.

        from instanseg.utils.biological_utils import get_mean_object_features
        import fastremap
        import numpy as np
        from instanseg.utils.utils import apply_cmap, _choose_device
        from instanseg.utils.pytorch_utils import torch_fastremap
        try:
            import rapids_singlecell as rsc
        except ImportError:
            import warnings
            warnings.warn("rapids_singlecell not installed. Not using GPU.")
            import scanpy as rsc

        import scanpy as sc
        import matplotlib.pyplot as plt

        device = _choose_device(device, verbose= False)

        labeled_output = _to_ndim(labeled_output, 4)
        image_tensor = _to_ndim(image_tensor, 3)

        if features is None:
            X_features = get_mean_object_features( image_tensor.to(device), labeled_output.to(device),)
        else:
            X_features = features

        adata = sc.AnnData(X_features.cpu().numpy())
        try:
            rsc.get.anndata_to_GPU(adata)
        except:
            pass

        if channel_names is not None:
            adata.var_names = channel_names

        if normalise:    
            rsc.pp.scale(adata)
            
        rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        rsc.tl.umap(adata)
        rsc.tl.leiden(adata, resolution=resolution)

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

        return adata



def _threshold_thumbnail(slide, level=None, sigma = 3):
    from skimage.color import rgb2gray
    from skimage import filters
    import numpy as np

    if level is None:
        level = slide.level_count - 1

    img_thumbnail = slide.read_region((0, 0), level, size=(10000, 10000), as_array=True, padding=False)
    downsample_factor_thumbnail = slide.level_downsamples[level]

    gray_image = rgb2gray(np.array(img_thumbnail))
    threshold_value = filters.threshold_otsu(gray_image)
    gray_image = filters.gaussian(gray_image,sigma = sigma)>threshold_value
    binary_image = ~(gray_image > threshold_value)  # Apply the threshold to create a binary image

    return binary_image, downsample_factor_thumbnail



def _find_non_empty_positions(mask, chop_list, tile_size, chopped_image_size, emptiness_threshold = 0.1):
    """
    Precompute all valid positions within the mask where tiles can be placed.
    """
    from itertools import product
    from instanseg.utils.utils import show_images
    valid_positions = []

    downsample_factor_mask = chopped_image_size[0] / mask.shape[0]
    scaled_tile_size = round(round(tile_size / downsample_factor_mask,0))

    for y,x in product((chop_list[0]),(chop_list[1])):

        y = round(round(y / downsample_factor_mask,0))
        x = round(round(x / downsample_factor_mask,0))

        if mask[y:y + scaled_tile_size, x:x + scaled_tile_size].max() > emptiness_threshold:
            valid_positions.append(1)
        else:
            valid_positions.append(0)

    return valid_positions


def _rescale_to_pixel_size(image: torch.Tensor, 
                           requested_pixel_size: float, 
                           model_pixel_size: float,
                           mode: str = "bilinear") -> torch.Tensor:
    
    original_dim = image.dim()

    image = _to_ndim(image, 4)

    scale_factor = requested_pixel_size / model_pixel_size

    if not np.allclose(scale_factor,1, pixel_size_precision): #if you change this value, you MUST modify the whole_slide_image function.
        image = interpolate(image, scale_factor=scale_factor, mode=mode)

    return _to_ndim(image, original_dim)

    
def _display_colourized(mIF, normalise = True):
    from instanseg.utils.utils import _move_channel_axis, generate_colors, percentile_normalize
 
    mIF = _to_tensor_float32(mIF)
 
    if normalise:
        mIF = percentile_normalize(mIF)
        mIF = torch.clamp(mIF, 0, 1)
    if mIF.shape[0]!=3:
        colours = generate_colors(num_colors=mIF.shape[0])
        colour_render = (mIF.flatten(1).T @ torch.tensor(colours)).reshape(mIF.shape[1],mIF.shape[2],3)
    else:
        colour_render = mIF
    colour_render = torch.clamp_(colour_render, 0, 1)
    colour_render = _move_channel_axis(colour_render,to_back = True).detach().numpy()*255
    return colour_render.astype(np.uint8)