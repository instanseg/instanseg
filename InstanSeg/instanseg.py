from typing import Union, List, Optional
import numpy as np
import torch
from torch.nn.functional import interpolate
from pathlib import Path, PosixPath
from InstanSeg.utils.utils import percentile_normalize



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

def _rescale_to_pixel_size(image: torch.Tensor, requested_pixel_size: float, model_pixel_size: float) -> torch.Tensor:

    if image.dim() == 3:
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


class InstanSeg():
    """
    Main class for running InstanSeg.
"""

    def __init__(self, model_type: str = "brightfield_nuclei", device: Optional[str] = None, image_reader: str = "tiffslide", verbose = True):
        from InstanSeg.utils.utils import download_model, _choose_device
        self.instanseg = download_model(model_type, return_model=True)
        self.inference_device = _choose_device(device)
        self.instanseg = self.instanseg.to(self.inference_device)
        self.verbose = verbose
        self.prefered_image_reader = image_reader
        self.small_image_threshold = 3 * 1500 * 1500
        self.medium_image_threshold = 5000 * 5000

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
            from InstanSeg.utils.utils import read_pixel_size
            img_pixel_size = read_pixel_size(image_str)


        if img_pixel_size is not None:
            assert float(img_pixel_size) > 0 and float(img_pixel_size) < 2, f"Pixel size {img_pixel_size} microns per pixel is invalid."
        
        return image_array, img_pixel_size
    

    def read_slide(self, image_str: str):      
        if self.prefered_image_reader == "tiffslide":
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
        elif self.prefered_image_reader == "AICSImageIO":
            from aicsimageio import AICSImage
            slide = AICSImage(image_str)
        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented.")
        return slide
    
    def _to_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return _to_tensor_float32(image)
    
    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        assert image.ndim == 3 or image.ndim == 4, f"Input image shape {image.shape} is not supported."
        if image.dim() == 3:
            image = percentile_normalize(image)
            image = image[None]
        else:
            image = torch.stack([percentile_normalize(i) for i in image])

        return image

    def eval(self, image: Union[str, List[str], np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]], 
            pixel_size: Optional[float] = None,
            normalise: bool = True, 
            batch_size: int = 1,
            **kwargs):
        """
        Evaluate the input image or list of images using the InstanSeg model.
        
        Args:
            image (Union[str, List[str], np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]): 
                The input image or images to be evaluated, either as file paths or arrays/tensors.
            pixel_size (Optional[float]): The pixel size of the image. If not provided, it will be read from the image metadata.
        """

        if isinstance(image, PosixPath):
            image = str(image)
        if isinstance(image, str):
            initial_type = "not_list"
            image_list = [image]
        elif isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
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
                                                       normalise = normalise,
                                                       return_image_tensor=False, **kwargs)

                else:
                    instances = self.eval_medium_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       normalise = normalise,
                                                       return_image_tensor=False,
                                                       batch_size = batch_size, **kwargs)

                output_list.append(instances)
                    
            else:
                self.eval_whole_slide_image(image_array, pixel_size, normalise, batch_size)

        if initial_type == "not_list":
            output = output_list[0]
        else:
            output = output_list
        
        return output
            

    def eval_small_image(self,image: torch.Tensor, pixel_size: Optional[float] = None,
                          normalise: bool = True,
                          return_image_tensor: bool = True,
                          **kwargs):
        """
        Evaluate the input image using the InstanSeg model.
        
        Args:
            image (torch.Tensor): The input image(s) to be evaluated.
            pixel_size (Optional[float]): The pixel size of the image. If not provided, it will be read from the image metadata.
        """

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

        with torch.cuda.amp.autocast():
            instances = self.instanseg(image, **kwargs)

        if pixel_size is not None and img_has_been_rescaled:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

        
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
                          **kwargs):
        
        image = _to_tensor_float32(image)
        
        from InstanSeg.utils.tiling import sliding_window_inference
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

        instances = sliding_window_inference(image,
                    self.instanseg, 
                    window_size = (tile_size,tile_size),
                    sw_device = self.inference_device,
                    device = 'cpu', 
                    batch_size= batch_size,
                    output_channels = output_dimension,
                    show_progress= self.verbose,
                    **kwargs).float()
        
        if pixel_size is not None and img_has_been_rescaled:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="linear")
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
            from InstanSeg.utils.tiling import segment_image_larger_than_memory
                                 
            segment_image_larger_than_memory(instanseg_class= self,
                                            image_path= image, 
                                            memory_block_size = (int(self.medium_image_threshold**0.5),int(self.medium_image_threshold **0.5)),  #this is the size of the image that will be read in memory
                                            inference_tile_size = (tile_size,tile_size), #this is the size of the image that will be passed to the model
                                            batch_size= batch_size,
                                            to_geojson= output_geojson, 
                                            pixel_size =pixel_size,
                                            prediction_tag = self.prediction_tag, 
                                            sw_device = self.inference_device,
                                            normalisation_subsampling_factor=10,
                                            **kwargs)
            
    
    def display(self, image: torch.tensor,
               instances: torch.Tensor):
        
        from InstanSeg.utils.utils import display_colourized, save_image_with_label_overlay

        im_for_display = display_colourized(image.squeeze().cpu())

        output_dimension = instances.shape[1]

        if output_dimension ==1: #Nucleus or cell mask]
            labels_for_display = instances[0,0].cpu().numpy() #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=labels_for_display,return_image=True, label_boundary_mode="thick", label_colors=None,thickness=10,alpha=0.5)
        elif output_dimension ==2: #Nucleus and cell mask
            nuclei_labels_for_display = instances[0,0].cpu().numpy()
            cell_labels_for_display = instances[0,1].cpu().numpy() #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=nuclei_labels_for_display,return_image=True, label_boundary_mode="thick", label_colors="red",thickness=10)
            image_overlay = save_image_with_label_overlay(image_overlay,lab=cell_labels_for_display,return_image=True, label_boundary_mode="inner", label_colors="green",thickness=1)

        return image_overlay

            
if __name__ == "__main__":
    import os
    import pdb
    from InstanSeg.utils.utils import show_images

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"./examples/"))

    instanseg = InstanSeg("brightfield_nuclei")
    # instances = instanseg.eval_whole_slide_image(example_image_folder / "HE_Hamamatsu.tiff")

    instanseg = InstanSeg("brightfield_nuclei")
    instances = instanseg.eval(example_image_folder / "HE_example.tif")

    instanseg.prefered_image_reader = "skimage.io"
    instances = instanseg.eval(example_image_folder / "HE_example.tif")

    instanseg.prefered_image_reader = "AICSImageIO"
    instances = instanseg.eval(example_image_folder / "HE_example.tif")

    instanseg = InstanSeg("fluorescence_nuclei_and_cells")
    instances = instanseg.eval(example_image_folder / "LuCa1.tif", batch_size=3)

    show_images(instances)

        



        














 

