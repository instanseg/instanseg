from instanseg.utils.utils import show_images, _move_channel_axis
import torchvision
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
from torchvision.transforms import RandomCrop, Resize, RandomPerspective
from monai.transforms import RandGaussianNoise, AdjustContrast
from instanseg.utils.utils import percentile_normalize, generate_colors
import warnings

import time
import fastremap

time_dict = {}


def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        if f.__name__ in time_dict.keys():
            time_dict[f.__name__] += te - ts
        else:
            time_dict[f.__name__] = te - ts
        # print((f.__name__, ))
        return result

    return timed


def measure_average_instance_area(lab):
    if torch.is_tensor(lab):
        if (len(torch.unique(lab) - 1)) > 0:
            return torch.sum((lab > 0).float()) / (len(torch.unique(lab) - 1))
        else:
            return 0
    else:
        if (len(np.unique(lab) - 1)) > 0:
            return np.sum((lab > 0)) / (len(np.unique(lab) - 1))
        else:
            return 0


def resize_lab(lab, size, double_check=False):
    """This function resizes a label to required instance area. It does so by first measuring the average instance area and assumes instances are roughly circular"""
    avg_size = measure_average_instance_area(lab)

    if avg_size == 0:
        return list(np.array(lab[0].shape))

    radius_orig = np.sqrt(avg_size / 3.14)  # Assuming labels are roughly circular
    radius_requested = np.sqrt(size / 3.14)

    # print(avg_size,size,radius_orig,radius_requested)

    ratio = radius_orig / radius_requested

    shape = np.array(lab[0].shape)

    if double_check:
        resized_lab = Resize(size=list(((shape / ratio).int())), antialias=True,
                             interpolation=torchvision.transforms.InterpolationMode.NEAREST)(lab[None,]).to(lab.dtype)
        assert np.isclose(measure_average_instance_area(resized_lab), size, rtol=0.6)

    return list(((shape / ratio).int()))


def generate_random_label_area(min=30, max=30): 
    from scipy.stats import skewnorm
    a = 5
    # mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
    mean = 0.7824
    r = skewnorm.rvs(a, size=1)
    r = (r - mean) + 1
    r = np.clip(((r - mean) + 0.2) * 500, min, max)
    return r



def get_marker_location(meta):
    from instanseg.utils.augmentation_config import markers_info
    stains = [channel_str.split(" ")[0] for channel_str in meta['channel_names']]
    subcellular_location = ["N/A" if channel.upper() not in markers_info.keys() else markers_info[channel.upper()]['Subcellular Location'] for channel in stains]
    meta["subcellular_location"] = subcellular_location
    return meta
    
class Augmentations(object):
    def __init__(self, augmentation_dict={}, shape=(256, 256), dim_in=3,
                 nuclei_channel=None, debug=False, modality=None, cells_and_nuclei=False, target_segmentation="N",channel_invariant = False):
        self.debug = debug
        self.shape = shape
        self.augmentation_dict = augmentation_dict
        self.modality = modality
        self.cells_and_nuclei = cells_and_nuclei
        self.target_segmentation = target_segmentation
        self.dim_in = dim_in  # Note, this is the number of input channels to the model, not the number of channels in the raw image. (Can be 'None' for channel invariant models)
        self.nuclei_channel = nuclei_channel  # The channel that contains the nuclei. (Can be 'None' for brightfield images or if in image metadata)
        self.channel_invariant = channel_invariant
    def to_tensor(self, image, labels=None, normalize=False, amount=None, metadata=None):


        if isinstance(image, np.ndarray):
            if self.debug:
                orig = image.copy()

            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float32)
            if normalize:
                image, _ = self.normalize(image)

            out = torch.tensor(_move_channel_axis(image), dtype=torch.float32)

        elif isinstance(image, torch.Tensor):
            if self.debug:
                orig = torch.clone(image)
            out = _move_channel_axis(image).float()
        
            if normalize:
                out, _ = self.normalize(out)

        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = np.atleast_3d(labels)
                labels = _move_channel_axis(labels)
                for n, lab in enumerate(labels):

                    if not (lab == -1).any():  # convention is to skip labels with a value of -1
                        labels[n] = fastremap.renumber(lab)[0]
                    else:
                        labels[n] = lab

                labels = torch.tensor(labels.astype(np.int32), dtype=torch.int32)
            labels = torch.atleast_3d(labels)
            labels = _move_channel_axis(labels)

        out = out.squeeze()
        out = _move_channel_axis(torch.atleast_3d(out))

    
        if self.debug:
            print("Tensor")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def normalize(self, image: torch.Tensor, labels=None, amount: float = 0., subsampling_factor: int = 1,
                  percentile=0.1, metadata=None):
        out = percentile_normalize(image, subsampling_factor=subsampling_factor, percentile=percentile)

        return out, labels
    

    def extract_hematoxylin_stain(self, image: torch.Tensor, labels=None, amount=0, metadata=None):
        # image should be 3 channel RGB between 0 and 255 (float32)
        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels

        import torchstain
        if self.debug:
            orig = torch.clone(image)

        tensor = (image / (image.max() + 0.001)) * 255

        tensor = torch.clamp(tensor, 0., 255.)

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        try:

            normalizer.HERef += (torch.rand_like(normalizer.HERef) - 0.5) * normalizer.HERef * amount
            normalizer.maxCRef += (torch.rand_like(normalizer.maxCRef) - 0.5) * normalizer.maxCRef * amount
            norm, H, E = normalizer.normalize(I=tensor, stains=True, Io=240)
        except:
            return image,labels

        out = _move_channel_axis(H) / 255.

        if self.debug:
            print("Stain separation")
            show_images([orig, out], titles=["Original", "Transformed"])

        return out, labels

    def normalize_HE_stains(self, image: torch.Tensor, labels=None, amount=0, metadata=None):
        # image should be 3 channel RGB between 0 and 255 (float32)
        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels
        
        assert image.shape[0] == 3
        
        import torchstain
        if self.debug:
            orig = torch.clone(image)

        tensor = (image / (image.max() + 0.001)) * 255

        tensor = torch.clamp(tensor, 0., 255.)

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        try:
            normalizer.HERef += (torch.rand_like(normalizer.HERef) - 0.5) * normalizer.HERef * amount
            normalizer.maxCRef += (torch.rand_like(normalizer.maxCRef) - 0.5) * normalizer.maxCRef * amount
            norm, _, _ = normalizer.normalize(I=tensor, stains=False, Io=240, beta=0.15)
        except:
            return image,labels


        out = _move_channel_axis(norm) / 255.

        if self.debug:
            print("Stain normalization")
            show_images([orig, out], titles=["Original", "Transformed"])

        return out, labels

    def randomJPEGcompression(self, image, labels=None, amount=0, metadata=None):
        """ This function applies random JPEG compression to an image. It is used to simulate the effect of JPEG compression on the image.

        :param image: A tensor of shape C,H,W
        :param labels: any labels associated with the image
        """

        if image.shape[0] != 3:  # Not implemented for >3 channels
            return image, labels

        import io
        from PIL import Image
        from torchvision import transforms
        import torchvision.transforms.functional as F
        import random

        if self.debug:
            orig = torch.clone(image)  # .copy()
        # Convert tensor to PIL image

        image = image.clamp(0, 1) * 255
        image = image.byte()


        image = F.to_pil_image(image)

        # # Apply random JPEG compression
        qf = random.randrange(int(100 * (1 - amount) - 1), 100)
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimize=True)
        outputIoStream.seek(0)

        # # Convert compressed image back to Torch tensor
        out = transforms.ToTensor()(Image.open(outputIoStream))

        if self.debug:
            print("JPEG")
            show_images([orig, out, orig - out], titles=["Original", "Transformed"])

        return out, labels

    def brightness_augment(self, image, labels=None, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)  # .copy()
        rand = 1 + ((torch.rand(len(image)) - 0.5) * amount)
        out = (image.permute(1, 2, 0) * rand).permute(2, 0, 1)
        if self.debug:
            print("Brightness")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def RandGaussianNoise(self, image, labels=None, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()

        out = RandGaussianNoise(prob=1, mean=0.0, std=amount / 5)(image)

        if self.debug:
            print("RandGaussianNoise")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def HistogramNormalize(self, image, labels=None, amount=0, metadata=None):
        from monai.transforms import RandStdShiftIntensity

        if self.debug:
            orig = torch.clone(image)  # .copy()

        normalizer = RandStdShiftIntensity(10)

        out = torch.stack([normalizer(c) for c in image])

        if self.debug:
            print("RandStdShiftIntensity")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels
    

    def AdjustContrast(self, image, labels=None, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()

        out = AdjustContrast(gamma=amount)(image)

        if self.debug:
            print("AdjustContrast")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels
    
    def flips(self, image, labels, amount=0, metadata=None):

        amount = 0.5

        if self.debug:
            orig = torch.clone(image)  # .copy()
        if random.random() > (1 - amount):
            image = TF.hflip(image)
            labels = TF.hflip(labels)
        if random.random() > (1 - amount):
            image = TF.vflip(image)
            labels = TF.vflip(labels)
        out = image
        if self.debug:
            print("Flips")
            show_images([orig, out])
        return out, labels

    def rotate(self, image, labels, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)

        angle = int(np.random.choice([180, 90, 270, 0]))
        out = TF.rotate(image, angle)
        labels = TF.rotate(labels, angle)

        if self.debug:
            print("Rotate")
            show_images([orig, out])
        return out, labels

    def adjust_hue(self, image, labels, amount=0, metadata=None):

        assert image.shape[0] == 3

        if metadata is not None and metadata["image_modality"] != "Brightfield":
            return image, labels

        if self.debug:
            orig = torch.clone(image)

        out = TF.adjust_hue(image, hue_factor=(torch.rand(1) - 0.5) * amount * 0.5)

        out = torch.nan_to_num(out, nan=0)  # This occasionally produces nan values, which we replace with 0

        if self.debug:
            print("Adjust Hue")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels

    def perspective(self, image, labels, amount=0, metadata=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()
        perspective_transformer = RandomPerspective(distortion_scale=amount / 2, p=1.0,
                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                    fill=float(image.max() if metadata[
                                                                                  "image_modality"] == "Brightfield" else image.min()))
        state = torch.get_rng_state()

        out = perspective_transformer(image)

        perspective_transformer = RandomPerspective(distortion_scale=amount / 2, p=1.0,
                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                    fill=0)

        torch.set_rng_state(state)
        labels = perspective_transformer(labels)

        if self.debug:
            print("perspective")
            show_images([orig, out, labels], )
        return out, labels

    def invert(self, image, labels, amount=0, metadata=None):

        out = 1 - image

        if self.debug:
            print("Inversion")
            show_images([image, out])
        return out, labels


    # @measure_time
    def pseudo_brightfield(self, image, labels=None, amount=0, c_nuclei=0, metadata=None, random_seed=None):

        if self.debug:
            orig = torch.clone(image)  # .copy()
        orig_dtype = image.dtype

        if random_seed is not None:
            np.random.seed(random_seed)

        if metadata is not None and metadata["image_modality"] != "Fluorescence":
            return image, labels
        
        image , _ , c_nuclei= self.extract_nucleus_and_cytoplasm_channels(image, labels, c_nuclei, metadata)

        image = percentile_normalize(image, 0.01, subsampling_factor=5)

        stains = {
            'hematoxylin': torch.Tensor([0.65, 0.70, 0.29]),
            'dab': torch.Tensor([0.27, 0.57, 0.78]),
            'eosin': torch.Tensor([0.2159, 0.8012, 0.5581])
        }

        assert c_nuclei is not None
        im_nuclei = image[c_nuclei]

        my_stains = [torch.clone(stains['hematoxylin'])]
        channels = [im_nuclei]

        if np.random.random() > 0.1:
            my_stains.append(torch.clone(stains['dab']))

            all_channels = np.arange(len(image))
            c_other = np.random.choice(all_channels[all_channels != c_nuclei])
            channels.append(image[c_other])

        for ii in range(len(my_stains)):
            stain = my_stains[ii]
            stain += np.random.random(3) / 4
            stain /= np.linalg.norm(stain)
            stain_scale = 1 + (np.random.random() - 0.5) / 5
            my_stains[ii] = stain * stain_scale

        im_output = 0
        channels = [temp for temp in channels]

        for stain, im_channel in zip(my_stains, channels):
            stain = stain.reshape((3, 1, 1))

            im_output += im_channel * stain * float(np.random.randint(3, 5))

        im_output = torch.exp(-np.log(10.0) * im_output)
        im_output = torch.clip(im_output, 0, 1)

        if self.debug:
            print("Eosin")
            show_images([orig, im_output], titles=["Original", "Transformed"])

        return torch.Tensor(im_output).to(orig_dtype), labels

    def colourize(self, image, labels=None, amount=0, c_nuclei=-1, random_seed=None, metadata=None):

        # Expects a 3D tensor C,H,W

        if self.debug:
            orig = torch.clone(image)  # .copy()

        if random_seed is not None:
            np.random.seed(random_seed)

        if metadata is not None and metadata["image_modality"] != "Fluorescence":
            return image, labels
        # colours = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]

        if image.shape[0] == 3:
            return image, labels
        
        if c_nuclei is None:
            c_nuclei = 0

        colours = generate_colors(num_colors=image.shape[0])
        np.random.shuffle(colours)
        if np.min(image.shape) == 1:
            c_nuclei = 0  # If we have a greyscale image, we want to colourize the one and only nuclei channel
        
        coloured_image = image[c_nuclei][None,] * torch.Tensor([0, 0, 1])[:, None, None]
        for i, image_channel in enumerate(image[np.arange(len(image)) != c_nuclei]):
            colour = colours[i]
            # greyscale_image_channel = image_channel[None,].expand(3, -1, -1)
            coloured_image_channel = image_channel * torch.Tensor(colour)[:, None, None]
            coloured_image += coloured_image_channel
        assert not coloured_image.isnan().any()
        image = percentile_normalize(coloured_image, 0.1, subsampling_factor=5)
        if self.debug:
            print("Colourize")
            show_images([orig, image], titles=["Original", "Transformed"])
        assert not coloured_image.isnan().any()
        return image, labels

    def pseudo_imc(self, image, labels, amount=0, metadata=None):

        # Expects a 3D tensor C,H,W and labels C,H,W
        

        if metadata is not None and metadata["image_modality"] != "Fluorescence":
            return image, labels

        if self.debug:
            orig = torch.clone(image)  # .copy()

        original_shape = image.shape

        if np.min(torch.squeeze(
                image).shape) == 3:  # Don't want to run the risk of having a brightfield image in the IMC pipeline.
            if self.debug:
                warnings.warn("Possible Brightfield image in IMC pipeline, ignoring.")
                show_images([orig, image], titles=["Original", "Transformed"])

            return image, labels

        shape = resize_lab(labels,
                           generate_random_label_area(min=30, max=50))  # Rough size for the nuclei in IMC images.

        resized_data = Resize(size=shape, antialias=True)(image)

        resized_data[resized_data < 0] = 0

        amount = 30 - (
                amount * 27)  # Assuming we have amount between [0,1], we want to map that to something like [30,3]

        resized_data = torch.poisson(resized_data * amount)

        resized_data = percentile_normalize(resized_data, percentile=0.1, subsampling_factor=3)

        image = Resize(size=original_shape[1:], interpolation=np.random.choice(
            [torchvision.transforms.InterpolationMode.NEAREST, torchvision.transforms.InterpolationMode.BILINEAR]),
                       antialias=True)(
            resized_data)

        if self.debug:
            print("Pseudo_imc")
            show_images([orig, image], titles=["Original", "Transformed"])

        return image, labels
    
    def channel_shuffle(self, image, labels=None, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)
        channels = torch.randperm(image.shape[0])
        out = image[channels]
        if self.debug:
            print("Channel shuffle")
            show_images([orig, out], titles=["Original", "Transformed"])
        return out, labels
    
    def add_noisy_channels(self, image, labels=None, max_channels = 30, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)

        new_channels_num = np.random.randint(1,max_channels)

        new_channels =np.random.choice(range(image.shape[0]),new_channels_num,replace = True)

        out = image[new_channels]
        out[out<0]=0
        amount = np.random.randint(2, 30 * amount)

        channel_weights = torch.randint(1,amount,size = [new_channels_num,1,1],dtype = torch.float32)


        out = torch.poisson(out * channel_weights) / (channel_weights*2) 

        out = torch.cat((image,out),dim=0)

        channels = torch.randperm(out.shape[0])
        out = out[channels]


        return out, labels

    # @measure_time
    def add_gradient(self, image, labels=None, amount=0, metadata=None):
        if self.debug:
            orig = torch.clone(image)  # .copy()
        _, h, w = image.shape
        xs = torch.linspace(0, 0.5, steps=w)
        ys = torch.linspace(0, 0.5, steps=h)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        if random.random() > 0.5:
            x = TF.hflip(x)
        if random.random() > 0.5:
            y = TF.hflip(y)
        if random.random() > 0.5:
            x = TF.vflip(x)
        if random.random() > 0.5:
            y = TF.vflip(y)
        out = (amount * (np.random.random() * x + np.random.random() * y)) * image.max() + image
        if self.debug:
            print("Gradient")
            show_images([orig, out], titles=["Original", "Transformed"])

        return torch.Tensor(out), labels

    # @measure_time

    def channel_subsample(self, image, labels=None, max_channels=None, c_nuclei=None, min_channels=1, metadata=None):
        channel_num = image.shape[0]

        if channel_num > max_channels:
            if c_nuclei is None or c_nuclei > channel_num:
                slice = np.random.choice(np.arange(channel_num), max_channels, replace=False)
            else:
                if max_channels == 1:
                    slice = [c_nuclei]
                    c_nuclei = 0
                else:

                    slice = np.random.choice(np.arange(channel_num)[np.arange(channel_num) != c_nuclei],
                                             np.random.randint(min_channels - 1, max_channels - 1), replace=False)

                    np.random.shuffle(slice)
                    slice = np.append(slice, c_nuclei)

                    c_nuclei = np.where(slice == c_nuclei)[0][0]

            image = image[slice]

        return image, labels, c_nuclei
    
    def channel_suppress(self, image, labels=None, amount = None, metadata=None):

        slice = torch.rand(image.shape[0]) < (1- amount)

        if torch.sum(slice) == 0:
            slice[np.random.randint(0,image.shape[0])] = True

        image = image[slice]

        return image, labels

    def extract_nucleus_and_cytoplasm_channels(self, image, labels=None, c_nuclei=None, metadata=None, amount=None):
        channel_num = image.shape[0]

        if metadata is None or "subcellular_location" not in metadata.keys() or len(
                metadata["subcellular_location"]) != channel_num or c_nuclei is None:
            return image, labels, c_nuclei

        cytoplasm_channel_ids = ["Cytoplasm" in x or "Membrane" in x or "N/A" in x for x in
                                 metadata["subcellular_location"]]
        nuclei_channels_ids = c_nuclei

        if len(cytoplasm_channel_ids) == 0:
            return image, labels, c_nuclei

        cytoplasm_channels = image[cytoplasm_channel_ids].sum(0)  # Sum projection of the cytoplasm channels
        nuclei_channels = image[nuclei_channels_ids]

        image = torch.stack([nuclei_channels, cytoplasm_channels])

        c_nuclei = 0

        return image, labels, c_nuclei

    
    def torch_rescale(self, image, labels=None, amount=0, current_pixel_size=None, requested_pixel_size=None, crop=True,
                      random_seed=None, metadata=None, modality = None):

    

        if random_seed is not None:
            torch.manual_seed(random_seed)

        if labels is not None:
            assert image.shape[-2:] == labels.shape[-2:]


        shape = (torch.tensor(image.shape[-2:]) * (1 + ((torch.rand(1) - 0.5) * amount))).int().tolist()

        if metadata is not None:
            if current_pixel_size is None and "pixel_size" in metadata.keys():
                current_pixel_size = metadata["pixel_size"]
            if modality is None and "image_modality" in metadata.keys():
                modality = metadata["image_modality"]
            else:
                modality = None

        if modality is None:
            modality = "Brightfield"
        #    print("Modality not specified in metadata or in function call, assuming Brightfield")


        if current_pixel_size is not None and requested_pixel_size is not None:
            scale = (current_pixel_size / requested_pixel_size)
            shape = (torch.tensor(image.shape[-2:]) * scale).int().tolist()

        resized_data = Resize(size=shape, antialias=True)(image)



        if labels is not None:

            resized_labels = Resize(size=shape, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(labels)

        while self.shape is not None and np.any(np.array(resized_data[0].shape) < self.shape[0]) and crop:
            pad = int((self.shape[0] - min(resized_data[0].shape)) / 2) + 3
            pad = torch.Tensor([pad, resized_data.shape[1], resized_data.shape[2]]).min().int() - 1

            

            resized_data = torch.nn.functional.pad(resized_data, (pad, pad, pad, pad), mode='constant',
                                                   value=image.max() if modality == "Brightfield" else resized_data.min())

            if labels is not None:
                resized_labels = torch.nn.functional.pad(resized_labels, (pad, pad, pad, pad), mode='constant', value = min(resized_labels.min(),0)).to(
                    labels.dtype)
                
              
        if not crop:
            if labels is None:
                return resized_data, None
            else:
                return resized_data, resized_labels

        cropper = RandomCrop(size=self.shape)
        (i, j, h, w) = cropper.get_params(resized_data, output_size=self.shape)
        out_image = resized_data[:, i:i + h, j:j + w]

        if labels is not None:
            out_labels = resized_labels[:, i:i + h, j:j + w]

            assert out_image.shape[-2:] == self.shape and out_labels.shape[-2:] == self.shape

        if self.debug:
            print("Rescaling")
            show_images([image[0], labels, out_image[0], out_labels],
                        titles=["Source", "Target", "Resize source", "Resized target"], n_cols=2, axes=True)
            
        

        if labels is None:
            return out_image.float(), None
        else:
            return out_image.float(), out_labels #torch.Tensor(out_labels)  # .short()
        
        
    def duplicate_grayscale_channels(self, image, labels, metadata=None):
        if self.debug:
            orig = torch.clone(image)
        if image.shape[0] == 1 and self.dim_in != 1 and not self.channel_invariant:
            image = image.repeat(self.dim_in, 1, 1)
        elif image.shape[0] != self.dim_in and not self.channel_invariant:
            raise ValueError(
                "Image has {} channels, but model expects {}, check the augmentations pipeline!".format(image.shape[0],
                                                                                                        self.dim_in))

        if self.debug:
            print("Duplicate channels")
            show_images([orig, image], titles=["Original", "Transformed"])
        return image, labels

    def __call__(self, image, labels, meta=None):

        from instanseg.utils.utils import _estimate_image_modality

        if self.modality is None:
            if meta is not None and "image_modality" in meta.keys():
                if meta["image_modality"] in ["Brightfield", "Chromogenic"]:
                    observed_modality = "Brightfield"
                else:
                    observed_modality = meta["image_modality"]
            else:
                observed_modality = _estimate_image_modality(image, labels)
            modality = observed_modality
        else:
            modality = self.modality

        if self.debug:
            print("Observed modality:", modality)


        augmentation_dict = self.augmentation_dict[modality]

        if self.debug:
            print(augmentation_dict.keys())

        if meta is not None:
            if "nuclei_channels" in meta.keys():
                channels_nuclei = meta["nuclei_channels"]
                c_nuclei = np.random.choice(channels_nuclei)

            else:
                c_nuclei = self.nuclei_channel

            if "pixel_size" in meta.keys():
                pixel_size = meta["pixel_size"]
            else:
                pixel_size = None

        if meta is not None and "channel_names" in meta.keys():
            from instanseg.utils.augmentation_config import markers_info
            stains = [channel_str.split(" ")[0] for channel_str in meta['channel_names']]
            subcellular_location = [
                "N/A" if channel.upper() not in markers_info.keys() else markers_info[channel.upper()][
                    'Subcellular Location'] for channel in stains]
        else:
            subcellular_location = ["N/A" for _ in range(image.shape[0])]

        metadata = {"image_modality": modality, "nuclei_channels": c_nuclei, "pixel_size": pixel_size,
                    "subcellular_location": subcellular_location}

        has_been_normalized = False

        for augmentation, values in augmentation_dict.items():

            if np.random.random() < values[0]:

                if augmentation in ["normalize_HE_stains", "extract_hematoxylin_stain", "normalize", "pseudo_background"]:
                    if not has_been_normalized:
                        if augmentation != "normalize":
                            _, amount = values
                        image, labels = getattr(self, augmentation)(image, labels, amount=amount, metadata=metadata)
                        has_been_normalized = True

                    else:
                        pass

                elif augmentation == "pseudo_brightfield":
                    image, labels = self.pseudo_brightfield(image, labels, c_nuclei=c_nuclei, metadata=metadata)
                    metadata["image_modality"] = "Brightfield"

                elif augmentation == "channel_subsample":
                    _, (min_channels, max_channels) = values
                    image, labels, c_nuclei = self.channel_subsample(image, labels, max_channels=max_channels + 1,
                                                                     c_nuclei=c_nuclei, min_channels=min_channels,
                                                                     metadata=metadata)
                elif augmentation == "extract_nucleus_and_cytoplasm_channels":
                    image, labels, c_nuclei = self.extract_nucleus_and_cytoplasm_channels(image, labels,
                                                                                          c_nuclei=c_nuclei,
                                                                                          metadata=metadata)

                elif augmentation == "torch_rescale":
                    _, requested_pixel_size, amount = values
                    image, labels = self.torch_rescale(image, labels, current_pixel_size=None,
                                                       requested_pixel_size=requested_pixel_size, amount=amount,
                                                       metadata=metadata)

                elif augmentation == "colourize":
                    _, amount = values
                    image, labels = self.colourize(image, labels, c_nuclei=c_nuclei, metadata=metadata)


                elif augmentation == "add_noisy_channels":
                    _, max_channels = values
                    image, labels = self.add_noisy_channels(image, labels, metadata=metadata, max_channels=max_channels, amount = 0.5)

                else:
                    p, *rest = values
                    amount = rest[0] if len(rest) == 1 else None

                    image, labels = getattr(self, augmentation)(image, labels, amount=amount, metadata=metadata)

                assert not image.isnan().any()
        

        image, labels = self.duplicate_grayscale_channels(image,
                                                          labels,
                                                          metadata=metadata)  # This will only catch single channel images fed to a multi channel network and duplicate the channel if required.

        if image.var() > 1e2:
            image = torch.clip(image, min=-1, max=5)
            warnings.warn("Warning, variance of image is very high, check augmentations")

        return image, labels


if __name__ == "__main__":
    from instanseg.utils.augmentation_config import get_augmentation_dict

    augmentation_dict = get_augmentation_dict(nuclei_channel=6, dim_in=3, amount=0.5,augmentation_type="heavy")['train']

    import tifffile

    img = tifffile.imread(r"../examples/LuCa1.tif")[:,:512,:512]
    label = tifffile.imread(r"../examples/LuCa1_label.tif")[:512,:512]
    meta = {"image_modality": "Fluorescence", "nuclei_channels": [7]}


    Augmenter = Augmentations(augmentation_dict=augmentation_dict, debug=False,dim_in = None)

    show_images([Augmenter(img, label, meta)[0] for i in range(30)])
