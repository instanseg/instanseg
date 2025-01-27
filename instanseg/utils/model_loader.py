import numpy as np


def build_monai_model(model_str: str, build_model_dictionary: dict):


    if model_str == "AttentionUNet":
        from monai.networks.nets import AttentionUnet

        model = AttentionUnet(spatial_dims=2, in_channels=int(build_model_dictionary["dim_in"]),
                              out_channels=build_model_dictionary["dim_out"], \
                              dropout=build_model_dictionary["dropprob"], channels=build_model_dictionary["layers"], \
                              strides=tuple([2 for _ in build_model_dictionary["layers"][:-1]])
                              )
    elif model_str == "FlexibleUNet":
        from monai.networks.nets import FlexibleUNet
        model = FlexibleUNet(in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"],
                             backbone="efficientnet-b0")
        

    elif model_str == "BasicUNetPlusPlus":
        from monai.networks.nets import BasicUNetPlusPlus
        model = BasicUNetPlusPlus(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                                  out_channels=build_model_dictionary["dim_out"],
                                  dropout=build_model_dictionary["dropprob"])

        class ModelWrapper(BasicUNetPlusPlus):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, inputs):
                output = super().forward(inputs)
                modified_output = output[0]  # Modify the output here as needed
                return modified_output

        model = ModelWrapper(spatial_dims=2, in_channels=build_model_dictionary["dim_in"],
                             out_channels=build_model_dictionary["dim_out"], dropout=build_model_dictionary["dropprob"])

    elif model_str == "UNETR":
        from monai.networks.nets import UNETR
        model = UNETR(in_channels=build_model_dictionary["dim_in"], out_channels=build_model_dictionary["dim_out"],
                      img_size=256, feature_size=32, norm_name='batch', spatial_dims=2)


    else:
        raise NotImplementedError("Model not implemented: " + model_str)

    return model


def read_model_args_from_csv(path=r"../results/", folder=""):
    import pandas as pd
    from pathlib import Path
    model_path = Path(path) / folder
    df = pd.read_csv(model_path / "experiment_log.csv", header=None)
    build_model_dictionary = dict(zip(list(df[0]), list(df[1])))

    if "model_shape" in build_model_dictionary.keys():
        build_model_dictionary["model_shape"] = eval(build_model_dictionary["model_shape"])
    for key in ["dim_in", "n_sigma", "dim_out", "dim_coords"]:
        build_model_dictionary[key] = eval(str(build_model_dictionary[key])) if str(
            build_model_dictionary[key]) != "nan" else None
    if "to_centre" in build_model_dictionary.keys():
        build_model_dictionary["to_centre"] = eval(build_model_dictionary["to_centre"])
    if "dropprob" in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = float(build_model_dictionary["dropprob"])
    if "layers" in build_model_dictionary.keys():
        build_model_dictionary["layers"] = tuple(eval(build_model_dictionary["layers"]))
    if "requested_pixel_size" in build_model_dictionary.keys():
        build_model_dictionary["pixel_size"] = float(build_model_dictionary["requested_pixel_size"])
    if "cells_and_nuclei" in build_model_dictionary.keys():
        build_model_dictionary["cells_and_nuclei"] = bool(eval(build_model_dictionary["cells_and_nuclei"]))
    if "norm" in build_model_dictionary.keys():
        if build_model_dictionary["norm"] == "None" or str(build_model_dictionary["norm"]).lower() == "nan":
            build_model_dictionary["norm"] = None
        else:
            build_model_dictionary["norm"] = str(build_model_dictionary["norm"])
    else:
        print("Norm not specified in model dictionary")
        build_model_dictionary["norm"] = None
    if "feature_engineering" in build_model_dictionary.keys():
        build_model_dictionary["feature_engineering"] = str(build_model_dictionary["feature_engineering"])
    else:
        print("Feature engineering not specified in model dictionary")
        build_model_dictionary["feature_engineering"] = "0"
    if "adaptor_net_str" in build_model_dictionary.keys():
        build_model_dictionary["adaptor_net_str"] = str(build_model_dictionary["adaptor_net_str"])
    if "multihead" in build_model_dictionary.keys():
        build_model_dictionary["multihead"] = bool(eval(build_model_dictionary["multihead"]))
    else:
        build_model_dictionary["multihead"] = False
    if "channel_invariant" in build_model_dictionary.keys():
        build_model_dictionary["channel_invariant"] = bool(eval(build_model_dictionary["channel_invariant"]))

    return build_model_dictionary


def build_model_from_dict(build_model_dictionary):
    if build_model_dictionary["dim_in"] == 0 or build_model_dictionary["dim_in"] is None:
        dim_in = 3  # Channel invariance currently outputs a 3 channel image
    else:
        dim_in = build_model_dictionary["dim_in"]

    if "dropprob" not in build_model_dictionary.keys():
        build_model_dictionary["dropprob"] = 0.0

    if build_model_dictionary["model_str"] == "InstanSeg_UNet":
            from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet
            print("Generating InstanSeg_UNet")
            multihead = build_model_dictionary["multihead"]

            if build_model_dictionary["cells_and_nuclei"]:
                if not multihead:
                    from itertools import chain
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],1] for i in range(2)]
                    out_channels = list(chain(*out_channels))
                
                else:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],1] for i in range(2)]

            else:
                if not multihead:
                    out_channels = [[build_model_dictionary["dim_coords"], build_model_dictionary["n_sigma"],1]]
                else:
                    out_channels = [[build_model_dictionary["dim_coords"]], [build_model_dictionary["n_sigma"]],[1]]

            model = InstanSeg_UNet(in_channels=dim_in, 
                            layers = np.array(build_model_dictionary["layers"])[::-1],
                            out_channels=out_channels,
                            norm  = build_model_dictionary["norm"], 
                            dropout=build_model_dictionary["dropprob"])
            
    else:
        model = build_monai_model(build_model_dictionary["model_str"], build_model_dictionary)

    return model


def remove_module_prefix_from_dict(dictionary):
    """
    Removes the module prefix from a dictionary of model weights
    :param dictionary: dictionary of model weights
    :return: modified dictionary
    """
    modified_dict = {}
    for key, value in dictionary.items():
        if key.startswith('module.'):
            modified_dict[key[7:]] = value
        else:
            modified_dict[key] = value
    return modified_dict


def has_pixel_classifier_state_dict(state_dict):
    return bool(sum(['pixel_classifier' in key for key in state_dict.keys()]))


def has_adaptor_net_state_dict(state_dict):
    return bool(sum(['AdaptorNet' in key for key in state_dict.keys()]))

def has_pixel_classifier_model(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'pixel_classifier' in module_class or 'ProbabilityNet' in module_class:
                return True
    return False


def load_model_weights(model, device, folder, path=r"../models/", dict = None):
    import torch
    from pathlib import Path
    model_path = Path(path) / folder
    if torch.cuda.is_available():
        model_dict = torch.load(model_path / "model_weights.pth", weights_only= False)
    else:
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
                print('CUDA not available - attempting to load MPS model')
            else:
                device = 'cpu'
                print('CUDA not available - attempting to load CPU model')
        model_dict = torch.load(model_path / "model_weights.pth", map_location=device)

    model_dict['model_state_dict'] = remove_module_prefix_from_dict(model_dict['model_state_dict'])

    if has_pixel_classifier_state_dict(model_dict['model_state_dict']) and not has_pixel_classifier_model(model):
        from instanseg.utils.loss.instanseg_loss import InstanSeg

        method = InstanSeg(n_sigma=int(dict["n_sigma"]), feature_engineering_function= dict["feature_engineering"],dim_coords = dict["dim_coords"],device =device)
        model = method.initialize_pixel_classifier(model, MLP_width=int(dict["mlp_width"]))

    from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
    if has_adaptor_net_state_dict(model_dict['model_state_dict']) and not has_AdaptorNet(model):
        from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper, has_AdaptorNet
        model = AdaptorNetWrapper(model, norm = dict["norm"],adaptor_net_str = dict["adaptor_net_str"])

    #from instanseg.utils.AI_utils import set_running_stats
    #set_running_stats(model,device = "cuda")

    model.load_state_dict(model_dict['model_state_dict'], strict=True)
    model.to(device)

    return model, model_dict

def load_model(folder,path=r"../models/", device='cpu'):
    build_model_dictionary = read_model_args_from_csv(path=path, folder=folder)

    empty_model = build_model_from_dict(build_model_dictionary)

    model, _ = load_model_weights(empty_model, path=path, folder=folder, device=device, dict = build_model_dictionary)

    return model, build_model_dictionary
