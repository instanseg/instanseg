import os
import pandas as pd
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse
import fastremap
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-d_p", "--data_path", type=str, default=r"../datasets")
parser.add_argument("-o_f", "--output_folder", type=str, default="Results")
parser.add_argument("-m_p", "--model_path", type=str, default=r"../models")
parser.add_argument("-m_f", "--model_folder", type=str)
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("-db", "--debug", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-save_ims", "--save_ims", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-data", "--dataset", type=str, default="segmentation", help="Name of the dataset to load")
parser.add_argument('-source', '--source_dataset', default=None, type=str)
parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'),help="Optimize postprocessing parameters")
parser.add_argument('-tta', '--tta', default=False, type=lambda x: (str(x).lower() == 'true'),help="Test time augmentations")
parser.add_argument('-target', '--target_segmentation', default=None, type=str,help=" Cells or nuclei or both? Accepts: C,N, NC")
parser.add_argument('-params', '--params', default="default", type=str, help="Either 'default' or 'best_params'")
parser.add_argument('-window', '--window_size', default=128, type=int)
parser.add_argument('-set', '--test_set', default="Validation", type=str, help = "Validation or Test or Train")
parser.add_argument('-export_to_torchscript', '--export_to_torchscript', default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-export_to_bioimageio', '--export_to_bioimageio', default=False,type=lambda x: (str(x).lower() == 'true'))


#@timer
def instanseg_inference(val_images, val_labels, model, postprocessing_fn, device, parser_args, output_path, params=None,
                        instanseg=None, tta=False):
    
    if tta:
        import ttach as tta

        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180, 90, 270]),
        ])

    from instanseg.utils.tiling import _instanseg_padding, _recover_padding
    count = 0
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

    #####
    #warmup
    #####
    imgs, _ = Augmenter.to_tensor(val_images[0], normalize=False)
    imgs = imgs.to(device)
    imgs, _ = Augmenter.normalize(imgs)
    with torch.no_grad():
        imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32)
        with torch.amp.autocast("cuda"):
            pred = model(imgs[None,])
        pred = pred.float()
        pred = _recover_padding(pred, pad).squeeze(0)
        if params is not None:
            with torch.amp.autocast("cuda"):
                lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
        else:
            with torch.amp.autocast("cuda"):
                lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)
        lab = lab.cpu().numpy()

        if tta:
            lab = method.TTA_postprocessing(imgs[None,], model, transforms, device=device)

    with torch.no_grad():
        for imgs, masks in tqdm(zip(val_images, val_labels), total=len(val_images)):

            torch.cuda.synchronize()
            start = time.time()
            imgs, masks = Augmenter.to_tensor(imgs, masks, normalize=False)
            imgs = imgs.to(device)
            imgs, _ = Augmenter.normalize(imgs)
            time_dict["preprocessing"] += time.time() - start
            torch.cuda.synchronize()
            start = time.time()

            if not tta:
                imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)
                with torch.amp.autocast("cuda"):
                    pred = model(imgs[None,])

                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)
                torch.cuda.synchronize()

                model_time = time.time() - start
                time_dict["model"] += model_time

                start = time.time()

                if params is not None:
                    with torch.amp.autocast("cuda"):
                        lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
                else:
                    with torch.amp.autocast("cuda"):
                        lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

                torch.cuda.synchronize()

                postprocessing_time = time.time() - start

                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({"time": model_time + postprocessing_time, "dimension": imgs.shape,
                                              "num_instances": len(torch.unique(lab) - 1)})

            else:
                if params is not None:
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms, **params,
                                                    window_size=parser_args.window_size, device=device)
                else:
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms, window_size=parser_args.window_size,
                                                    device=device)

            imgs = imgs.cpu().numpy()

            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            if isinstance(lab, torch.Tensor):
                lab = lab.cpu().numpy()

            count += 1

            lab, _ = fastremap.renumber(lab, in_place=True)
            masks[masks > 0], _ = fastremap.renumber(masks[masks > 0], in_place=True)

            pred_masks.append(lab.astype(np.int16))
            gt_masks.append(masks.astype(np.int16))

            if parser_args.save_ims:
                from instanseg.utils.augmentations import Augmentations
                augmenter = Augmentations()
                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(img, gt, return_image=True, alpha=0.8,
                                                         label_boundary_mode="thick", label_colors=color)

                show_images(overlay(display.numpy(), torch.tensor(lab)),
                            save_str=output_path / str("images/" "overlay" + str(count)))

    print("Time spent in preprocessing", time_dict["preprocessing"], "Time spent in model:", time_dict["model"],
          "Time spent in postprocessing:", time_dict["postprocessing"])

    return pred_masks, gt_masks, time_dict


if __name__ == "__main__":

    from instanseg.utils.utils import show_images, save_image_with_label_overlay, _move_channel_axis
    from instanseg.utils.model_loader import load_model
    from instanseg.utils.metrics import compute_and_export_metrics
    from instanseg.utils.augmentations import Augmentations

    parser_args = parser.parse_args()
    if parser_args.model_folder == "None":
        parser_args.model_folder = ""

    model, model_dict = load_model(path=parser_args.model_path, folder=parser_args.model_folder)

    data_path = Path(parser_args.data_path)
    os.environ["INSTANSEG_DATASET_PATH"] = str(parser_args.data_path)
    device = parser_args.device
    n_sigma = model_dict['n_sigma']

    model_path = Path(parser_args.model_path) / Path(parser_args.model_folder)

    parser_args.loss_function = model_dict['loss_function']
    if parser_args.source_dataset is None:
        parser_args.source_dataset = model_dict['source_dataset']

    if parser_args.loss_function.lower() == "instanseg_loss":
        from instanseg.utils.loss.instanseg_loss import InstanSeg

        method = InstanSeg(binary_loss_fn_str=model_dict["binary_loss_fn"], seed_loss_fn=model_dict["seed_loss_fn"],
                           n_sigma=model_dict["n_sigma"],
                           cells_and_nuclei=model_dict["cells_and_nuclei"], to_centre=model_dict["to_centre"],
                           window_size=parser_args.window_size, dim_coords=model_dict["dim_coords"],
                           feature_engineering_function=model_dict["feature_engineering"])

        if parser_args.target_segmentation is None:
            parser_args.cells_and_nuclei = model_dict["cells_and_nuclei"]
            parser_args.target_segmentation = model_dict["target_segmentation"]

        else:
            if len(parser_args.target_segmentation) == 2:
                parser_args.cells_and_nuclei = True
            else:
                parser_args.cells_and_nuclei = False

        parser_args.pixel_size = model_dict["pixel_size"]

        import math

        if math.isnan(parser_args.pixel_size):
            parser_args.pixel_size = None

        method.initialize_pixel_classifier(model)


        def loss_fn(*args, **kwargs):
            return method.forward(*args, **kwargs)


        def get_labels(pred, **kwargs):
            return method.postprocessing(pred, **kwargs, device=device, max_seeds=10000)


        dim_out = method.dim_out

    else:
        raise NotImplementedError("Loss function not recognized", parser_args.loss_function)

    model.eval()
    if "inference_folder" not in parser_args or parser_args.inference_folder is None:
        from instanseg.utils.data_loader import _read_images_from_pth

        val_images, val_labels, val_meta = _read_images_from_pth(args=parser_args, sets=[parser_args.test_set],
                                                                 dataset=parser_args.dataset)

    else:
        from instanseg.utils.data_loader import _read_images_from_path

        val_images, val_labels = _read_images_from_path(sets=[parser_args.test_set])

    datasets_str = np.unique([item['parent_dataset'] for item in val_meta])
    print("Datasets used:", datasets_str)

    model.to(device)

    output_path = model_path / parser_args.output_folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if parser_args.save_ims:
        if not os.path.exists(output_path / "images"):
            os.mkdir(output_path / "images")

    torch.cuda.empty_cache()
    count = 0


    Augmenter = Augmentations(dim_in=model_dict['dim_in'], shape=None,
                              channel_invariant=model_dict['channel_invariant'])

    val_data = [Augmenter.duplicate_grayscale_channels(*Augmenter.to_tensor(img, label, normalize= True)) for img, label in
            zip(val_images, val_labels)]

    # val_data = [(img, label) for img, label in zip(val_images, val_labels)]
    # val_data = [Augmenter.normalize(img,label,percentile=0.) for img, label in val_data]

    if parser_args.pixel_size is not None and parser_args.pixel_size != "None":
        print("Warning, rescaling image and ground truth labels to pixel size:", parser_args.pixel_size,
              "microns/pixel")

        for i, (img, label) in enumerate(val_data):
            if "pixel_size" not in val_meta[i].keys():
                val_meta[i]["pixel_size"] = 0.5

            elif val_meta[i]["pixel_size"] == "pixel_size":
                val_meta[i]["pixel_size"] = 0.5  #bug in mesmer dataset

        val_data = [Augmenter.torch_rescale(img, label, current_pixel_size=val_meta[i]['pixel_size'],
                                            requested_pixel_size=parser_args.pixel_size,
                                            modality=val_meta[i]["image_modality"], crop=False) for i, (img, label) in
                    enumerate(val_data)]

    # val_data = [Augmenter.colourize(img,label,c_nuclei = val_meta[i]['nuclei_channels'][0]) for i, (img,label) in enumerate(val_data)]

    # from instanseg.utils.augmentations import get_marker_location
    # val_meta = [get_marker_location(meta) for meta in val_meta]
    # val_data = [Augmenter.extract_nucleus_and_cytoplasm_channels(img,label,c_nuclei = val_meta[i]['nuclei_channels'][0],metadata = val_meta[i]) for i, (img,label) in enumerate(val_data)]

    val_images = [item[0] for item in val_data]  #[::-1]
    val_labels = [item[1] for item in val_data]  #[::-1]

    from instanseg.utils.utils import count_instances

    freq = np.array([count_instances(label) for label in val_labels])
    area = np.array(
        [(len(label[label > 0].flatten())) / f for f, label in
         zip(freq, val_labels)])  # this will break for images with 0 labels

    print("Found:", sum(freq), "instances, across", len(freq), "images.", "Median area:", np.median(area), "pixels")

    if parser_args.optimize_hyperparameters:
        from instanseg.utils.AI_utils import optimize_hyperparameters

        params = optimize_hyperparameters(model, postprocessing_fn=method.postprocessing, val_images=val_images,
                                          val_labels=val_labels, verbose=True)
        pd.DataFrame.from_dict(params, orient='index').to_csv(output_path / "best_params.csv",
                                                              header=False)
    else:
        if parser_args.params == "default":
            params = None
        else:
            df = pd.read_csv(output_path / "best_params.csv", header=None)
            params = {row[0]: row[1] for row in df.values}

    # params["window_size"] = parser_args.window_size
    instanseg = None
    if parser_args.export_to_torchscript:
        from instanseg.utils.utils import export_to_torchscript

        print("Exporting model to torchscript")
        export_to_torchscript(parser_args.model_folder)
        instanseg = torch.jit.load("../torchscripts/" + parser_args.model_folder + ".pt")
    if parser_args.export_to_bioimageio:
        print("Exporting model to bioimageio")
        from instanseg.utils.create_bioimageio_model import export_bioimageio

        instanseg = torch.jit.load("../torchscripts/" + parser_args.model_folder + ".pt")
        export_bioimageio(instanseg, deepimagej=True, test_img_path="../examples/HE_example.tif",
                          model_name=parser_args.model_folder)

    pred_masks, gt_masks, time_dict = instanseg_inference(val_images,
                                                          val_labels,
                                                          model,
                                                          postprocessing_fn=get_labels,
                                                          device=device,
                                                          parser_args=parser_args,
                                                          output_path=output_path,
                                                          params=params,
                                                          instanseg=instanseg,
                                                          tta=parser_args.tta)

    pd.DataFrame(time_dict['combined']).to_csv(output_path / "timing_dict.csv", header=True)

    if parser_args.cells_and_nuclei:
        pred_nuclei_masks = [pred_mask[0] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[0].min() >= 0]
        gt_nuclei_masks = [gt_mask[0] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[0].min() >= 0]

        pred_cell_masks = [pred_mask[1] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[1].min() >= 0]
        gt_cell_masks = [gt_mask[1] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[1].min() >= 0]

        compute_and_export_metrics(gt_nuclei_masks, pred_nuclei_masks, output_path, target="Nuclei")
        compute_and_export_metrics(gt_cell_masks, pred_cell_masks, output_path, target="Cells")
    else:
        pred_masks = [(pred_mask).squeeze()[None] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if
                      gt_mask.min() >= 0]
        gt_masks = [(gt_mask).squeeze()[None] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask.min() >= 0]
        compute_and_export_metrics(gt_masks, pred_masks, output_path,
                                   target="Cells" if parser_args.target_segmentation == "C" else "Nuclei")

