import collections

def get_augmentation_dict(dim_in,nuclei_channel,amount,pixel_size=0.5, augmentation_type="minimal"):

    """
    This function returns the augmentation dictionary for the training and test sets.

    Args:
        image_modality (str): The image modality. Options are ["Brightfield","Fluorescence","Chromogenic"]
        dim_in (int): The number of input channels
        nuclei_channel (int): The channel that contains the nuclei
        amount (float): The amount of augmentation to apply (between 0 and 1)
        minmax (tuple): The min and max values instance surface area to rescale the image to. If None, rescaling is done on a per image basis.
    
    In the augmentations.py file, the image modality is automatically determined by checking if the mean pixel
    values under the labels is darker than the mean pixel values of the background. If so, the image is assumed to be brightfield.
    
    """

    channel_invariance = (dim_in is None or dim_in <= 0)

    if augmentation_type == "minimal":


        augmentation_dict = {
            "train": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]), #Probability
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("flips", [1]),#Probability
                    ("rotate", [1]),#Probability
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel,
                    ("channel_subsample", [0, (5 if channel_invariance else dim_in , 20 if channel_invariance else dim_in)]),  #proba,(min,max) #(1, 1)]), #
                    ("flips", [1]),
                    ("rotate", [1]),
                ]) 
            },
            "test": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("flips", [1])
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("channel_subsample", [0, (5 if channel_invariance else dim_in , 20 if channel_invariance else dim_in)]),  #proba,(min,max)  (1, 1)]), #
                    ("flips", [1])
                ])
            }
        }

    elif augmentation_type == "heavy":
            
        augmentation_dict = {
            "train": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]), #Probability
                    ("normalize_HE_stains", [0.1, amount*0]), #Probability/Amount, make sure this goes in front of normalize
                    ("extract_hematoxylin_stain", [0.1, amount*0]), #Probability/Amount ,make sure this goes in front of normalize
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("randomJPEGcompression", [0.2, amount]),  #Probability/Amount
                    ("adjust_hue", [0.2, amount]),#Probability/Amount
                    ("AdjustContrast", [0.2, amount]),#Probability/Amount
                    ("flips", [1]),#Probability
                    ("rotate", [1]),#Probability
                    ("brightness_augment", [0.2, amount]),#Probability/Amount
                    ("RandGaussianNoise", [0.2, amount]),#Probability/Amount
                    ("perspective", [0, amount]),#Probability/Amount
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("pseudo_brightfield", [0, nuclei_channel]),
                    ("randomJPEGcompression", [0.2, amount]),
                    ("extract_nucleus_and_cytoplasm_channels", [0.05, amount]),
                    ("pseudo_imc", [0, amount]),
                    ("colourize", [0.1, nuclei_channel]),
                   # ("draw_shapes", [0.05, amount]),
                    ("flips", [1]),
                    ("rotate", [1]),
                    ("perspective", [0, amount]),
                    ("add_gradient", [0.05, amount]),
                    ("brightness_augment", [0.2, amount]),
                    ("RandGaussianNoise", [0.1, amount]),
                    ("HistogramNormalize", [0.1, amount]),
                    ("add_noisy_channels", [0.3, 5]),#Probability/ max total channels
                    ("channel_suppress", [1, 0.3]),  #proba, supression_factor
                ]) 
            },
            "test": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("flips", [1])
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("pseudo_brightfield", [0, nuclei_channel]),
                    ("extract_nucleus_and_cytoplasm_channels", [0, amount]),
                   # ("channel_subsample", [0, (5 if channel_invariance else dim_in , 20 if channel_invariance else dim_in)]),  #proba,(min,max)  (1, 1)]), #
                    ("colourize", [0, nuclei_channel]),
                    ("flips", [1])

                ])
            }
        }


    elif augmentation_type == "two_channel":
            
        augmentation_dict = {
            "train": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]), #Probability
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("extract_nucleus_and_cytoplasm_channels", [1, amount]),
                    ("flips", [1]),
                    ("rotate", [1]),
                ]) 
            },
            "test": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]),
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("extract_nucleus_and_cytoplasm_channels", [1, amount]),
                    ("flips", [1])
                ])
            }
        }


    elif augmentation_type == "colourize":
            
        augmentation_dict = {
            "train": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]), #Probability
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("colourize", [1, nuclei_channel]),
                    ("flips", [1]),
                    ("rotate", [1]),
                ]) 
            },
            "test": {
                "Brightfield": collections.OrderedDict([
                    ("to_tensor", [1]),
                ]),
                "Fluorescence": collections.OrderedDict([
                    ("to_tensor", [1]),
                    ("normalize", [1]), #Probability
                    ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                    ("colourize", [1, nuclei_channel]),
                    ("flips", [1])
                ])
            }
        }


    elif augmentation_type == "brightfield_only":
            

            augmentation_dict = {
                "train": {
                    "Brightfield": collections.OrderedDict([
                        ("to_tensor", [1]), #Probability
                        ("normalize_HE_stains", [0.1, amount*0]), #Probability/Amount, make sure this goes in front of normalize
                        ("extract_hematoxylin_stain", [0.1, amount*0]), #Probability/Amount ,make sure this goes in front of normalize
                        ("normalize", [1]), #Probability
                        ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                        ("randomJPEGcompression", [0.2, amount]),  #Probability/Amount
                        ("adjust_hue", [0.2, amount]),#Probability/Amount
                        ("AdjustContrast", [0.2, amount]),#Probability/Amount
                        ("flips", [1]),#Probability
                        ("rotate", [1]),#Probability
                        ("brightness_augment", [0.2, amount]),#Probability/Amount
                        ("RandGaussianNoise", [0.2, amount]),#Probability/Amount
                        ("perspective", [0.1, amount]),#Probability/Amount
                    ]),
                    "Fluorescence": collections.OrderedDict([
                        ("to_tensor", [1]),
                        ("normalize", [1]), #Probability
                        ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                        ("pseudo_brightfield", [1, nuclei_channel]),
                        ("randomJPEGcompression", [0.2, amount]),
                        ("adjust_hue", [0.2, amount]),#Probability/Amount
                        ("AdjustContrast", [0.2, amount]),#Probability/Amount
                        ("flips", [1]),
                        ("rotate", [1]),
                        ("brightness_augment", [0.2, amount]),
                        ("RandGaussianNoise", [0.1, amount]),
                        ("HistogramNormalize", [0.1, amount]),
                    ]) 
                },
                "test": {
                    "Brightfield": collections.OrderedDict([
                        ("to_tensor", [1]),
                        ("normalize", [1]), #Probability
                        ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                        ("flips", [1])
                    ]),
                    "Fluorescence": collections.OrderedDict([
                        ("to_tensor", [1]),
                        ("normalize", [1]), #Probability
                        ("torch_rescale", [1,pixel_size, 0]),#in microns per pixel
                        ("pseudo_brightfield", [1, nuclei_channel]),
                        ("flips", [1])
                    ])
                }
            }
        
    else:
        raise ValueError("Invalid augmentation type. Options are ['minimal','heavy','brightfield_only']")
        

    return augmentation_dict

#Experimentally determined from CPMD 2023 annotations.
markers_info = {
'GITR': {'Subcellular Location': 'Cytoplasm'},
 'IDO': {'Subcellular Location': 'Nucleus'},
 'Ki67': {'Subcellular Location': 'Nucleus'},
 'Foxp3': {'Subcellular Location': 'Nucleus'},
 'CD8': {'Subcellular Location': 'Nucleus'},
 'DAPI': {'Subcellular Location': 'Nucleus'},
 'panCK': {'Subcellular Location': 'Cytoplasm'},
 'Autofluorescence': {'Subcellular Location': 'Cytoplasm'},
 'CD40-L': {'Subcellular Location': 'Nucleus'},
 'PD-1': {'Subcellular Location': 'Cytoplasm'},
 'CD40': {'Subcellular Location': 'Cytoplasm'},
 'PD-L1': {'Subcellular Location': 'Cytoplasm'},
 'PD1': {'Subcellular Location': 'Nucleus'},
 'PDL1': {'Subcellular Location': 'Cytoplasm'},
 'PD-L2': {'Subcellular Location': 'Cytoplasm'},
 'CD30': {'Subcellular Location': 'Cytoplasm'},
 'MHC-I': {'Subcellular Location': 'Cytoplasm'},
 'MUM1': {'Subcellular Location': 'Nucleus'},
 'Hoechst': {'Subcellular Location': 'Nucleus'},
 'Class-II': {'Subcellular Location': 'Cytoplasm'},
 'ICOS': {'Subcellular Location': 'Nucleus'},
 'CTLA4': {'Subcellular Location': 'Nucleus'},
 'TCF1': {'Subcellular Location': 'Nucleus'},
 'panCK+CK7+CAM5.2': {'Subcellular Location': 'Cytoplasm'},
 'LAG3': {'Subcellular Location': 'Cytoplasm'},
 'CD68': {'Subcellular Location': 'Cytoplasm'},
 'CD4': {'Subcellular Location': 'Cytoplasm'},
 'CD163': {'Subcellular Location': 'Cytoplasm'},
 'P63': {'Subcellular Location': 'Nucleus'},
 'Arg-1': {'Subcellular Location': 'Cytoplasm'},
 'CD11b': {'Subcellular Location': 'Nucleus'},
 'MHC-II': {'Subcellular Location': 'Cytoplasm'},
 'CK': {'Subcellular Location': 'Cytoplasm'},
 'DAPI1': {'Subcellular Location': 'Nucleus'},
 'NA': {'Subcellular Location': 'Cytoplasm'},
 'DAPI2': {'Subcellular Location': 'Nucleus'},
 'CD3': {'Subcellular Location': 'Cytoplasm'},
 'CD20': {'Subcellular Location': 'Cytoplasm'},
 'DAPI3': {'Subcellular Location': 'Nucleus'},
 'PanCK': {'Subcellular Location': 'Cytoplasm'},
 'DAPI4': {'Subcellular Location': 'Nucleus'},
 'CD21': {'Subcellular Location': 'Cytoplasm'},
 'CD31': {'Subcellular Location': 'Cytoplasm'},
 'DAPI5': {'Subcellular Location': 'Nucleus'},
 'CD45RO': {'Subcellular Location': 'Cytoplasm'},
 'CD11c': {'Subcellular Location': 'Cytoplasm'},
 'DAPI6': {'Subcellular Location': 'Nucleus'},
 'HLA-DR': {'Subcellular Location': 'Cytoplasm'},
 'DAPI7': {'Subcellular Location': 'Nucleus'}}


# This is the output from chat gpt, it seems to match the protein atlas data (for 14 markers I randomly checked). Use with caution!
markers_info_gpt = {
    "CD66B": {
        "Cellular Location": "Granulocytes",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Granulocyte marker",
        "Application": "Identify and quantify granulocytes in tissues"
    },
    "CD68": {
        "Cellular Location": "Macrophages",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "General macrophage marker",
        "Application": "Identify and quantify macrophages in tissues"
    },
    "CK7": {
        "Cellular Location": "Epithelial cells",
        "Subcellular Location": "Cytoplasm",
        "Role": "Cytokeratin marker for epithelial cells",
        "Application": "Identify and visualize epithelial cells in tissues"
    },
    "CTLA4": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Negative regulator of T cell activation",
        "Application": "Study T cell activation and immune regulation"
    },
    "FOXP3": {
        "Cellular Location": "Regulatory T cells (Tregs)",
        "Subcellular Location": "Nucleus",
        "Role": "Transcription factor for Tregs",
        "Application": "Identify and quantify regulatory T cells in tissues"
    },
    "GITR": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "T cell activation marker",
        "Application": "Study T cell activation and immune responses"
    },
    "GZMB": {
        "Cellular Location": "Cytotoxic T cells, NK cells",
        "Subcellular Location": "Cytoplasm",
        "Role": "Serine protease in cytotoxic cells",
        "Application": "Assess cytotoxic activity of T cells and NK cells"
    },
    "ICOS": {
        "Cellular Location": "Activated T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Co-stimulatory molecule on T cells",
        "Application": "Identify activated T cells in the tissue"
    },
    "IDO": {
        "Cellular Location": "Macrophages, dendritic cells",
        "Subcellular Location": "Cytoplasm",
        "Role": "Indoleamine 2,3-dioxygenase; immune suppressor",
        "Application": "Assess immunosuppressive environment in tissues"
    },
    "ARG-1": {
        "Cellular Location": "Myeloid cells",
        "Subcellular Location": "Cytoplasm",
        "Role": "Urea cycle; associated with M2 macrophages",
        "Application": "Identify M2 macrophages in the tumor microenvironment"
    },
    "CD11B": {
        "Cellular Location": "Myeloid cells",
        "Subcellular Location": "Membrane",
        "Role": "Adhesion molecule for immune cell interactions",
        "Application": "Myeloid cell marker in immunohistochemistry"
    },
    "CD138": {
        "Cellular Location": "Plasma cells",
        "Subcellular Location": "Membrane",
        "Role": "Plasma cell marker",
        "Application": "Identify and quantify plasma cells, often in hematologic malignancies"
    },
    "CD163": {
        "Cellular Location": "Macrophages",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "M2 macrophage marker",
        "Application": "Identify M2 macrophages in tissue sections"
    },
    "CD20": {
        "Cellular Location": "B cells",
        "Subcellular Location": "Membrane",
        "Role": "B cell marker",
        "Application": "Identify and quantify B cells in tissues, especially in lymphomas"
    },
    "CD3": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "T cell marker",
        "Application": "Identify and quantify T cells in tissues"
    },
    "CD30": {
        "Cellular Location": "B and T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Marker for lymphoma, especially Hodgkin's lymphoma",
        "Application": "Used in the diagnosis of lymphomas"
    },
    "CD4": {
        "Cellular Location": "Helper T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Helper T cell marker",
        "Application": "Identify and quantify helper T cells in tissues"
    },
    "CD40": {
        "Cellular Location": "B cells, dendritic cells",
        "Subcellular Location": "Membrane",
        "Role": "Important in B cell activation",
        "Application": "Study B cell function and activation"
    },
    "CD40L": {
        "Cellular Location": "Activated T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Critical for B cell activation",
        "Application": "Study T cell-B cell interactions"
    },
    "KI67": {
        "Cellular Location": "Nuclei",
        "Subcellular Location": "Nucleus",
        "Role": "Proliferation marker",
        "Application": "Assess cell proliferation rate in tissues"
    },
    "LAG3": {
        "Cellular Location": "Activated T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Immune checkpoint receptor",
        "Application": "Study immune checkpoint regulation"
    },
    "MHC-I": {
        "Cellular Location": "Nucleated cells",
        "Subcellular Location": "Membrane",
        "Role": "Major histocompatibility complex class I",
        "Application": "Antigen presentation to CD8+ T cells"
    },
    "MHC-II": {
        "Cellular Location": "Antigen-presenting cells",
        "Subcellular Location": "Membrane",
        "Role": "Major histocompatibility complex class II",
        "Application": "Antigen presentation to CD4+ T cells"
    },
    "MUM1": {
        "Cellular Location": "B cells, plasma cells",
        "Subcellular Location": "Nucleus",
        "Role": "B cell and plasma cell marker",
        "Application": "Identify and quantify B cells and plasma cells in tissues"
    },
    "P63": {
        "Cellular Location": "Basal and squamous epithelial cells",
        "Subcellular Location": "Nucleus",
        "Role": "Epithelial cell marker",
        "Application": "Identify and visualize basal and squamous epithelial cells"
    },
    "PANCK": {
        "Cellular Location": "Epithelial cells",
        "Subcellular Location": "Cytoplasm",
        "Role": "Pan-cytokeratin marker for epithelial cells",
        "Application": "Identify and visualize epithelial cells in tissues"
    },
    "PAX8": {
        "Cellular Location": "Certain epithelial cells",
        "Subcellular Location": "Nucleus",
        "Role": "Transcription factor",
        "Application": "Marker for certain epithelial tissues, including ovarian and thyroid"
    },
    "PD-1": {
        "Cellular Location": "Activated T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Immune checkpoint receptor",
        "Application": "Study immune checkpoint regulation"
    },
    "PD-L1": {
        "Cellular Location": "Tumor cells, immune cells",
        "Subcellular Location": "Membrane",
        "Role": "Programmed Death-Ligand 1",
        "Application": "Assess expression in tumor microenvironment, response to immunotherapy"
    },
    "PD-L2": {
        "Cellular Location": "Tumor cells, immune cells",
        "Subcellular Location": "Membrane",
        "Role": "Programmed Death-Ligand 2",
        "Application": "Assess expression in tumor microenvironment, response to immunotherapy"
    },
    "TCF1": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Nucleus",
        "Role": "Transcription factor",
        "Application": "Regulator of T cell development and function"
    },
    "TOX": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Nucleus",
        "Role": "Transcription factor",
        "Application": "Regulator of T cell exhaustion and dysfunction"
    },
    "VISTA": {
        "Cellular Location": "Immune cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Immune checkpoint receptor",
        "Application": "Study immune checkpoint regulation"
    },
    "CD8": {
        "Cellular Location": "Cytotoxic T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Cytotoxic T cell marker",
        "Application": "Identify and quantify cytotoxic T cells in tissues"
    },
    "DAPI": {
        "Cellular Location": "Nuclei",
        "Subcellular Location": "Nucleus",
        "Role": "DNA-binding dye for cell nuclei",
        "Application": "Stain cell nuclei for visualization in microscopy"
    },
    "HOECHST": {
        "Cellular Location": "Nuclei",
        "Subcellular Location": "Nucleus",
        "Role": "DNA-binding dye for cell nuclei",
        "Application": "Stain cell nuclei for visualization in microscopy"
    },
    "SOX10": {
        "Cellular Location": "Neural crest-derived cells",
        "Subcellular Location": "Nucleus",
        "Role": "Transcription factor",
        "Application": "Marker for neural crest-derived cells, especially in neuroectodermal tumors"
    },
    "CD21": {
        "Cellular Location": "B cells, follicular dendritic cells",
        "Subcellular Location": "Membrane",
        "Role": "Complement receptor 2",
        "Application": "Identify B cells and follicular dendritic cells in lymphoid tissues"
    },
    "CD31": {
        "Cellular Location": "Endothelial cells",
        "Subcellular Location": "Membrane",
        "Role": "Platelet endothelial cell adhesion molecule (PECAM-1)",
        "Application": "Marker for endothelial cells and angiogenesis"
    },
    "CD3E": {
        "Cellular Location": "T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Part of the T cell receptor complex",
        "Application": "Identify and quantify T cells in tissues"
    },
    "CD45RO": {
        "Cellular Location": "Memory T cells",
        "Subcellular Location": "Membrane/Cytoplasm",
        "Role": "Memory T cell marker",
        "Application": "Identify and quantify memory T cells in tissues"
    },
    "HLA-DR": {
        "Cellular Location": "Antigen-presenting cells",
        "Subcellular Location": "Membrane",
        "Role": "Major histocompatibility complex class II",
        "Application": "Antigen presentation to CD4+ T cells"
    },
    "Autofluorescence": {
        "Cellular Location": "N/A", 
        "Subcellular Location": "N/A",
        "Role": "N/A",
        "Application": "N/A"
    }
}



