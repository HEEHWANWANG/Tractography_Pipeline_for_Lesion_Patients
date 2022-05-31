# Tractography_Pipeline_for_Lesion_Patients
Probablistic Tractography (i.e., mrtrix single shell CSD and multishell CSD with Anatomical Constraint) Pipeline for Patients with focal Brain Lesion (e.g., Ischemic Stroke, )


## Step 1. Lesion Mask Generation 
Generating **_brain lesion mask_** in a fully-automated way.  
By using pre-trained deep learning model (i.e., **_TransUNet_**), generating a **_brain lesion mask_** image from **_T1w image_**. 

## Step 2. Prerpocessing T1w and DWI images with brain lesion mask 
Running QSIprep preprocessing process. 

## Step 3. Freesurfer ```recon-all``` with the Virtual Brain Grfting
Running **_the Virtual Brain Grafting_** with **_preprocessed T1w image_** and **_brain lesion mask_**.  
That's because Freesufer ```recon-all``` could not run with T1w image with focal brain lesion.  
The Virtual Brain Grafting is a software using brain lesion mask to fill lesion and running Freesurfer ```recon-all```.

## Step 4. Estimating Anatomical Constraint Tractography with results from Freesurfer ```recon-all```
- Running **_5 tissue type (5tt image) Hypbrid Surface and Volume Segmentation_** via mrtrix3 (https://mrtrix.readthedocs.io/en/latest/reference/commands/5ttgen.html). Once obtaining **_5tt image_**, excluding lesion spots by using **_brain lesion mask_** (https://mrtrix.readthedocs.io/en/latest/reference/commands/5ttedit.html).
- Getting _**response function**_ for _**Constrained Spherical Deconvolution**_ (https://mrtrix.readthedocs.io/en/dev/reference/commands/dwi2response.html).
- Getting **_FOD (Fiber Orientation Distribution) images_** with **_response function_** and **_preprocessed dwi images_** (https://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2fod.html).
- Getting Anatomical Constraint Tracktogram file (i.e., .tck) with **_FOD images_** and **_5tt images_** (https://mrtrix.readthedocs.io/en/latest/reference/commands/tckgen.html).
- Running SIFT2 with **_Anatomical Constraint Tracktogram file_** and **_FOD images_** (https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html).
- Getting **_connectivity matrix_** with various brain atlases including individual atlas obtained from Freesurfer ```recon-all``` (https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html). 
- Getting **_FA (Fractional Anisotropy) images_** with **_preprocessed T1w image_** and **_brain mask_** obtained from QSIprep (i.e, ANT's brain extraction) (https://mrtrix.readthedocs.io/en/dev/reference/commands/dwi2tensor.html).
- With **_Anatomical Constraint Tracktogram file_**, **_results from SIFT2 (txt file)_**, **_FA image_**, and **_brain mask_**, running **_automatic fiber bundle segmentation via MICAPIPE_** (https://micapipe.readthedocs.io/en/latest/pages/05.autotract/index.html)  
