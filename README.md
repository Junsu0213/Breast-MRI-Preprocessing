# Breast MRI Preprocessing Pipeline

This repository contains a preprocessing pipeline for breast MRI data, designed to prepare medical images for AI model analysis. The pipeline handles DICOM to NIfTI conversion and image coregistration for multiple MRI sequences.

## Overview

The pipeline processes breast MRI data with multiple sequences (T1WI, T2WI, DWI, ADC) and includes the following steps:
1. DICOM to NIfTI conversion
2. Normalization of image intensities
3. Image coregistration (optional)
4. Automatic organization of processed files

## Input Data Structure

The expected input data structure is as follows:
```
└─Input_Directory
    ├─PatientID_1
    │  ├─ADC_map
    │  ├─DWI
    │  ├─FS_T1WI
    │  ├─NonFS_T1WI
    │  ├─reference_CE_MRI
    │  └─T2WI
    ├─PatientID_2
    │  ├─ADC_map
    │  ├─DWI
    │  ├─FS_T1WI
    │  └─...
    └─...
```

Each patient directory contains multiple MRI sequences in DICOM format:
- FS_T1WI: Fat-suppressed T1-weighted images
- NonFS_T1WI: Non-fat-suppressed T1-weighted images
- T2WI: T2-weighted images
- DWI: Diffusion-weighted images (includes multiple b-values: b0, b800, b1200)
- ADC_map: Apparent Diffusion Coefficient maps
- reference_CE_MRI: Contrast-enhanced reference images

## Output Structure

The processed data is organized as follows:
```
└─Output_Directory
    ├─input
    │   ├─b800
    │   │   └─PatientID
    │   │       ├─PatientID_T1FS_wm.nii.gz
    │   │       ├─PatientID_T2FS_wm.nii.gz
    │   │       ├─PatientID_ADC_wm.nii.gz
    │   │       ├─PatientID_B0_wm.nii.gz
    │   │       └─PatientID_DWI_wm.nii.gz
    │   └─b1200
    │       └─PatientID
    │           └─...
    └─output
        ├─b800
        │   └─PatientID
        │       └─reference_CE_MRI.nii.gz
        └─b1200
            └─PatientID
                └─reference_CE_MRI.nii.gz
```

## Features

- Automatic DICOM to NIfTI conversion
- Handles multiple MRI sequences
- Special processing for DWI sequences with different b-values
- Image intensity normalization
- Optional image coregistration (supports rigid and affine transformations)
- Organized output structure for AI model input
- Progress tracking with tqdm
- Comprehensive error handling

## Requirements

```
numpy
antspy
SimpleITK
pydicom
tqdm
```

## Installation

```bash
git clone https://github.com/yourusername/breast-mri-preprocessing.git
cd breast-mri-preprocessing
pip install -r requirements.txt
```

## Usage

Basic usage:

```python
from mri_preprocessing_ver_01 import DCM2NIIConverter, Coregistration

# Convert DICOM to NIfTI
input_dir = '/path/to/dicom/data'
converter = DCM2NIIConverter(input_dir)
converter.convert_all()

# Perform coregistration
output_dir = '/path/to/output'
coregistration = Coregistration(output_dir)

# Without registration (default)
coregistration.process_all()

# With registration (optional)
coregistration.process_all(registration_type='rigid')  # or 'affine' or 'both'
```

## Registration Options

The pipeline supports different registration types:
- No registration (default): Only performs interpolation
- Rigid registration: Corrects for rotation and translation
- Affine registration: Corrects for rotation, translation, scaling, and shearing
- Both: Applies both rigid and affine transformations sequentially

## Note

- The pipeline uses FS T1WI as the reference image for coregistration
- All images are normalized to the range [0,1]
- DWI sequences are automatically sorted by b-values
- Progress bars show processing status for each step
