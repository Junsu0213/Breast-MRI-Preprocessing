# -*- coding:utf-8 -*-

"""
Created on Wed. Aug. 28 17:24:07 2024
@author: JUN-SU Park

[DICOM to NIfTI Converter]

This script provides functionality to convert DICOM series to NIfTI format.

1. Reads a DICOM series from the input directory.
2. Converts the series into a NIfTI file and saves it to the output directory.

Example Usage:
    Run this script directly or import `convert_dicom_to_nii` in another project.
"""

import SimpleITK as sitk
import os
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")


def convert_dicom_to_nii(input_dir: str, output_dir: str = None, file_name: str = None) -> None:
    """
    Converts a DICOM series to a NIfTI file.

    Args:
        input_dir (str): Path to the directory containing the DICOM series.
        output_dir (str, optional): Path to save the converted NIfTI file. Defaults to a 'nii' folder in the parent directory of `input_dir`.
        file_name (str, optional): Name of the output NIfTI file. Defaults to the name of the `input_dir`.

    Returns:
        None: The function saves the converted NIfTI file directly to the specified `output_dir`.

    Raises:
        Exception: Ignores exceptions during processing for robustness.
    """
    # Set default output directory if not provided
    if output_dir is None:
        parent_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(parent_dir, 'nii')

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    # Set default file name if not provided
    if file_name is None:
        file_name = os.path.basename(input_dir)

    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Write NIfTI file
        sitk.WriteImage(image, os.path.join(output_dir, f'{file_name}.nii.gz'))
    except Exception as e:
        # Silently pass exceptions for robustness
        pass


if __name__ == '__main__':
    import os

    ds_dr = r'C:\Users\user\Desktop\DATSET\Test_Dataset\Coregistration_MRI'
    raw_dir = rf'{ds_dr}\raw'
    input_dir = rf'{ds_dr}\input'

    patient_id_list = os.listdir(raw_dir)

    for patient_id in patient_id_list:
        save_dir = rf'{input_dir}\{patient_id}'
        os.makedirs(save_dir, exist_ok=True)
        seq_dirs = [os.path.join(raw_dir, patient_id, f) for f in os.listdir(os.path.join(raw_dir, patient_id))]
        
        for seq_dir in seq_dirs:
            # print(seq_dir)
            if seq_dir.endswith('DWI'):
                dwi_seq_dirs = [os.path.join(seq_dir, f) for f in os.listdir(seq_dir)]
                for dwi_seq_dir in dwi_seq_dirs:
                    convert_dicom_to_nii(dwi_seq_dir, save_dir)
            else:
                convert_dicom_to_nii(seq_dir, save_dir)
