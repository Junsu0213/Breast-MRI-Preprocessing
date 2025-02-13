# -*- coding:utf-8 -*-
"""
Created on Thu. Oct. 31 10:14:31 2024
@author: JUN-SU PARK

MRI Data Preprocessing Utilities

This module provides utilities for:
1. Converting DICOM files to NIfTI format
2. Performing image coregistration using ANTs
"""
import os
import numpy as np
import ants
import shutil
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple, Any
from ants.core.ants_image import ANTsImage


class DCM2NIIConverter:
    """
    Converts DICOM image series to NIfTI format.
    Handles special cases for DWI sequences with different b-values.
    """

    def __init__(self, input_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the converter.

        Args:
            input_dir: Directory containing DICOM files
            output_dir: Output directory for NIfTI files (default: './tmp')
        """
        self.input_dir = input_dir
        self.output_dir = output_dir or './tmp'

    def convert_dicom_to_nii(self, dicom_dir: str, patient_name: str) -> None:
        """
        Convert a DICOM series to NIfTI format.

        Args:
            dicom_dir: Directory containing DICOM series
            patient_name: Patient identifier for output naming
        """
        output_path = os.path.join(self.output_dir, patient_name)
        os.makedirs(output_path, exist_ok=True)

        sequence_name = os.path.basename(dicom_dir)

        # Read DICOM series and convert to NIfTI
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        output_file = os.path.join(output_path, f'{sequence_name}.nii.gz')
        sitk.WriteImage(image, output_file)

    def split_dwi_files(self, file_dir: str) -> None:
        """
        Split DWI files into separate directories based on b-values.

        Args:
            file_dir: Directory containing DWI DICOM files
        """
        dwi_files = os.listdir(file_dir)
        for dwi_file in dwi_files:
            dwi_file_path = os.path.join(file_dir, dwi_file)

            try:
                dcm = pydicom.dcmread(dwi_file_path)
                sequence_name = dcm.SequenceName if hasattr(dcm, 'SequenceName') else 'Unknown'

                # Determine target directory based on b-value
                if 'b0' in sequence_name:
                    target_dir = os.path.join(file_dir, 'DWI_b0')
                elif 'b800' in sequence_name:
                    target_dir = os.path.join(file_dir, 'DWI_b800')
                elif 'b1200' in sequence_name:
                    target_dir = os.path.join(file_dir, 'DWI_b1200')
                else:
                    target_dir = os.path.join(file_dir, 'DWI')

                os.makedirs(target_dir, exist_ok=True)
                shutil.move(dwi_file_path, os.path.join(target_dir, dwi_file))
            except Exception:
                pass

    def convert_all(self) -> None:
        """
        Convert all DICOM series in the input directory to NIfTI format.
        Handles DWI sequences specially by separating them based on b-values.
        """
        for patient in tqdm(os.listdir(self.input_dir), desc="Converting DICOM to NIfTI"):
            patient_dir = os.path.join(self.input_dir, patient)

            for sequence in os.listdir(patient_dir):
                sequence_dir = os.path.join(patient_dir, sequence)

                if 'DWI' in sequence:
                    try:
                        # Split DWI files by b-value
                        self.split_dwi_files(sequence_dir)

                        # Convert each b-value directory
                        for b_value_dir in os.listdir(sequence_dir):
                            if os.path.isdir(os.path.join(sequence_dir, b_value_dir)):
                                self.convert_dicom_to_nii(
                                    os.path.join(sequence_dir, b_value_dir),
                                    patient
                                )
                    except Exception as e:
                        print(f"Error processing DWI sequence for {patient}: {e}")
                        continue
                else:
                    try:
                        self.convert_dicom_to_nii(sequence_dir, patient)
                    except Exception as e:
                        print(f"Error converting sequence {sequence} for {patient}: {e}")

        print("\nDICOM to NIfTI conversion completed!")


class Coregistration:
    """
    Performs image coregistration using ANTs.
    Supports both simple interpolation and full registration (rigid + affine).
    Default is interpolation only, with optional registration if specified.
    """
    SEQUENCE_MAPPING = {
        'ADC map.nii.gz': 'ADC_wm.nii.gz',
        'ADC_map.nii.gz': 'ADC_wm.nii.gz',
        'DWI_b0.nii.gz': 'B0_wm.nii.gz',
        'DWI_b800.nii.gz': 'DWI_wm.nii.gz',
        'DWI_b1200.nii.gz': 'DWI_wm.nii.gz',
        'NonFS T1WI.nii.gz': 'T1_wm.nii.gz',
        'NonFS_T1WI.nii.gz': 'T1_wm.nii.gz',
        'T2WI.nii.gz': 'T2FS_wm.nii.gz'
    }

    DWI_SEQUENCES = [
        'DWI_b0.nii.gz', 'DWI_b800.nii.gz', 'DWI_b1200.nii.gz',
        'ADC map.nii.gz', 'ADC_map.nii.gz'
    ]

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize coregistration processor.

        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = output_dir

    @staticmethod
    def normalize_image(image: ANTsImage) -> ANTsImage:
        """
        Normalize image intensity to [0,1] range.

        Args:
            image: Input ANTs image
        Returns:
            Normalized ANTs image
        """
        img_array = image.numpy()
        img_array = img_array.astype(np.float32)
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))

        return ants.from_numpy(
            img_array,
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction
        )

    @staticmethod
    def get_transforms(fixed: ANTsImage, moving: ANTsImage,
                       registration_type: Optional[str] = None) -> List[str]:
        """
        Generate transformation files based on registration type.

        Args:
            fixed: Reference image
            moving: Image to be registered
            registration_type: Type of registration ('rigid', 'affine', 'both', or None)
        Returns:
            List of transformation file paths
        """
        if not registration_type:
            return []

        transforms = []

        if registration_type in ['rigid', 'both']:
            rigid_tx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Rigid',
                grad_step=0.1,
                reg_iterations=(1000, 500, 250, 50),
                convergence_threshold=1e-07,
                convergence_window_size=10,
                shrink_factors=(4, 3, 2, 1),
                smoothing_sigmas=(6, 4, 1, 0)
            )
            transforms.append(rigid_tx['fwdtransforms'][0])

        if registration_type in ['affine', 'both']:
            affine_tx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Affine',
                grad_step=0.1,
                reg_iterations=(1000, 1000, 1000, 1000),
                convergence_threshold=1e-07,
                convergence_window_size=10,
                shrink_factors=(4, 3, 2, 1),
                smoothing_sigmas=(6, 4, 1, 0)
            )
            transforms.append(affine_tx['fwdtransforms'][0])

        if registration_type == 'nonlinear':
            nonlinear_tx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='SyN',
                grad_step=0.01,
                reg_iterations=(10, 10, 5, 5),
                convergence_threshold=1e-03,
                convergence_window_size=3,
                shrink_factors=(1, 1, 1, 1),
                smoothing_sigmas=(0, 0, 0, 0)
            )
            transforms.append(nonlinear_tx['fwdtransforms'][0])

        return transforms

    @staticmethod
    def apply_transforms(fixed: ANTsImage, moving: ANTsImage,
                         transforms: List[str]) -> ANTsImage:
        """
        Apply pre-computed transformations to an image.

        Args:
            fixed: Reference image
            moving: Image to be transform
            transforms: List of transformation file paths
        Returns:
            Transformed image
        """
        if transforms:
            return ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=transforms
            )
        return moving

    def process_sequence(self, fixed_img: ANTsImage, moving_img: ANTsImage,
                         output_path: str, transforms: Optional[List[str]] = None) -> None:
        """
        Process a single MRI sequence.

        Args:
            fixed_img: Reference image
            moving_img: Image to be processed
            output_path: Path for output file
            transforms: Optional list of pre-computed transformations
        """
        # Normalize images
        norm_moving = self.normalize_image(moving_img)

        # Resample to match fixed image
        interp_moving = ants.resample_image_to_target(
            norm_moving, fixed_img,
            interp_type='bSpline',
            use_voxels=True
        )

        # Apply transforms if provided
        final_img = self.apply_transforms(fixed_img, interp_moving, transforms or [])

        ants.image_write(final_img, output_path)

    def process_all(self, registration_type: Optional[str] = None) -> None:
        """
        Process all sequences for all patients and copy reference CE MRI files.
        Uses a single transformation for all DWI-related sequences.

        Args:
            registration_type: Type of registration to apply (None, 'rigid', 'affine', or 'both')
        """
        input_dir = './tmp'
        data_types = ['b1200'] # ['b800', 'b1200']

        for data_type in data_types:
            for patient in tqdm(os.listdir(input_dir), desc=f"Processing {data_type}"):
                # Set up input and output directories
                input_patient_dir = os.path.join(input_dir, patient)

                # Create input directory structure
                input_patient_type_dir = (os.path.join(self.output_dir, 'input', data_type, patient)
                                          if self.output_dir else f'./input/{data_type}/{patient}')
                os.makedirs(input_patient_type_dir, exist_ok=True)

                if os.path.exists(input_patient_type_dir) and any(f.endswith('.nii.gz') for f in os.listdir(input_patient_type_dir)):
                    # print(f"Skipping {patient} for {data_type} - Coregistered files already exist")
                    continue

                # Create output directory structure
                output_patient_type_dir = (os.path.join(self.output_dir, 'output', data_type, patient)
                                           if self.output_dir else f'./output/{data_type}/{patient}')
                os.makedirs(output_patient_type_dir, exist_ok=True)

                # Copy reference CE MRI if exists
                reference_files = [f for f in os.listdir(input_patient_dir)
                                   if f.startswith('reference_CE') and f.endswith('MRI.nii.gz')]

                if reference_files:
                    reference_file = reference_files[0]
                    src_path = os.path.join(input_patient_dir, reference_file)
                    dst_path = os.path.join(output_patient_type_dir, reference_file)
                    shutil.copy2(src_path, dst_path)

                # Process fixed (reference) image
                try:
                    fixed_img = ants.image_read(os.path.join(input_patient_dir, 'FS T1WI.nii.gz'))
                except Exception:
                    fixed_img = ants.image_read(os.path.join(input_patient_dir, 'FS_T1WI.nii.gz'))

                norm_fixed = self.normalize_image(fixed_img)
                ants.image_write(
                    norm_fixed,
                    os.path.join(input_patient_type_dir, f'{patient}_T1FS_wm.nii.gz')
                )

                # Generate transforms using first DWI sequence found
                dwi_transforms = None
                first_dwi = None
                for sequence in os.listdir(input_patient_dir):
                    if sequence in self.DWI_SEQUENCES:
                        first_dwi = ants.image_read(os.path.join(input_patient_dir, sequence))
                        dwi_transforms = self.get_transforms(norm_fixed, first_dwi, registration_type)
                        break

                # Process sequences
                for sequence in os.listdir(input_patient_dir):
                    if (sequence.startswith('FS T1WI') or
                            sequence.startswith('FS_T1WI') or
                            'reference_CE' in sequence or
                            (data_type == 'b800' and 'DWI_b1200' in sequence) or
                            (data_type == 'b1200' and 'DWI_b800' in sequence)):
                        continue

                    moving_img = ants.image_read(os.path.join(input_patient_dir, sequence))
                    output_name = self.SEQUENCE_MAPPING.get(sequence, sequence)
                    output_path = os.path.join(input_patient_type_dir, f'{patient}_{output_name}')

                    # Use DWI transforms for DWI-related sequences, otherwise compute new transforms
                    if sequence in self.DWI_SEQUENCES and dwi_transforms is not None:
                        self.process_sequence(norm_fixed, moving_img, output_path, dwi_transforms)
                    else:
                        transforms = self.get_transforms(norm_fixed, moving_img, registration_type)
                        self.process_sequence(norm_fixed, moving_img, output_path, transforms)

        # Clean up temporary directory
        shutil.rmtree(input_dir)
        print(f"Processed all sequences. Removed temporary directory: {input_dir}")


if __name__ == '__main__':
    # Example usage
    input_dir = r'D:\DATASET\MRI_CE\Test_dataset'
    output_dir = r'C:\Users\BMC\Desktop\CE_Dataset'

    # Convert DICOM to NIfTI
    converter = DCM2NIIConverter(input_dir)
    converter.convert_all()

    # Perform coregistration without registration (interpolation only)
    coregistration = Coregistration(output_dir)
    # coregistration.process_all()  # Default: no registration

    # Or with specific registration type if needed:
    # coregistration.process_all(registration_type='both')  # For both rigid and affine
    # coregistration.process_all(registration_type='rigid')  # For rigid only
    # coregistration.process_all(registration_type='affine')  # For affine only
    coregistration.process_all(registration_type='nonlinear')  # For nonlinear only
