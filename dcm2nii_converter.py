# -*- coding:utf-8 -*-
"""
Created on Mon. Oct. 7 14:19:22 2024
@author: JUN-SU PARK
"""
import os
import shutil
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


class DCM2NIIConverter:
    def __init__(self, input_dir, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def convert_dicom_to_nii(self, file_dir, patient_name):
        if self.output_dir is None:
            self.output_dir = f'./tmp'
        output_path = f'{self.output_dir}/{patient_name}'
        os.makedirs(output_path, exist_ok=True)

        file_name = os.path.basename(file_dir)

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, f'{output_path}/{file_name}.nii.gz')

    def main(self):
        data_list = os.listdir(self.input_dir)

        for patient in tqdm(data_list, desc=f"Converting Dicom to Nifti", unit="patient"):
            patient_id_dir = os.path.join(self.input_dir, patient)
            mri_sequences = os.listdir(patient_id_dir)

            for mri_sequence in mri_sequences:
                mri_sequence_dir = os.path.join(patient_id_dir, mri_sequence)

                if 'DWI' in mri_sequence_dir:
                    try:
                        self.split_dwi_files(mri_sequence_dir)
                    except Exception:
                        pass
                    dwi_files = os.listdir(mri_sequence_dir)
                    for dwi_file in dwi_files:
                        dwi_file_dir = os.path.join(mri_sequence_dir, dwi_file)
                        self.convert_dicom_to_nii(dwi_file_dir, patient)
                else:
                    self.convert_dicom_to_nii(mri_sequence_dir, patient)

        print("\nDCM2NII complete!")

    @staticmethod
    def split_dwi_files(file_dir):
        dwi_files = os.listdir(file_dir)
        for dwi_file in dwi_files:
            dwi_file_path = os.path.join(file_dir, dwi_file)

            dcm = pydicom.dcmread(dwi_file_path)
            sequence_name = dcm.SequenceName if hasattr(dcm, 'SequenceName') else 'Unknown'

            file_name = os.path.basename(dwi_file_path)

            if 'b0' in sequence_name:
                target_dir = os.path.join(file_dir, 'DWI_b0')
            elif 'b800' in sequence_name:
                target_dir = os.path.join(file_dir, 'DWI_b800')
            elif 'b1200' in sequence_name:
                target_dir = os.path.join(file_dir, 'DWI_b1200')
            else:
                target_dir = os.path.join(file_dir, 'DWI')

            os.makedirs(target_dir, exist_ok=True)

            target_file_path = os.path.join(target_dir, file_name)
            shutil.move(dwi_file_path, target_file_path)


if __name__ == '__main__':
    input_dir = r'D:\DATASET\Breast_MRI_test'
    DCM2NIIConverter(input_dir).main()
