# -*- coding:utf-8 -*-
import os
import pydicom
import numpy as np
import cv2

methods_name = ['ADC_map', 'DWI_b0', 'DWI_b1200', 'FS_T1WI', 'NonFS_T1WI', 'T2WI']

dwi_b_value = '1200 - Copy - Copy'
dwi_name = f'DWI{dwi_b_value}'
data_dir = f'/mnt/nasw337n2/junsu_work/DATASET/MRI/{dwi_name}/data_dicom'

files_name = os.listdir(data_dir)

for file_name in files_name:
    for method in methods_name:
        method_dir = os.path.join(data_dir, f'{file_name}/{method}')
        dcm_names = os.listdir(method_dir)

        out_dir = os.path.join(data_dir, f'{file_name}/RE{method}')
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        for dcm_name in dcm_names:
            dcm_path = os.path.join(method_dir, dcm_name)
            ds = pydicom.dcmread(dcm_path)

            if method == 'FS_T1WI' or method == 'NonFS_T1WI':
                ds_array = ds.pixel_array[80:-80, 10:-10]
            else:
                ds_array = ds.pixel_array

            ds_image = cv2.resize(ds_array, (384, 236))
            ds_array = np.array(ds_image)

            if method == 'ADC_map' or method == 'DWI_b0' or method == 'DWI_b800' or method == 'DWI_b1200':
                ds_array = ds_array[:-16, :]
            else:
                ds_array = ds_array[16:, :]

            row, col = ds_array.shape

            ds.PixelData = ds_array.tobytes()
            ds.Columns = col
            ds.Rows = row
            # ds.SamplesPerPixel = 0.89
            ds.PhotometricInterpretation = 'MONOCHROME2'

            out_path = os.path.join(out_dir, dcm_name)
            if method == 'T2WI' and 9 <= count < 109:
                ds.save_as(out_path)
            elif method == 'NonFS_T1WI' and 9 <= count < 176:
                ds.save_as(out_path)
            elif method == 'FS_T1WI' and 9 <= count < 176:
                ds.save_as(out_path)
            elif method == 'ADC_map' or method == 'DWI_b0' or method == 'DWI_b800' or method == 'DWI_b1200':
                ds.save_as(out_path)
            count += 1