# -*- coding:utf-8 -*-
import os
import shutil
import pydicom
from pydicom.uid import generate_uid

dwi_b_value = '800'
dwi_name = f'DWI{dwi_b_value}'
data_dir = f'/mnt/nasw337n2/junsu_work/DATASET/MRI/{dwi_name}/data_dicom'

files_name = os.listdir(data_dir)

for file_name in files_name:
    uid_seed = generate_uid()
    sub_dir = os.path.join(data_dir, f'{file_name}/DWI')
    adc_dir = os.path.join(data_dir, f'{file_name}/ADC_map')
    dwi_0_dir = os.path.join(data_dir, f'{file_name}/DWI_b0')
    dwi_b_dir = os.path.join(data_dir, f'{file_name}/DWI_b{dwi_b_value}')
    dwi_dcm_names = os.listdir(sub_dir)

    SeriesInstanceUID_list = []
    count = 0
    for dwi_dcm_name in dwi_dcm_names:
        print(dwi_dcm_name)
        if count <= 49:
            dwi_dir = os.path.join(sub_dir, dwi_dcm_name)
            ds = pydicom.read_file(dwi_dir)
            ds.SeriesNumber = 0
            print(ds.SeriesInstanceUID)
            SeriesInstanceUID_list.append(ds.SeriesInstanceUID)
            save_path = os.path.join(dwi_0_dir, dwi_dcm_name)
            ds.save_as(save_path)

        elif 49 < count <= 99 and dwi_b_value == '800':
            dwi_dir = os.path.join(sub_dir, dwi_dcm_name)
            ds = pydicom.read_file(dwi_dir)
            ds.SeriesNumber = 8
            ds.SeriesInstanceUID = uid_seed
            save_path = os.path.join(dwi_b_dir, dwi_dcm_name)
            ds.save_as(save_path)

        elif 99 < count <= 149 and dwi_b_value == '1200':
            dwi_dir = os.path.join(sub_dir, dwi_dcm_name)
            ds = pydicom.read_file(dwi_dir)
            ds.SeriesNumber = 12
            ds.SeriesInstanceUID = uid_seed
            save_path = os.path.join(dwi_b_dir, dwi_dcm_name)
            ds.save_as(save_path)
        # print(SeriesInstanceUID_list)
        count += 1
