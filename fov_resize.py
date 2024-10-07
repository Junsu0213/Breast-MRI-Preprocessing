# -*- coding:utf-8 -*-
import os
import pydicom
import numpy as np
import cv2
from PIL import Image


dwi_b_value = '1200 - Copy'
dwi_name = f'DWI{dwi_b_value}'
data_dir = f'/mnt/nasw337n2/junsu_work/DATASET/MRI/{dwi_name}/data_dicom'

files_name = os.listdir(data_dir)
methods_name = ['FS_T1WI', 'NonFS_T1WI']

for file_name in files_name:
    t1_dir = os.path.join(data_dir, f'{file_name}/NonFS_T1WI')
    fs_t1_dir = os.path.join(data_dir, f'{file_name}/FS_T1WI')

    # t1_out_dir = os.path.join(data_dir, f'{file_name}/RENonFS_T1WI')
    # fs_t1_out_dir = os.path.join(data_dir, f'{file_name}/REFS_T1WI')
    # os.makedirs(t1_out_dir, exist_ok=True)
    # os.makedirs(fs_t1_out_dir, exist_ok=True)

    # t2_dir = os.path.join(data_dir, f'{file_name}/T2WI')
    # t2_dcm_names = os.listdir(t2_dir)
    # t2_path = os.path.join(t2_dir, t2_dcm_names[0])
    # t2_ds = pydicom.dcmread(t2_path)
    # print(t2_ds.pixel_array.shape)
    # print(t2_ds.Rows, t2_ds.Columns)
    # exit()

    t1_dcm_names = os.listdir(t1_dir)
    fs_t1_dcm_names = os.listdir(fs_t1_dir)

    for t1, fs_t1 in zip(t1_dcm_names, fs_t1_dcm_names):
        t1_path = os.path.join(t1_dir, t1)
        fs_t1_path = os.path.join(fs_t1_dir, fs_t1)
        # t1_out_path = os.path.join(t1_out_dir, t1)
        # fs_t1_out_path = os.path.join(fs_t1_out_dir, fs_t1)
        t1_ds = pydicom.dcmread(t1_path)
        fs_t1_ds = pydicom.dcmread(fs_t1_path)

        t1_array = t1_ds.pixel_array[80:-80, 10:-10]
        fst1_array = fs_t1_ds.pixel_array[80:-80, 10:-10]

        # t1_image = Image.fromarray(t1_array).resize((156, 256))
        # fst1_image = Image.fromarray(fst1_array).resize((156, 256))


        t1_image = cv2.resize(t1_array, (256, 156))
        fst1_image = cv2.resize(fst1_array, (256, 156))

        t1_array = np.array(t1_image)
        fst1_array = np.array(fst1_image)


        print(t1_array.shape, fst1_array.shape)
        print(f"Rows: {t1_ds.Rows}, Columns: {t1_ds.Columns}")
        print(f"Bits Allocated: {t1_ds.BitsAllocated}")
        print(f"Samples per Pixel: {t1_ds.SamplesPerPixel}")
        t1_ds.PixelData = t1_array.tobytes()
        t1_ds.Rows = 156
        t1_ds.Columns = 256
        t1_ds.SamplesPerPixel = 1
        t1_ds.PhotometricInterpretation = 'MONOCHROME2'
        print(t1_ds.pixel_array.shape)


        fs_t1_ds.PixelData = fst1_array.tobytes()
        fs_t1_ds.Rows = 156
        fs_t1_ds.Columns = 256
        fs_t1_ds.SamplesPerPixel = 1
        fs_t1_ds.PhotometricInterpretation = 'MONOCHROME2'

        # t1_ds.save_as(t1_path)
        # fs_t1_ds.save_as(fs_t1_path)
