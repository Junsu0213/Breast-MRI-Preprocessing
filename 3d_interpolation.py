# -*- coding:utf-8 -*-
import os
import pydicom
import numpy as np
from skimage.transform import resize
from pydicom.uid import generate_uid


methods_name = ['READC_map', 'REDWI_b0', 'REDWI_b1200', 'REFS_T1WI', 'RENonFS_T1WI', 'RET2WI']

dwi_b_value = '1200 - Copy - Copy'
dwi_name = f'DWI{dwi_b_value}'
data_dir = f'/mnt/nasw337n2/junsu_work/DATASET/MRI/{dwi_name}/data_dicom'

b_value = '1200'
out_dir = f'/mnt/nasw337n2/junsu_work/DATASET/MRI/final/DWI{b_value}/data_dicom'

files_name = os.listdir(data_dir)
SeriesNumber = 0
for file_name in files_name:
    for method in methods_name:
        uid_seed = generate_uid()
        SeriesNumber += 1
        if method == 'RET2WI':
            out_method_dir = os.path.join(out_dir, f'{file_name}/{method}')
            os.makedirs(out_method_dir, exist_ok=True)
            for t2_dcm, i in zip(t2_dcm_list, range(100)):
                number_str = str(i).zfill(5)
                fname = f'{file_name}_{method}{number_str}.dcm'
                # print(fname)
                save_path = os.path.join(out_method_dir, fname)
                print(save_path)

                ds = pydicom.dcmread(os.path.join(t2_dir, t2_dcm))
                ds.save_as(save_path)
            pass
        else:
            method_dir = os.path.join(data_dir, f'{file_name}/{method}')
            dcm_names = os.listdir(method_dir)

            for dcm_name in dcm_names:
                dcm_path = os.path.join(method_dir, dcm_name)
                ds = pydicom.dcmread(dcm_path)

                mri_array = ds.pixel_array[:, :, np.newaxis]
                # print(mri_array.shape)
                if dcm_name == dcm_names[0]:
                    mri_3d_array = mri_array
                else:
                    mri_3d_array = np.concatenate((mri_3d_array, mri_array), axis=2)

            if method == 'READC_map' or method == 'REDWI_b0' or method == 'REDWI_b800' or method == 'REDWI_b1200':
                resized_img = resize(mri_3d_array, (220, 384, 100), order=3, anti_aliasing=True)
                # 원래 데이터 유형이 int16이었다면 다시 변환
                resized_img = (resized_img * np.iinfo(mri_3d_array.dtype).max).astype(mri_3d_array.dtype)
            elif method == 'REFS_T1WI' or method == 'RENonFS_T1WI':
                resized_img = resize(mri_3d_array, (220, 384, 100), order=3, anti_aliasing=True)
                # 원래 데이터 유형이 int16이었다면 다시 변환
                resized_img = (resized_img * np.iinfo(mri_3d_array.dtype).max).astype(mri_3d_array.dtype)
            print(f'reshape({method}): {resized_img.shape}')
            exit()
            _, _, z_shape = resized_img.shape

            out_method_dir = os.path.join(out_dir, f'{file_name}/{method}')
            os.makedirs(out_method_dir, exist_ok=True)

            t2_dir = os.path.join(data_dir, f'{file_name}/RET2WI')
            t2_dcm_list = os.listdir(t2_dir)

            for t2_dcm, i in zip(t2_dcm_list, range(z_shape)):
                number_str = str(i).zfill(5)
                fname = f'{file_name}_{method}{number_str}.dcm'
                # print(fname)
                save_path = os.path.join(out_method_dir, fname)
                print(save_path)

                ds = pydicom.dcmread(os.path.join(t2_dir, t2_dcm))
                ds.SeriesNumber = SeriesNumber
                ds.SeriesInstanceUID = uid_seed
                ds.PixelData = resized_img[:, :, i].tobytes()
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.save_as(save_path)
