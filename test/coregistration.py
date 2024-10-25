# -*- coding:utf-8 -*-
"""
Created on Thu. Oct. 15 16:56:11 2024
@author: JUN-SU PARK
"""
import os
import numpy as np
import ants
import shutil
from tqdm import tqdm


class Coregistration:
    def __init__(self, output_dir=None, registration=False):
        self.output_dir = output_dir
        self.registration = registration

    def apply_coregistration(self, nom_fix_img, mov_img, output_dir_, patient_name, mri_sequence):

        nom_mov_img = self.normalize_image(mov_img)

        save_sequence = self.save_file_name(mri_sequence)

        output_path = os.path.join(output_dir_, f'{patient_name}_{save_sequence}')
        interp_mov_img = ants.resample_image_to_target(nom_mov_img, nom_fix_img, interp_type='bSpline', use_voxels=True)

        if self.registration is True:
            # Rigid
            rigid_tx = ants.registration(
                fixed=nom_fix_img,
                moving=interp_mov_img,
                type_of_transform='Rigid',
                grad_step=0.1,
                reg_iterations=(1000, 500, 250, 50),
                convergence_threshold=1e-07,
                convergence_window_size=10,
                shrink_factors=(4, 3, 2, 1),
                smoothing_sigmas=(6, 4, 1, 0),
            )

            # Affine
            affine_tx = ants.registration(
                fixed=nom_fix_img,
                moving=interp_mov_img,
                type_of_transform='Affine',
                grad_step=0.1,
                reg_iterations=(1000, 1000, 1000, 1000),
                convergence_threshold=1e-07,
                convergence_window_size=10,
                shrink_factors=(4, 3, 2, 1),
                smoothing_sigmas=(6, 4, 1, 0)
            )

            reg_img = ants.apply_transforms(
                fixed=nom_fix_img,
                moving=interp_mov_img,
                transformlist=[rigid_tx['fwdtransforms'][0], affine_tx['fwdtransforms'][0]]
            )

            ants.image_write(reg_img, output_path)

        else:
            ants.image_write(interp_mov_img, output_path)

    def main(self):
        input_dir = './tmp'
        data_list = os.listdir(input_dir)
        data_types = ['b800', 'b1200']
        for data_type in data_types:
            for patient in tqdm(data_list, desc="Coregistration", unit="patient"):
                if self.output_dir is None:
                    output_dir_ = f'./input/{data_type}/{patient}'
                else:
                    output_dir_ = f'{self.output_dir}/input/{data_type}/{patient}'
                os.makedirs(output_dir_, exist_ok=True)

                patient_dir = os.path.join(input_dir, patient)
                try:
                    fix_img = ants.image_read(f'{patient_dir}/FS T1WI.nii.gz')
                except Exception:
                    fix_img = ants.image_read(f'{patient_dir}/FS_T1WI.nii.gz')
                nom_fix_img = self.normalize_image(fix_img)

                output_path = os.path.join(output_dir_, f'{patient}_T1FS_wm.nii.gz')
                ants.image_write(nom_fix_img, output_path)

                mri_sequences = os.listdir(patient_dir)

                for mri_sequence in mri_sequences:
                    if mri_sequence == 'FS T1WI.nii.gz':
                        pass
                    elif mri_sequence == 'FS_T1WI.nii.gz':
                        pass
                    elif mri_sequence == 'reference_CE MRI.nii.gz':
                        pass
                    elif mri_sequence == 'reference_CE_MRI.nii.gz':
                        pass
                    else:
                        if data_type == 'b800' and mri_sequence == 'DWI_b1200.nii.gz':
                            pass
                        elif data_type == 'b1200' and mri_sequence == 'DWI_b800.nii.gz':
                            pass
                        else:
                            mri_path = os.path.join(patient_dir, mri_sequence)
                            mov_img = ants.image_read(mri_path)
                            self.apply_coregistration(nom_fix_img, mov_img, output_dir_, patient, mri_sequence)

        shutil.rmtree(input_dir)
        print(f'{input_dir} directory has been deleted.')
        print("\nCoregistration complete!")

    @staticmethod
    def normalize_image(image):
        img_np = image.numpy()
        img_np = img_np.astype(np.float32)
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
        return ants.from_numpy(
            img_np, origin=image.origin, spacing=image.spacing,
            direction=image.direction
    )
    @staticmethod
    def save_file_name(mri_sequence):
        if mri_sequence == 'ADC map.nii.gz':
            save_sequence = 'ADC_wm.nii.gz'
        elif mri_sequence == 'ADC_map.nii.gz':
            save_sequence = 'ADC_wm.nii.gz'
        elif mri_sequence == 'DWI_b0.nii.gz':
            save_sequence = 'B0_wm.nii.gz'
        elif mri_sequence == 'DWI_b800.nii.gz':
            save_sequence = 'DWI_wm.nii.gz'
        elif mri_sequence == 'DWI_b1200.nii.gz':
            save_sequence = 'DWI_wm.nii.gz'
        elif mri_sequence == 'NonFS T1WI.nii.gz':
            save_sequence = 'T1_wm.nii.gz'
        elif mri_sequence == 'NonFS_T1WI.nii.gz':
            save_sequence = 'T1_wm.nii.gz'
        elif mri_sequence == 'T2WI.nii.gz':
            save_sequence = 'T2FS_wm.nii.gz'
        return save_sequence


if __name__ == '__main__':
    output_dir = '/mnt/nasw337n2/junsu_work/DATASET/Breast-MRI/results'
    Coregistration(output_dir).main()
