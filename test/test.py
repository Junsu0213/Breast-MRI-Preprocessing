import os

import ants

input_dir = './tmp'

data_list = os.listdir(input_dir)

for patient in data_list:
    patient_dir = os.path.join(input_dir, patient)
    mri_sequences = os.listdir(patient_dir)

    t2 = ants.image_read(f'{patient_dir}/T2WI.nii.gz')


    for mri_sequence in mri_sequences:
        if mri_sequence not in 'T2WI.nii.gz':
            mri_path = os.path.join(patient_dir, mri_sequence)
            mov_img = ants.image_read(mri_path)
            print(mri_path)

