import os
import nibabel as nib
import numpy as np
import ants


def normalize_image(image):
    img_np = image.numpy()
    img_np = img_np.astype(np.float32)
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    return ants.from_numpy(
        img_np, origin=image.origin, spacing=image.spacing,
        direction=image.direction
    )

# data load
non_fs_t1 = ants.image_read('./tmp/NonFS T1WI.nii.gz')
fs_t1 = ants.image_read('./tmp/FS T1WI.nii.gz')
fs_t2 = ants.image_read('./tmp/T2WI.nii.gz')
# ref_ce = ants.image_read('./tmp/reference_CE MRI.nii.gz')
dwi_b0 = ants.image_read('./tmp/DWI_b0.nii.gz')
# dwi_b800 = ants.image_read('./tmp/DWI_b800.nii.gz')
# dwi_b1200 = ants.image_read('./tmp/DWI_b1200.nii.gz')
adc = ants.image_read('./tmp/ADC map.nii.gz')

file_name = 'fs_t1'
input_img = fs_t1

fs_t2 = normalize_image(fs_t2)
input_img = normalize_image(input_img)
# fs_t2 = ants.iMath(fs_t2, "Normalize")
# input_img = ants.iMath(input_img, "Normalize")

interp_img = ants.resample_image_to_target(input_img, fs_t2, interp_type='bSpline', use_voxels=True)

# Rigid
rigid_tx = ants.registration(
    fixed=fs_t2,
    moving=interp_img,
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
    fixed=fs_t2,
    moving=interp_img,
    type_of_transform='Affine',
    grad_step=0.1,
    reg_iterations=(1000, 1000, 1000, 1000),
    convergence_threshold=1e-07,
    convergence_window_size=10,
    shrink_factors=(4, 3, 2, 1),
    smoothing_sigmas=(6, 4, 1, 0)
)


# rigid = ants.apply_transforms(
#     fixed=fs_t1,
#     moving=interp_img,
#     transformlist=rigid_tx['fwdtransforms']
# )
#
# affine = ants.apply_transforms(
#     fixed=fs_t2,
#     moving=interp_img,
#     transformlist=affine_tx['fwdtransforms']
# )

rigid_affine = ants.apply_transforms(
    fixed=fs_t2,
    moving=interp_img,
    transformlist=[rigid_tx['fwdtransforms'][0], affine_tx['fwdtransforms'][0]]
)


out_path = r'.\tmp2'
os.makedirs(out_path, exist_ok=True)

ants.image_write(interp_img, os.path.join(out_path, f"{file_name}_interp.nii.gz"))
# ants.image_write(rigid, os.path.join(out_path, f"{file_name}_rigid.nii.gz"))
# ants.image_write(affine, os.path.join(out_path, f"{file_name}_affine.nii.gz"))
ants.image_write(rigid_affine, os.path.join(out_path, f"{file_name}_rigid_affine.nii.gz"))
