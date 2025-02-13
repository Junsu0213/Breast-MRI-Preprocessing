# -*- coding:utf-8 -*-
import os
import ants
from typing import Optional, List, Dict
from ants.core.ants_image import ANTsImage


def register_images(fixed: ANTsImage, moving: ANTsImage,
                    registration_type: Optional[str] = None) -> ANTsImage:
    """
    Perform image registration based on specified type.

    Args:
        fixed: Reference image
        moving: Image to be registered
        registration_type: Type of registration ('rigid', 'affine', 'both', or None)
                         If None, returns the moving image without registration
    Returns:
        Registered image (or original image if no registration requested)
    """
    if not registration_type:
        return moving

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
            reg_iterations=(1000, 500, 250, 50),
            convergence_threshold=1e-07,
            convergence_window_size=10,
            shrink_factors=(4, 3, 2, 1),
            smoothing_sigmas=(6, 4, 1, 0)
        )
        transforms.append(affine_tx['fwdtransforms'][0])

    if registration_type in 'nonlinear':
        nonlinear_tx = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='SyN',
            grad_step=0.05,
            reg_iterations=(10, 10, 5, 5),
            convergence_threshold=1e-05,
            convergence_window_size=3,
            shrink_factors=(1, 1, 1, 1),
            smoothing_sigmas=(0, 0, 0, 0)
        )
        transforms.append(nonlinear_tx['fwdtransforms'][0])

    if transforms:
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transforms
        )
    return moving


def process_sequence(fixed_img: ANTsImage, moving_img: ANTsImage,
                     output_path: str, registration_type: Optional[str] = None) -> None:
    """
    Process a single MRI sequence.

    Args:
        fixed_img: Reference image
        moving_img: Image to be processed
        output_path: Path for output file
        registration_type: Type of registration (None, 'rigid', 'affine', or 'both')
    """

    interp_fixed = ants.resample_image_to_target(
        fixed_img,
        moving_img,
        use_voxels=True
    )

    # Perform registration if requested
    reg_moving_img = register_images(interp_fixed, moving_img, registration_type)

    # Resample to match fixed image
    final_img = ants.resample_image_to_target(
        reg_moving_img,
        fixed_img,
        interp_type='bSpline',
        use_voxels=True
    )

    ants.image_write(final_img, output_path)


if __name__ == '__main__':
    input_dir = r'C:\Users\BMC\Desktop\Coregistration_test\input\11250083'
    output_dir = r'C:\Users\BMC\Desktop\Coregistration_test\output\11250083\affine_2_dwi.nii.gz'

    dwi_dir = fr'{input_dir}/DWI_b0.nii.gz'
    t1_dir = fr'{input_dir}/FS T1WI.nii.gz'

    fixed_img = ants.image_read(t1_dir)
    moving_img = ants.image_read(dwi_dir)

    process_sequence(fixed_img, moving_img, output_dir, registration_type='affine')
