import os
import time
import ants
import numpy as np
from pathlib import Path

def ants_registration(fixed_path, moving_path, save_dir, method='rigid'):
    """
    ANTs를 사용한 이미지 정합
    
    Args:
        fixed_path: 고정 이미지 경로
        moving_path: 이동 이미지 경로
        save_dir: 저장 경로
        method: 정합 방법 ('rigid', 'affine', 'nonlinear')
    
    Returns:
        float: 처리 시간 (초)
    """
    start_time = time.time()
    
    # 이미지 로드
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)
    
    # 정합 파라미터 설정
    if method == 'rigid':
        transform_type = 'Rigid'
    elif method == 'affine':
        transform_type = 'Affine'
    else:  # nonlinear
        transform_type = 'SyN'
    
    # 정합 수행
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        grad_step=0.1 if method != 'nonlinear' else 0.01,
        reg_iterations=(1000, 500, 250, 50),
        convergence_threshold=1e-07,
        convergence_window_size=10,
        shrink_factors=(4, 3, 2, 1),
        smoothing_sigmas=(6, 4, 1, 0)
    )
    
    # 결과 저장
    warped_image = registration['warpedmovout']
    output_path = os.path.join(save_dir, f'registered_{method}.nii.gz')
    ants.image_write(warped_image, output_path)
    
    elapsed_time = time.time() - start_time
    return elapsed_time

def run_ants_test(raw_dir, save_dir):
    """ANTs 테스트 실행"""
    methods = ['rigid', 'affine', 'nonlinear']
    results = {}
    
    for method in methods:
        time_taken = ants_registration(
            fixed_path=os.path.join(raw_dir, 'fixed.nii.gz'),
            moving_path=os.path.join(raw_dir, 'moving.nii.gz'),
            save_dir=save_dir,
            method=method
        )
        results[method] = time_taken
        print(f'ANTs {method.capitalize()} Registration 완료: {time_taken:.2f}초')
    
    return results

if __name__ == '__main__':
    raw_dir = r'C:\Users\user\Desktop\DATSET\Test_Dataset\CE_MRI\Registration_test\raw'
    save_dir = r'C:\Users\user\Desktop\DATSET\Test_Dataset\CE_MRI\Registration_test\ants'
    run_ants_test(raw_dir, save_dir)