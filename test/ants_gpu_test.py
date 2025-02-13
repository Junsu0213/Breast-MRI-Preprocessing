# -*- coding:utf-8 -*-
import subprocess
import os

# 데이터 경로
moving_path = r'C:\Users\user\Desktop\JS_PROJECT\DATASET\Registration\DWI_b0.nii.gz'
fixed_path = r'C:\Users\user\Desktop\JS_PROJECT\DATASET\Registration\reference_CE_MRI.nii.gz'
output_path = r'C:\Users\user\Desktop\JS_PROJECT\DATASET\Registration\ants_gpu'

# WSL 경로로 변환 함수
def windows_to_wsl_path(windows_path):
    result = subprocess.run(["wsl", "wslpath", "-u", windows_path], capture_output=True, text=True)
    return result.stdout.strip()

# 경로 변환
moving_path_wsl = windows_to_wsl_path(moving_path)
fixed_path_wsl = windows_to_wsl_path(fixed_path)
output_path_wsl = windows_to_wsl_path(output_path)

# ANTs 실행 파일 경로 (WSL 내부)
ants_executable = "/home/junsu0213/JS_PROJECT/ANTs/build/bin/antsRegistration"

# ANTs 명령 실행
command = [
    "wsl", "-d", "Ubuntu-20.04",  # WSL에서 실행
    ants_executable,
    "--dimensionality", "3",
    "--float", "0",
    "--output", f"{output_path_wsl}/ants_",
    "--interpolation", "Linear",
    "--use-histogram-matching", "1",
    "--winsorize-image-intensities", "[0.005,0.995]",
    "--transform", "Affine[0.1]",
    "--metric", f"MI[{fixed_path_wsl},{moving_path_wsl},1,32,Regular,0.25]",
    "--convergence", "[1000x500x250x100,1e-6,10]",
    "--shrink-factors", "8x4x2x1",
    "--smoothing-sigmas", "3x2x1x0vox"
]

try:
    # 명령 실행
    result = subprocess.run(command, check=True, capture_output=True)
    stdout_decoded = result.stdout.decode('utf-8', errors='ignore')
    stderr_decoded = result.stderr.decode('utf-8', errors='ignore')
    print("STDOUT:", stdout_decoded)
    print("STDERR:", stderr_decoded)
except subprocess.CalledProcessError as e:
    error_decoded = e.stderr.decode('utf-8', errors='ignore')
    print("Error:", error_decoded)
