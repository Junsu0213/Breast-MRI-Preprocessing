import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error
from skimage.metrics import structural_similarity, normalized_mutual_information, normalized_root_mse
import matplotlib.pyplot as plt
import torch
from torchmetrics.clustering import MutualInfoScore




def calculate_mi(img1, img2):
    """
    Calculate Mutual Information using log base 2
    
    Returns:
    --------
    float
        Mutual information value
    """
    # Flatten the 3D images to 1D
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # Calculate 2D histogram
    hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=50)
    
    # Convert to joint probability
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # Only non-zero pxy values
    
    # Calculate mutual information
    mi_value = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    return mi_value

def calculate_smape(img1, img2):
    diff = np.abs(img1 - img2)
    denomizaotr = (np.abs(img1) + np.abs(img2))/2 + 1e-6
    return 100 * np.mean(diff / denomizaotr)

if __name__ == "__main__":
    ds_dir = r'C:\Users\user\Desktop\DATSET\TEST\similarity_matrix_MRI'
    img1_itk = sitk.ReadImage(fr"{ds_dir}\ce.nii.gz")
    img2_itk = sitk.ReadImage(fr"{ds_dir}\ce.nii.gz")
    # img2_itk = sitk.ReadImage(fr"{ds_dir}\simulated.nii.gz")
    img1 = sitk.GetArrayFromImage(img1_itk)
    img2 = sitk.GetArrayFromImage(img2_itk)
    
    # Flatten images and convert to integer type for mutual_info_score
    img1_flat = img1.flatten().astype(int)
    img2_flat = img2.flatten().astype(int)
    print(img1_flat.shape, img2_flat.shape)
    
    # Calculate MI using custom method
    mi_custom = calculate_mi(img1, img2)
    print(f"Custom MI: {mi_custom}")
    
    # Calculate MI using mutual_info_score
    mi_sklearn = normalized_mutual_information(img1, img2)
    print(f"Skimage MI: {mi_sklearn - 1}")

    # # Calculate MI using sklearn
    # mi_sklearn = normalized_mutual_info_score(img1_flat, img2_flat)
    # print(f"Sklearn MI: {mi_sklearn}")

    # It was proposed to be useful in registering images by Colin Studholme and colleagues [1]. It ranges from 1 (perfectly uncorrelated image values) to 2 (perfectly correlated image values, whether positively or negatively).

    # # Calculate MI using torchmetrics
    # target = torch.tensor(img1_flat)
    # pred = torch.tensor(img2_flat)
    # mi_score = MutualInfoScore()
    # print('torchmetrics MI: ', mi_score(target, pred))

    # Calculate SSIM using skimage
    ssim_skimage = structural_similarity(img1, img2, win_size=7, data_range=img1.max() - img1.min())
    print(f"Skimage SSIM: {ssim_skimage}")

    
    # Calculate NRMSE using skiamge
    nrmse_skimage = normalized_root_mse(img1, img2)
    print(f"Skimage NRMSE: {nrmse_skimage}")

    # Calculate SMAPE using skimage
    smape_sklearn = mean_absolute_percentage_error(img1_flat, img2_flat)
    print(f"Sklearn SMAPE: {smape_sklearn}")

    print(f"Custom SMAPE: {calculate_smape(img1, img2)}")
