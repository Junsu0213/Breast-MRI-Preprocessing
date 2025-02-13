# -*- coding:utf-8 -*-
"""
Created on Wed. FEB. 12 17:07:11 2025
@author: JUN-SU Park

[3D Medical Image Similarity Metrics]

This script provides various metrics for measuring similarity and error between 3D medical images 
(e.g., CE-MRI, Simulated MRI). Each metric is specifically implemented to consider the characteristics 
of 3D volume images.

Implemented Similarity Metrics:
1. CC (Cross-Correlation): Measures linear correlation between two images
2. MI (Mutual Information): Measures statistical dependency between two images
3. SSIM (Structural Similarity Index): Measures structural similarity

Implemented Error Metrics:
1. NRMSE (Normalized Root Mean Square Error): Normalized version of root mean square error
2. SMAPE (Symmetric Mean Absolute Percentage Error): Symmetric percentage error measurement
3. LOGAC (Logarithmic Accuracy): Accuracy measurement in logarithmic scale
4. MEDSYMAC (Median Symmetric Accuracy): Median symmetric accuracy measurement

Example Usage:
    # Run tests
    python evaluation_test.py
    
    # Use individual metrics
    evaluator = ImageEvaluation(image1_path, image2_path)
    ssim_value = evaluator.calculate_ssim()
    nrmse_value = evaluator.calculate_nrmse()
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

class ImageEvaluation:
    def __init__(self, ce_img_path, simulated_ce_img_path, normalization_method=None):
        """
        Initialize the Image Evaluation class.
        
        Parameters:
        -----------
        ce_img_path : str
            Path to CE-MRI image
        simulated_ce_img_path : str
            Path to simulated CE-MRI image
        normalization_method : str, optional
            Normalization method ('min_max', 'z_score', 'unit_norm', 'histogram_matching', None)
        """
        self.ce_img = ce_img_path
        self.simulated_ce_img = simulated_ce_img_path
        self.img1_itk = sitk.ReadImage(ce_img_path)
        self.img2_itk = sitk.ReadImage(simulated_ce_img_path)
        
        # Apply normalization if specified
        if normalization_method:
            if normalization_method == 'histogram_matching':
                self._normalize_itk_image(normalization_method)
            else:
                self._normalize_both_formats(normalization_method)
        
        # Convert ITK images to numpy arrays
        self.img1_np = sitk.GetArrayFromImage(self.img1_itk)
        self.img2_np = sitk.GetArrayFromImage(self.img2_itk)
        self.data_range = self.img1_np.max() - self.img1_np.min()
        
        # Set epsilon for numerical stability
        self.eps = np.finfo(float).eps

    def _normalize_both_formats(self, method):
        """
        Normalize both ITK and numpy array formats
        
        Parameters:
        -----------
        method : str
            Normalization method ('min_max', 'z_score', 'unit_norm')
        """
        # First normalize numpy arrays
        self.img1_np = sitk.GetArrayFromImage(self.img1_itk)
        self.img2_np = sitk.GetArrayFromImage(self.img2_itk)
        self.img1_np = self._normalize_numpy_image(self.img1_np, method)
        self.img2_np = self._normalize_numpy_image(self.img2_np, method)
        
        # Convert normalized numpy arrays back to ITK images
        self.img1_itk = sitk.GetImageFromArray(self.img1_np)
        self.img2_itk = sitk.GetImageFromArray(self.img2_np)
        
        # Copy original image metadata to maintain geometric information
        self.img1_itk.CopyInformation(sitk.ReadImage(self.ce_img))
        self.img2_itk.CopyInformation(sitk.ReadImage(self.simulated_ce_img))

    def _normalize_itk_image(self, method):
        """
        Normalize ITK images using SimpleITK methods
        
        Parameters:
        -----------
        method : str
            Normalization method for ITK images ('histogram_matching')
        """
        if method == 'histogram_matching':
            # Convert images to float32
            castFilter = sitk.CastImageFilter()
            castFilter.SetOutputPixelType(sitk.sitkFloat32)
            self.img1_itk = castFilter.Execute(self.img1_itk)
            self.img2_itk = castFilter.Execute(self.img2_itk)
            
            # Perform histogram matching
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(128)
            matcher.SetNumberOfMatchPoints(7)
            matcher.SetThresholdAtMeanIntensity(True)
            self.img2_itk = matcher.Execute(self.img2_itk, self.img1_itk)
            
            # Update numpy arrays after ITK normalization
            self.img1_np = sitk.GetArrayFromImage(self.img1_itk)
            self.img2_np = sitk.GetArrayFromImage(self.img2_itk)
    
    def _normalize_numpy_image(self, image, method):
        """
        Normalize numpy array images
        
        Parameters:
        -----------
        image : ndarray
            Image to normalize
        method : str
            Normalization method ('min_max', 'z_score', 'unit_norm')
            
        Returns:
        --------
        ndarray
            Normalized image
        """
        if method == 'min_max':
            return (image - image.min()) / (image.max() - image.min() + self.eps)
        elif method == 'z_score':
            return (image - image.mean()) / (image.std() + self.eps)
        elif method == 'unit_norm':
            return image / (np.sqrt(np.sum(image**2)) + self.eps)
        else:
            raise ValueError("Unsupported normalization method. Choose from 'min_max', 'z_score', 'unit_norm', 'histogram_matching'")

    def calculate_cc(self):
        """
        Calculate Cross Correlation using Pearson correlation coefficient
        
        Returns:
        --------
        float
            Cross correlation value between -1 and 1
        """
        return pearsonr(self.img1_np.flatten(), self.img2_np.flatten())[0]
    
    def calculate_mi(self):
        """
        Calculate Mutual Information using log base 2
        
        Returns:
        --------
        float
            Mutual information value
        """
        hist_2d, _, _ = np.histogram2d(
            self.img1_np.flatten(), 
            self.img2_np.flatten(), 
            bins=50
        )
        
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    
    def calculate_ssim(self):
        """
        Calculate 3D Structural Similarity Index
        
        Returns:
        --------
        float
            SSIM value between -1 and 1
        """
        win_size = 7
        ssim_values = []
        
        # Calculate SSIM for each direction
        for axis in range(3):
            # Transpose array to calculate SSIM along different axes
            img1_transpose = np.moveaxis(self.img1_np, axis, 0)
            img2_transpose = np.moveaxis(self.img2_np, axis, 0)
            
            # Calculate SSIM for each slice in current direction
            ssim_direction = [
                structural_similarity(
                    img1_transpose[i], 
                    img2_transpose[i],
                    win_size=win_size,
                    data_range=self.data_range
                )
                for i in range(img1_transpose.shape[0])
            ]
            ssim_values.append(np.mean(ssim_direction))
        
        return np.mean(ssim_values)

    def calculate_nrmse(self):
        """
        Calculate Normalized Root Mean Square Error
        
        Returns:
        --------
        float
            NRMSE value, smaller is better
        """
        mse = mean_squared_error(self.img1_np.flatten(), self.img2_np.flatten())
        return np.sqrt(mse) / (self.data_range + self.eps)

    def calculate_smape(self):
        """
        Calculate Symmetric Mean Absolute Percentage Error
        
        Returns:
        --------
        float
            SMAPE value as percentage
        """
        return 100.0 * np.mean(
            2.0 * np.abs(self.img1_np - self.img2_np) / 
            (np.abs(self.img1_np) + np.abs(self.img2_np) + self.eps)
        )

    def calculate_logac(self):
        """
        Calculate Logarithmic Accuracy using natural log
        
        Returns:
        --------
        float
            LOGAC value
        """
        # Ensure positive values by adding minimum value and epsilon
        img1_positive = self.img1_np - self.img1_np.min() + self.eps
        img2_positive = self.img2_np - self.img2_np.min() + self.eps
        
        return np.mean(np.abs(np.log(img1_positive/img2_positive)))
    
    def calculate_medsymac(self):
        """
        Calculate Median Symmetric Accuracy
        
        Returns:
        --------
        float
            MEDSYMAC value as percentage
        """
        img1_positive = self.img1_np - self.img1_np.min() + self.eps
        img2_positive = self.img2_np - self.img2_np.min() + self.eps
        
        log_ratio = np.log(img1_positive/img2_positive)
        return 100 * (np.exp(np.median(np.abs(log_ratio))) - 1)

    def calculate_all_metrics(self):
        """
        Calculate all similarity and error metrics
        
        Returns:
        --------
        tuple
            All metric values (CC, MI, SSIM, NRMSE, SMAPE, LOGAC, MEDSYMAC)
        """
        return (
            self.calculate_cc(),
            self.calculate_mi(),
            self.calculate_ssim(),
            self.calculate_nrmse(),
            self.calculate_smape(),
            self.calculate_logac(),
            self.calculate_medsymac()
        )


def main():
    """
    Main function to run the evaluation on a dataset
    """
    b_value = 800
    ds_path = rf'C:\Users\user\Desktop\DATSET\MRI\CE_MRI\output\b{b_value}'
    ds_dirs = [os.path.join(ds_path, f) for f in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, f))]

    for ds_dir in ds_dirs:
        ce_img_path = None
        simulated_ce_img_path = None
        
        for data_ in os.listdir(ds_dir):
            if data_.startswith('reference_'):
                ce_img_path = os.path.join(ds_dir, data_)
                simulated_ce_img_path = os.path.join(ds_dir, data_)
        
        if ce_img_path and simulated_ce_img_path:
            evaluator = ImageEvaluation(ce_img_path, simulated_ce_img_path, normalization_method='unit_norm')
            metrics = evaluator.calculate_all_metrics()
            
            print(f"""
            Metrics Results for {os.path.basename(ds_dir)}:
            - CC: {metrics[0]:.4f}
            - MI: {metrics[1]:.4f}
            - SSIM: {metrics[2]:.4f}
            - NRMSE: {metrics[3]:.4f}
            - SMAPE: {metrics[4]:.4f}
            - LOGAC: {metrics[5]:.4f}
            - MEDSYMAC: {metrics[6]:.4f}
            """)
            break
        else:
            print(f"Could not find both reference and simulated images in {ds_dir}")


if __name__ == "__main__":
    main()