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
    evaluator = TestImageEvaluation()
    ssim_value = evaluator.calculate_3d_ssim(image1, image2)
    nrmse_value = evaluator.calculate_3d_nrmse(image1, image2)
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity, normalized_mutual_information, normalized_root_mse
import matplotlib.pyplot as plt


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
            return (image - image.min()) / (image.max() - image.min())
        elif method == 'z_score':
            return (image - image.mean()) / image.std()
        elif method == 'unit_norm':
            return image / np.sqrt(np.sum(image**2))
        else:
            raise ValueError("Unsupported normalization method. Choose from 'min_max', 'z_score', 'unit_norm', 'histogram_matching'")

    @staticmethod
    def scale12bit(img):
        # constants
        new_mean = 2048.
        new_std = 400.

        return np.clip(((img - np.mean(img)) / (np.std(img) / new_std)) + new_mean, 1e-10, 4095)

    def calculate_cc(self):
        """
        Calculate Cross Correlation using Pearson correlation coefficient
        
        Returns:
        --------
        float
            Cross correlation value between -1 and 1
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range

        return pearsonr(img1_scaled.flatten(), img2_scaled.flatten())[0]
    
    def calculate_mi(self):
        """
        Calculate Mutual Information using log base 2
        
        Returns:
        --------
        float
            Mutual information value between 0 and 1
        
        Note:
        -----
        It was proposed to be useful in registering images by Colin Studholme and colleagues.
        It ranges from 0 (perfectly uncorrelated image values) to 1 (perfectly correlated image values,
        whether positively or negatively).
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range

        return normalized_mutual_information(img1_scaled, img2_scaled)
    
    def calculate_ssim(self):
        """
        Calculate 3D Structural Similarity Index
        
        Returns:
        --------
        float
            SSIM value between -1 and 1
        
        Note:
        -----
        If data_range is not specified, the range is automatically guessed based on the image data type.
        However, for floating-point image data, this estimate yields a result double the value of the desired range,
        as the dtype_range in skimage.util.dtype.py has defined intervals from -1 to +1. This yields an estimate of 2,
        instead of 1, which is most often required when working with image data (as negative light intensities are nonsensical).
        In case of working with YCbCr-like color data, note that these ranges are different per channel (Cb and Cr have double
        the range of Y), so one cannot calculate a channel-averaged SSIM with a single call to this function, as identical ranges
        are assumed for each channel.

        To match the implementation of Wang et al., set gaussian_weights to True, sigma to 1.5, use_sample_covariance to False,
        and specify the data_range argument.
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range

        return structural_similarity(img1_scaled, img2_scaled, win_size=7, data_range=(img1_scaled.max() - img1_scaled.min()))

    def calculate_nrmse(self):
        """
        Calculate Normalized Root Mean Square Error
        
        Returns:
        --------
        float
            NRMSE value, smaller is better
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range

        return normalized_root_mse(img1_scaled, img2_scaled)

    def calculate_smape(self):
        """
        Calculate Symmetric Mean Absolute Percentage Error
        
        Returns:
        --------
        float
            SMAPE value as percentage
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range

        return np.mean(
            np.abs(img1_scaled - img2_scaled) / 
            (np.abs(img1_scaled) + np.abs(img2_scaled) + self.eps)
        )

    def calculate_logac(self):
        """
        Calculate Log accuracy ratio
        
        Returns:
        --------
        float
            LOGAC value
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range
        # # Ensure positive values by adding minimum value and epsilon
        # img1_positive = self.img1_np - self.img1_np.min() + self.eps
        # img2_positive = self.img2_np - self.img2_np.min() + self.eps
        
        return np.mean(np.fabs(np.log(img1_scaled/img2_scaled)))
    
    def calculate_medsymac(self):
        """
        Calculate Median Symmetric Accuracy
        
        Returns:
        --------
        float
            MEDSYMAC value as percentage
        """
        img1_scaled = self.scale12bit(self.img1_np) # scale to 12 bit range
        img2_scaled = self.scale12bit(self.img2_np) # scale to 12 bit range
        # img1_positive = self.img1_np - self.img1_np.min() + self.eps
        # img2_positive = self.img2_np - self.img2_np.min() + self.eps
        
        log_ratio = np.log(img1_scaled/img2_scaled)
        return np.exp(np.median(np.fabs(log_ratio))) - 1

    def calculate_all_metrics(self):
        """Calculate all similarity and error metrics"""
        return (
            self.calculate_cc(),
            self.calculate_mi(),
            self.calculate_ssim(),
            self.calculate_nrmse(),
            self.calculate_smape(),
            self.calculate_logac(),
            self.calculate_medsymac()
        )


if __name__ == "__main__":
    b_value = 1200
    ds_path = rf'C:\Users\user\Desktop\DATSET\MRI\CE_MRI\output\b{b_value}'
    ds_dirs = [os.path.join(ds_path, f) for f in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, f))]

    results = []

    for ds_dir in ds_dirs:
        print('ds_dir: ', ds_dir)
        ce_img_path = None
        simulated_ce_img_path = None
        
        for data_ in os.listdir(ds_dir):
            if data_.startswith('reference_'):
                ce_img_path = os.path.join(ds_dir, data_)
                # simulated_ce_img_path = os.path.join(ds_dir, data_)
            else:
                simulated_ce_img_path = os.path.join(ds_dir, data_)
        
        if ce_img_path and simulated_ce_img_path:
            evaluator = ImageEvaluation(ce_img_path, simulated_ce_img_path, normalization_method='unit_norm')
            metrics = evaluator.calculate_all_metrics()
            print(f"""
            Metrics Results for {os.path.basename(ds_dir)}:
            유사도 지표
            - CC: {metrics[0]:.4f}
            - MI: {metrics[1]:.4f}
            - SSIM: {metrics[2]:.4f}
            오차 지표
            - NRMSE: {metrics[3]:.4f}
            - SMAPE: {metrics[4]:.4f}
            - LOGAC: {metrics[5]:.4f}
            - MEDSYMAC: {metrics[6]:.4f}
            """)
            results.append({
                'Subject': os.path.basename(ds_dir),
                'CC': metrics[0],
                'MI': metrics[1],
                'SSIM': metrics[2],
                'NRMSE': metrics[3],
                'SMAPE': metrics[4],
                'LOGAC': metrics[5],
                'MEDSYMAC': metrics[6]
            })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate mean and standard deviation, excluding 'Subject' column
    metrics = ['CC', 'MI', 'SSIM', 'NRMSE', 'SMAPE', 'LOGAC', 'MEDSYMAC']
    mean_values = df[metrics].mean()
    std_values = df[metrics].std()

    # Plotting
    x_pos = np.arange(len(metrics))

    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_values[metrics], yerr=std_values[metrics], align='center', alpha=0.7, ecolor='black', capsize=10)
    ax.set_ylabel('Metric Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_title('Quantitative Similarity and Error Metrics')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('metrics_bar_plot.png')
    plt.show()
