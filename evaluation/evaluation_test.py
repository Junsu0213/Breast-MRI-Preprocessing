"""
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

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import unittest
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
import numpy as np

class TestImageEvaluation(unittest.TestCase):
    def setUp(self):
        # 테스트용 3D 이미지 생성 (예시 크기: 64x64x64)
        self.original_image = np.random.rand(64, 64, 64)
        # 약간의 노이즈를 추가한 시뮬레이션 이미지
        noise = np.random.normal(0, 0.1, (64, 64, 64))
        self.simulated_image = self.original_image + noise
        
    def calculate_mse(self, img1, img2):
        """평균 제곱 오차(Mean Squared Error) 계산"""
        return np.mean((img1 - img2) ** 2)
    
    def calculate_psnr(self, img1, img2):
        """PSNR (Peak Signal-to-Noise Ratio) 계산"""
        mse = self.calculate_mse(img1, img2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        return 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
    def calculate_3d_ssim(self, img1, img2):
        """3D SSIM (Structural Similarity Index) 계산"""
        ssim_scores = []
        # z축을 따라 각 슬라이스에 대해 SSIM 계산
        for z in range(img1.shape[0]):
            score = ssim(img1[z], img2[z], data_range=img1[z].max() - img1[z].min())
            ssim_scores.append(score)
        return np.mean(ssim_scores)
    
    def calculate_cc(self, img1, img2):
        """상관계수(Correlation Coefficient) 계산"""
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        correlation, _ = pearsonr(img1_flat, img2_flat)
        return correlation

    def calculate_3d_mi(self, img1, img2):
        """3D 상호 정보량(Mutual Information) 계산
        각 축 방향으로 MI를 계산하고 평균"""
        mi_scores = []
        
        # 각 축 방향으로 MI 계산
        for axis in range(3):
            # 해당 축을 따라 슬라이스들의 MI 계산
            for slice_idx in range(img1.shape[axis]):
                if axis == 0:
                    slice1 = img1[slice_idx, :, :]
                    slice2 = img2[slice_idx, :, :]
                elif axis == 1:
                    slice1 = img1[:, slice_idx, :]
                    slice2 = img2[:, slice_idx, :]
                else:
                    slice1 = img1[:, :, slice_idx]
                    slice2 = img2[:, :, slice_idx]
                
                # 히스토그램 계산
                bins = 50
                slice1_flat = slice1.flatten()
                slice2_flat = slice2.flatten()
                
                c_min = min(slice1_flat.min(), slice2_flat.min())
                c_max = max(slice1_flat.max(), slice2_flat.max())
                
                hist1 = np.histogram(slice1_flat, bins=bins, range=(c_min, c_max))[0]
                hist2 = np.histogram(slice2_flat, bins=bins, range=(c_min, c_max))[0]
                
                # 정규화
                hist1 = hist1 / float(np.sum(hist1))
                hist2 = hist2 / float(np.sum(hist2))
                
                mi_scores.append(mutual_info_score(hist1, hist2))
        
        return np.mean(mi_scores)

    def calculate_3d_nrmse(self, img1, img2):
        """3D NRMSE 계산"""
        mse = self.calculate_mse(img1, img2)
        rmse = np.sqrt(mse)
        # 3D 볼륨 전체의 범위로 정규화
        return rmse / (np.max(img1) - np.min(img1))

    def calculate_3d_smape(self, img1, img2):
        """3D SMAPE 계산"""
        eps = np.finfo(float).eps
        return 100 * np.mean(2 * np.abs(img1 - img2) / (np.abs(img1) + np.abs(img2) + eps))

    def calculate_3d_logac(self, img1, img2):
        """3D LOGAC 계산"""
        # 음수나 0을 피하기 위해 작은 값 추가
        eps = np.finfo(float).eps
        return np.mean(np.abs(np.log10(img1 + eps) - np.log10(img2 + eps)))

    def calculate_3d_medsymac(self, img1, img2):
        """3D MEDSYMAC 계산"""
        eps = np.finfo(float).eps
        ratio = (img1 + eps) / (img2 + eps)
        return np.exp(np.median(np.abs(np.log(ratio))))

    def test_image_similarity(self):
        """이미지 유사도 메트릭 테스트"""
        # 유사도 지표 계산
        cc_value = self.calculate_cc(self.original_image, self.simulated_image)
        mi_value = self.calculate_3d_mi(self.original_image, self.simulated_image)
        ssim_value = self.calculate_3d_ssim(self.original_image, self.simulated_image)
        
        print("\n=== 3D 유사도 지표 ===")
        print(f"3D CC: {cc_value:.4f}")
        print(f"3D MI: {mi_value:.4f}")
        print(f"3D SSIM: {ssim_value:.4f}")
        
        # 오차 지표 계산
        nrmse_value = self.calculate_3d_nrmse(self.original_image, self.simulated_image)
        smape_value = self.calculate_3d_smape(self.original_image, self.simulated_image)
        logac_value = self.calculate_3d_logac(self.original_image, self.simulated_image)
        medsymac_value = self.calculate_3d_medsymac(self.original_image, self.simulated_image)
        
        print("\n=== 3D 오차 지표 ===")
        print(f"3D NRMSE: {nrmse_value:.6f}")
        print(f"3D SMAPE: {smape_value:.2f}%")
        print(f"3D LOGAC: {logac_value:.6f}")
        print(f"3D MEDSYMAC: {medsymac_value:.6f}")
        
        # 기본적인 검증
        self.assertGreaterEqual(cc_value, -1)
        self.assertLessEqual(cc_value, 1)
        self.assertGreaterEqual(mi_value, 0)
        self.assertGreaterEqual(ssim_value, 0)
        self.assertLessEqual(ssim_value, 1)
        self.assertGreaterEqual(nrmse_value, 0)
        self.assertGreaterEqual(smape_value, 0)
        self.assertLessEqual(smape_value, 100)

if __name__ == '__main__':
    unittest.main()