# -*- coding:utf-8 -*-
from dcm2nii_converter import DCM2NIIConverter
from coregistration import Coregistration

if __name__ == '__main__':
    input_dir = r'D:\DATASET\Breast_MRI\Breast_Cancer_MRI'
    output_dir = r'C:\Users\BMC\Desktop\CE Dataset'

    DCM2NIIConverter(input_dir).main()
    Coregistration(output_dir).main()
