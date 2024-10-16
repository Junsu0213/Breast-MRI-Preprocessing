import os
import zipfile
import shutil
import subprocess
import logging
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import exposure
from scipy.ndimage import binary_opening, binary_closing
import ants
import glob
from datetime import datetime
import streamlit as st

logging.basicConfig(level=logging.DEBUG)


class DICOMConverter:
    def __init__(self, zip_path, extract_dir='temp_zip', output_dir='output'):
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.output_dir = output_dir

    def extract_zip(self):
        logging.info(f"Extracting ZIP file: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
        logging.info(f"Extraction complete: {self.extract_dir}")

    def get_subfolders(self):
        subfolders = [f.path for f in os.scandir(self.extract_dir) if f.is_dir()]
        logging.info(f"Found subfolders: {subfolders}")
        return subfolders

    def run_dcm2niix(self):
        subfolders = self.get_subfolders()
        for subfolder_path in subfolders:
            subfolder_name = os.path.basename(subfolder_path)
            output_subfolder = os.path.join(self.output_dir, subfolder_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder, exist_ok=True)
                logging.info(f"Created output directory: {output_subfolder}")

            subfolder_subdirs = [
                f.path for f in os.scandir(subfolder_path) if f.is_dir()
            ]

            for subfolder_subdir in subfolder_subdirs:
                subfolder_subdir_name = os.path.basename(subfolder_subdir)

                try:
                    result = subprocess.run(
                        [
                            "dcm2niix", "-z", "y", "-f",
                            f"{subfolder_subdir_name}", "-o",
                            output_subfolder, subfolder_subdir
                        ],
                        check=True, capture_output=True, text=True
                    )
                    logging.info(f"dcm2niix output: {result.stdout}")
                    if result.stderr:
                        logging.error(f"dcm2niix error: {result.stderr}")

                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"Error running dcm2niix in {subfolder_subdir}: {e.stderr}"
                    )

    def process(self):
        self.extract_zip()
        self.run_dcm2niix()


class NIfTIChecker:
    def __init__(self, input_dirs, output_dir):
        self.input_dirs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
        self.output_dir = output_dir

    def check_dim4_is_3(self, nifti_file):
        img = nib.load(nifti_file)
        shape = img.header.get_data_shape()
        if len(shape) >= 4 and shape[3] == 3:
            logging.info(f"Fourth dimension of {nifti_file} is 3.")
            return True
        else:
            logging.info(
                f"Fourth dimension of {nifti_file} is not 3. Current shape: {shape}"
            )
            return False

    def split_dwi_file(self, nifti_file, output_subdir):
        img = nib.load(nifti_file)
        data = img.get_fdata()
        affine = img.affine

        b0 = data[:, :, :, 0]
        b800 = data[:, :, :, 1]
        b1200 = data[:, :, :, 2]

        b0_img = nib.Nifti1Image(b0, affine)
        b800_img = nib.Nifti1Image(b800, affine)
        b1200_img = nib.Nifti1Image(b1200, affine)

        b0_file = os.path.join(output_subdir, "DWI_b0_orig.nii.gz")
        b800_file = os.path.join(output_subdir, "DWI_b800_orig.nii.gz")
        b1200_file = os.path.join(output_subdir, "DWI_b1200_orig.nii.gz")

        nib.save(b0_img, b0_file)
        logging.info(f"Saved {b0_file}")

        nib.save(b800_img, b800_file)
        logging.info(f"Saved {b800_file}")

        nib.save(b1200_img, b1200_file)
        logging.info(f"Saved {b1200_file}")

    def extract_last_dim_image(self, nifti_file, output_subdir):
        img = nib.load(nifti_file)
        data = img.get_fdata()
        affine = img.affine

        if len(data.shape) >= 4:
            last_dim_index = data.shape[3] - 1
            last_dim_image = data[:, :, :, last_dim_index]

            last_dim_img = nib.Nifti1Image(last_dim_image, affine)
            last_dim_file = os.path.join(output_subdir, "refCEMRI.nii.gz")

            nib.save(last_dim_img, last_dim_file)
            logging.info(f"Saved {last_dim_file}")
        else:
            logging.error(f"{nifti_file} is not 4D.")
            new_path = os.path.join(output_subdir, "refCEMRI.nii.gz")
            shutil.copy2(nifti_file, new_path)
            logging.info(f"Copied {nifti_file} to {new_path}")

    def rename_and_copy(self, nifti_file, new_name, output_subdir):
        new_path = os.path.join(output_subdir, new_name)
        shutil.copy2(nifti_file, new_path)
        logging.info(f"Copied and renamed {nifti_file} to {new_path}")

    def process_files(self):
        for input_dir in self.input_dirs:
            subdir_name = os.path.basename(input_dir)
            output_subdir = os.path.join(self.output_dir, subdir_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir, exist_ok=True)
                logging.info(f"Created output sub-directory: {output_subdir}")

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".nii.gz"):
                        nifti_file = os.path.join(root, file)
                        file_lower = file.lower()
                        logging.info(f"Processing file: {nifti_file}")
                        if "dwi.nii.gz" in file_lower and self.check_dim4_is_3(nifti_file):
                            self.split_dwi_file(nifti_file, output_subdir)
                        elif "reference_ce_mri.nii.gz" in file_lower:
                            self.extract_last_dim_image(nifti_file, output_subdir)
                        elif (
                                "fs_t1wi.nii.gz" in file_lower and
                                "nonfs_t1wi.nii.gz" not in file_lower
                        ):
                            self.rename_and_copy(nifti_file, "fsT1.nii.gz", output_subdir)
                        elif "nonfs_t1wi.nii.gz" in file_lower:
                            self.rename_and_copy(nifti_file, "nonfsT1.nii.gz", output_subdir)
                        else:
                            new_name = os.path.join(output_subdir, file)
                            shutil.copy2(nifti_file, new_name)
                            logging.info(f"Copied {nifti_file} to {new_name}")


class MRIProcessor:
    def __init__(self, input_files, smoothing=3, output=None, prefix='output'):
        self.input_files = input_files
        self.smoothing = smoothing
        self.prefix = prefix
        self.output_filenames = output if output else {
            'output_image': '{prefix}_output_image.nii.gz',
            'mask_smoothed': '{prefix}_output_image_mask.nii.gz',
            'masked_output_image': '{prefix}_masked_output_image.nii.gz'
        }
        self.output_filenames = {
            key: value.format(prefix=self.prefix)
            for key, value in self.output_filenames.items()
        }
        self.data = self.load_images()
        self.summed_image = None
        self.object_mask_smoothed = None
        self.masked_summed_image = None

    def load_images(self):
        return [nib.load(f).get_fdata() for f in self.input_files]

    def sum_images(self):
        self.summed_image = np.sum(self.data, axis=0)

    def enhance_contrast(self):
        p2, p98 = np.percentile(self.summed_image, (2, 98))
        rescaled_image = exposure.rescale_intensity(self.summed_image, in_range=(p2, p98))
        rescaled_image = rescaled_image.astype(np.float32)
        rescaled_image = (rescaled_image * 65535 / rescaled_image.max()).astype(np.uint16)
        self.summed_image = exposure.equalize_adapthist(
            rescaled_image, kernel_size=None, clip_limit=0.03, nbins=256
        )

    def cluster_and_smooth(self):
        flat_data = self.summed_image.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(flat_data)
        labels = kmeans.labels_
        segmented_image = labels.reshape(self.summed_image.shape)

        if np.mean(self.summed_image[segmented_image == 0]) > np.mean(self.summed_image[segmented_image == 1]):
            object_mask = segmented_image == 0
        else:
            object_mask = segmented_image == 1

        structure = np.ones((self.smoothing, self.smoothing, self.smoothing), dtype=int)
        self.object_mask_smoothed = binary_opening(object_mask, structure=structure)
        self.object_mask_smoothed = binary_closing(self.object_mask_smoothed, structure=structure)

    def apply_mask(self):
        self.masked_summed_image = self.summed_image * self.object_mask_smoothed

    def save_results(self):
        os.makedirs(os.path.dirname(list(self.output_filenames.values())[0]), exist_ok=True)
        nib.save(
            nib.Nifti1Image(
                self.summed_image, nib.load(self.input_files[0]).affine
            ),
            self.output_filenames['output_image']
        )
        nib.save(
            nib.Nifti1Image(
                self.object_mask_smoothed.astype(np.uint8),
                nib.load(self.input_files[0]).affine
            ),
            self.output_filenames['mask_smoothed']
        )
        nib.save(
            nib.Nifti1Image(
                self.masked_summed_image, nib.load(self.input_files[0]).affine
            ),
            self.output_filenames['masked_output_image']
        )
        logging.info(f"Results saved: {self.output_filenames}")

    def process(self):
        self.sum_images()
        self.enhance_contrast()
        self.cluster_and_smooth()
        self.apply_mask()
        self.save_results()
        self.show_slices()
        print(f'Saved: {self.output_filenames}')

    def show_slices(self):
        central_slice = self.summed_image.shape[2] // 2
        slices = [
            self.summed_image[:, :, central_slice],
            self.object_mask_smoothed[:, :, central_slice],
            self.masked_summed_image[:, :, central_slice]
        ]
        titles = ["Summed Image", "Smoothed Mask", "Masked Summed Image"]
        fig, axes = plt.subplots(1, len(slices), figsize=(15, 5))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
            axes[i].set_title(titles[i])
        plt.show()


class MRITransformer:
    def __init__(self, t1_image_path, dwi_dir='output', output_dir='output'):
        self.t1_image_path = t1_image_path
        self.dwi_dir = dwi_dir
        self.output_dir = output_dir

    def read_and_normalize_images(self):
        self.t1_image = ants.image_read(self.t1_image_path)
        self.t1_image = self.normalize_image(self.t1_image)

    def normalize_image(self, image):
        img_np = image.numpy()
        img_np = img_np.astype(np.float32)
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
        return ants.from_numpy(
            img_np, origin=image.origin, spacing=image.spacing,
            direction=image.direction
        )

    def register_images(self):
        self.warp_tx = ants.registration(
            fixed=self.t1_image, moving=self.t1_image, type_of_transform='Rigid'
        )

    def apply_transform_to_split_images(self):
        t1_pre_image = ants.image_read(
            os.path.join(self.output_dir, 'nonfsT1.nii.gz')
        )

        split_images = [
            f for f in os.listdir(self.dwi_dir) if f.startswith('DWI_b') and f.endswith('_orig.nii.gz')
        ]
        for split_image in split_images:
            img_path = os.path.join(self.dwi_dir, split_image)
            img = ants.image_read(img_path)
            resampled_img = ants.resample_image_to_target(img, t1_pre_image, interp_type='bSpline')
            transformed_img = ants.apply_transforms(
                fixed=self.t1_image, moving=resampled_img,
                transformlist=self.warp_tx['fwdtransforms']
            )
            new_name = self.get_new_name(split_image)
            output_path = os.path.join(self.output_dir, new_name)
            ants.image_write(transformed_img, output_path)
            print(f'Transformed and saved: {output_path}')

    def get_new_name(self, old_name):
        if "DWI_b0_orig.nii.gz" in old_name:
            return "DWI_b0_transformed.nii.gz"
        elif "DWI_b800_orig.nii.gz" in old_name:
            return "DWI_b800_transformed.nii.gz"
        elif "DWI_b1200_orig.nii.gz" in old_name:
            return "DWI_b1200_transformed.nii.gz"
        else:
            return old_name

    def process(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.read_and_normalize_images()
        self.register_images()
        self.apply_transform_to_split_images()


class MRIPipeline:
    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.output_dir = 'output'
        self.archive_dir = 'archive_zip'
        self.temp_dir = 'temp_zip'
        self.output_zip_dir = 'output_zip'
        self.checked_output_dir = 'checked_output'

    def delete_folder(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Deleted folder: {folder_path}")

    def run(self):
        subject_id = os.path.splitext(os.path.basename(self.zip_file))[0]
        output_subject_folder = os.path.join(self.output_dir, subject_id)
        checked_output_subject_folder = os.path.join(self.checked_output_dir, subject_id)
        temp_subject_folder = os.path.join(self.temp_dir, subject_id)

        self.delete_folder(temp_subject_folder)
        self.delete_folder(output_subject_folder)
        self.delete_folder(checked_output_subject_folder)

        with st.status(f"Processing {self.zip_file}...", expanded=True) as status:
            status.update(label="Extracting and converting DICOM files...")
            st.write("Extracting and converting DICOM files...")
            converter = DICOMConverter(zip_path=self.zip_file, output_dir=self.output_dir)
            converter.process()

            status.update(label="Checking and processing NIfTI files...")
            st.write("Checking and processing NIfTI files...")
            checker = NIfTIChecker(
                input_dirs=[output_subject_folder], output_dir=self.checked_output_dir
            )
            checker.process_files()

            status.update(label="Archiving the original ZIP file...")
            st.write("Archiving the original ZIP file...")
            if not os.path.exists(self.archive_dir):
                os.makedirs(self.archive_dir)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            archive_name = os.path.join(self.archive_dir, f"{subject_id}_{timestamp}.zip")
            shutil.move(self.zip_file, archive_name)

            status.update(label="Processing T1 files...")
            st.write("Processing T1 files...")
            t1_files = [
                os.path.join(self.checked_output_dir, subject_id, f)
                for f in ['nonfsT1.nii.gz', 'fsT1.nii.gz', 'refCEMRI.nii.gz']
            ]
            t1_processor = MRIProcessor(
                t1_files, smoothing=2, output={
                    'output_image': f'{self.checked_output_dir}/{subject_id}/T1_output_image.nii.gz',
                    'mask_smoothed': f'{self.checked_output_dir}/{subject_id}/T1_output_image_mask.nii.gz',
                    'masked_output_image': f'{self.checked_output_dir}/{subject_id}/T1_masked_output_image.nii.gz'
                }, prefix="T1"
            )
            t1_processor.process()

            status.update(label="Processing DWI files...")
            st.write("Processing DWI files...")
            dwi_input_files = glob.glob(
                os.path.join(self.checked_output_dir, subject_id, 'DWI_b*_orig.nii.gz')
            )
            dwi_processor = MRIProcessor(
                dwi_input_files, smoothing=2, output={
                    'output_image': f'{self.checked_output_dir}/{subject_id}/dwi_output_image.nii.gz',
                    'mask_smoothed': f'{self.checked_output_dir}/{subject_id}/dwi_output_image_mask.nii.gz',
                    'masked_output_image': f'{self.checked_output_dir}/{subject_id}/dwi_masked_output_image.nii.gz'
                }, prefix="dwi"
            )
            dwi_processor.process()

            status.update(label="Transforming DWI files...")
            st.write("Transforming DWI files...")
            transformer = MRITransformer(
                t1_image_path=os.path.join(
                    self.checked_output_dir, subject_id, 'T1_masked_output_image.nii.gz'
                ),
                dwi_dir=os.path.join(self.checked_output_dir, subject_id),
                output_dir=os.path.join(self.checked_output_dir, subject_id)
            )
            transformer.process()

            status.update(label="Creating final output ZIP file...")
            st.write("Creating final output ZIP file...")
            zip_filename = f"{subject_id}_output.zip"
            if not os.path.exists(self.output_zip_dir):
                os.makedirs(self.output_zip_dir)
            with zipfile.ZipFile(
                    os.path.join(self.output_zip_dir, zip_filename), 'w'
            ) as zipf:
                for file in [
                    'DWI_b0_orig.nii.gz', 'DWI_b800_orig.nii.gz',
                    'DWI_b1200_orig.nii.gz', 'nonfsT1.nii.gz', 'fsT1.nii.gz',
                    'refCEMRI.nii.gz', 'DWI_b0_transformed.nii.gz',
                    'DWI_b800_transformed.nii.gz', 'DWI_b1200_transformed.nii.gz'
                ]:
                    zipf.write(
                        os.path.join(self.checked_output_dir, subject_id, file),
                        arcname=file
                    )

            self.delete_folder(output_subject_folder)
            self.delete_folder(checked_output_subject_folder)
            self.delete_folder(temp_subject_folder)

            status.update(label="Processing complete!", state="complete", expanded=False)

        print(f"Processed and archived: {subject_id}")


class MRIPipelineRunner:
    def __init__(self, zip_folder):
        self.zip_folder = zip_folder

    def run(self):
        zip_files = [
            os.path.join(self.zip_folder, f) for f in os.listdir(self.zip_folder)
            if f.endswith('.zip')
        ]
        for zip_file in zip_files:
            pipeline = MRIPipeline(zip_file)
            pipeline.run()


def main():
    st.title("Breast MR ImgProc: DWI-to-Sharpen")
    st.markdown("Transforms DWI into sharpened DWI images, registered and resampled to T1WI space.")
    st.markdown("Upload single or multiple ZIP files and download the processed results.")
    st.divider()

    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'combined_output_ready' not in st.session_state:
        st.session_state.combined_output_ready = False
    if 'individual_file_ready' not in st.session_state:
        st.session_state.individual_file_ready = False
    if 'individual_file_path' not in st.session_state:
        st.session_state.individual_file_path = None
    if 'combined_output_path' not in st.session_state:
        st.session_state.combined_output_path = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    uploaded_files = st.file_uploader(
        "Upload single or multiple ZIP files (including folders DWI, FS_T1WI, NonFS_T1WI, reference_CE_MRI)",
        type="zip", accept_multiple_files=True
    )

    if uploaded_files:
        zip_folder = "temp_zip"
        output_zip_folder = "output_zip"

        st.session_state.processed_files = []
        if not os.path.exists(zip_folder):
            os.makedirs(zip_folder)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(zip_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.processed_files.append(file_path)

        if st.button("Processing Start", type="primary"):
            runner = MRIPipelineRunner(zip_folder=zip_folder)
            runner.run()

            if len(uploaded_files) == 1:
                zip_file_path = os.path.join(
                    output_zip_folder, os.listdir(output_zip_folder)[0]
                )
                st.session_state.individual_file_path = zip_file_path
                st.session_state.individual_file_ready = True
            else:
                combined_zip_path = os.path.join(zip_folder, "combined_output.zip")
                with zipfile.ZipFile(combined_zip_path, 'w') as combined_zip:
                    for zip_file in os.listdir(output_zip_folder):
                        zip_file_path = os.path.join(output_zip_folder, zip_file)
                        combined_zip.write(
                            zip_file_path, arcname=os.path.basename(zip_file_path)
                        )

                st.session_state.combined_output_ready = True
                st.session_state.combined_output_path = combined_zip_path

            st.session_state.processing_complete = True
            st.rerun()

    if st.session_state.individual_file_ready:
        st.success("Processing complete.")
        if st.download_button(
                label="Download file",
                type="primary",
                data=open(st.session_state.individual_file_path, "rb").read(),
                file_name=os.path.basename(st.session_state.individual_file_path)
        ):
            st.session_state.individual_file_ready = False
            st.session_state.processing_complete = False
            st.session_state.individual_file_path = None
            clear_temp_folders()
            st.rerun()

    if st.session_state.combined_output_ready:
        st.success("Processing complete.")
        if st.download_button(
                label="Download combined_output.zip",
                data=open(st.session_state.combined_output_path, "rb").read(),
                file_name="combined_output.zip"
        ):
            st.session_state.combined_output_ready = False
            st.session_state.processing_complete = False
            st.session_state.combined_output_path = None
            clear_temp_folders()
            st.rerun()

    if st.button("Clear"):
        clear_temp_folders()
        st.rerun()


def clear_temp_folders():
    zip_folder = "temp_zip"
    output_zip_folder = "output_zip"
    if os.path.exists(zip_folder):
        shutil.rmtree(zip_folder)
    if os.path.exists(output_zip_folder):
        shutil.rmtree(output_zip_folder)


if __name__ == "__main__":
    main()
