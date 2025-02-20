from nipype.interfaces.ants import MeasureImageSimilarity
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as struct_sim
from sklearn.metrics import mean_squared_error


# scale to 12 bit range
def scale12bit(img):
    # constants
    new_mean = 2048.
    new_std = 400.

    return np.clip(((img - np.mean(img)) / (np.std(img) / new_std)) + new_mean, 1e-10, 4095)


# get list of locals
start_globals = list(globals().keys())


# perform image comparrison using neighborhood CC
def cc(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
        sim = MeasureImageSimilarity()
        if not verbose:
            sim.terminal_output = 'allatonce'
        sim.inputs.dimension = 3
        sim.inputs.metric = 'CC'
        sim.inputs.fixed_image = true_nii
        sim.inputs.moving_image = pred_nii
        sim.inputs.metric_weight = 1.0
        sim.inputs.radius_or_number_of_bins = 5
        sim.inputs.sampling_strategy = 'None'  # None = dense sampling
        sim.inputs.sampling_percentage = 1.0
        if mask:
            sim.inputs.fixed_image_mask = mask_nii
            sim.inputs.moving_image_mask = mask_nii
        if verbose:
            print(sim.cmdline)

        return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mi(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
        sim = MeasureImageSimilarity()
        if not verbose:
            sim.terminal_output = 'allatonce'
        sim.inputs.dimension = 3
        sim.inputs.metric = 'MI'
        sim.inputs.fixed_image = true_nii
        sim.inputs.moving_image = pred_nii
        sim.inputs.metric_weight = 1.0
        sim.inputs.radius_or_number_of_bins = 64
        sim.inputs.sampling_strategy = 'None'  # None = dense sampling
        sim.inputs.sampling_percentage = 1.0
        if mask:
            sim.inputs.fixed_image_mask = mask_nii
            sim.inputs.moving_image_mask = mask_nii
        if verbose:
            print(sim.cmdline)

        return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mse(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    sim = MeasureImageSimilarity()
    if not verbose:
        sim.terminal_output = 'allatonce'
    sim.inputs.dimension = 3
    sim.inputs.metric = 'MeanSquares'
    sim.inputs.fixed_image = true_nii
    sim.inputs.moving_image = pred_nii
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 32  # not used
    sim.inputs.sampling_strategy = 'None'  # None = dense sampling
    sim.inputs.sampling_percentage = 1.0
    if mask:
        sim.inputs.fixed_image_mask = mask_nii
        sim.inputs.moving_image_mask = mask_nii
    if verbose:
        print(sim.cmdline)

    return np.abs(sim.run().outputs.similarity)


# normalized RMS error
def nrmse(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # verbosity
    if verbose:
        print("Calculating nrmse")

    return mean_squared_error(true_img[mask_img], pred_img[mask_img], squared=False) / (
                np.max(true_img[mask_img]) - np.min(true_img[mask_img]))


# mean absolute percentage error
def mape(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_img[mask_img])
    pred_img = scale12bit(pred_img[mask_img])
    # verbosity
    if verbose:
        print("Calculating MAPE on 12 bit range scaled images")

    return np.mean(np.fabs((true_img - pred_img)) / (np.fabs(true_img)))


# symmetric mean absolute percentage error
def smape(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_img[mask_img])
    pred_img = scale12bit(pred_img[mask_img])
    # verbosity
    if verbose:
        print("Calculating sMAPE on 12 bit range scaled images")

    return np.mean(np.fabs(pred_img - true_img) / (np.fabs(true_img) + np.fabs(pred_img)))


# log of accuracy ratio
def logac(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_img[mask_img])
    pred_img = scale12bit(pred_img[mask_img])
    # verbosity
    if verbose:
        print("Calculating logac on 12 bit range scaled images")

    return np.mean(np.fabs((np.log(pred_img / true_img))))


# median symmetric accuracy (cf. Morley, 2016)
def medsymac(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
    else:
        mask_img = np.ones_like(true_img, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_img[mask_img])
    pred_img = scale12bit(pred_img[mask_img])
    # verbosity
    if verbose:
        print("Calculating medsymac on 12 bit range scaled images")

    return np.exp(np.median(np.fabs(np.log(pred_img / true_img)))) - 1


# structural similarity index
def ssim(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # load images
    true_img = nib.load(true_nii).get_fdata()
    pred_img = nib.load(pred_nii).get_fdata()
    if mask:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = nib.load(mask_nii).get_fdata().astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_img[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_img[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return struct_sim(true_img, pred_img, win_size=9, data_range=true_img.max() - true_img.min())


def metric_picker(metric, true_nii, pred_nii, mask_nii, mask=False, verbose=False):

    # sanity checks
    if not isinstance(metric, str):
        raise ValueError("Metric parameter must be a string")

    # check for specified loss method and error if not found
    if metric.lower() in globals():
        metric_val = globals()[metric.lower()](true_nii, pred_nii, mask_nii, mask, verbose)
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified metric: '{}' is not one of the available methods: {}".format(metric, methods))

    return metric_val


if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt


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

            
        cc_metric = cc(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        mi_metric = mi(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        ssim_metric = ssim(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        nrmse_metric = nrmse(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        smape_metric = smape(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        logac_metric = logac(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        medsymac_metric = medsymac(ce_img_path, simulated_ce_img_path, mask_nii=None, mask=False, verbose=True)
        
        print(f"""
        Metrics Results for {os.path.basename(ds_dir)}:
        유사도 지표
        - CC: {cc_metric:.4f}
        - MI: {mi_metric:.4f}
        - SSIM: {ssim_metric:.4f}
        오차 지표
        - NRMSE: {nrmse_metric:.4f}
        - SMAPE: {smape_metric:.4f}
        - LOGAC: {logac_metric:.4f}
        - MEDSYMAC: {medsymac_metric:.4f}
        """)
        results.append({
            'Subject': os.path.basename(ds_dir),
            'CC': cc_metric,
            'MI': mi_metric,
            'SSIM': ssim_metric,
            'NRMSE': nrmse_metric,
            'SMAPE': smape_metric,
            'LOGAC': logac_metric,
            'MEDSYMAC': medsymac_metric
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
    plt.savefig('metrics_bar_plot_test.png')
    plt.show()
