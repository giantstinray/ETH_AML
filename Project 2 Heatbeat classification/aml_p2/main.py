import csv
import os
import neurokit2 as nk
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import multiprocess as mp
from tqdm import tqdm
import hrvanalysis

# Change this path to the folder that has your data
fpath = "data/"

def load_data(train_path, test_path):
    X_train = pd.read_csv(train_path, index_col="id")
    y_train = X_train.iloc[:, 0]
    X_train = X_train.iloc[:, 1:]
    X_test = pd.read_csv(test_path, index_col="id")
    return transform_data(X_train), y_train.values, transform_data(X_test)

def transform_data(df):
    return np.array([row.dropna().to_numpy(dtype='float32') for _, row in df.iterrows()], dtype=object)

X_train_raw, y_train_raw, X_test_raw = load_data(
    train_path = f"{fpath}train.csv",
    test_path = f"{fpath}test.csv"
)
print(
    "X_train_raw shape: ",
    X_train_raw.shape,
    "\ny_train_raw shape",
    y_train_raw.shape,
    "\nX_test_raw shape",
    X_test_raw.shape,
)

def _feature_engineering(signal, method="neurokit"):
    # Feature vector to store the extracted features
    features = []

    try:
        # Attempt using the first method
        signals, info = nk.ecg_process(
            ecg_signal=signal, sampling_rate=300, method="neurokit"
        )
    except Exception as e:
        print("Method 'neurokit' failed:", e)
        try:
            # Fallback to second method
            signals, info = nk.ecg_process(
                ecg_signal=signal, sampling_rate=300, method="pantompkins1985"
            )
        except Exception as e:
            print("Method 'pantompkins1985' failed:", e)
            try:
                # Fallback to third method
                signals, info = nk.ecg_process(
                    ecg_signal=signal, sampling_rate=300, method="hamilton2002"
                )
            except Exception as e:
                print("Method 'hamilton2002' failed:", e)
                try:
                    # Fallback to fourth method
                    signals, info = nk.ecg_process(
                        ecg_signal=signal, sampling_rate=300, method="elgendi2010"
                    )
                except Exception as e:
                    print("Method 'elgendi2010' failed:", e)
                    try:
                        # Fallback to fifth method
                        signals, info = nk.ecg_process(
                            ecg_signal=signal, sampling_rate=300, method="engzeemod2012"
                        )
                    except Exception as e:
                        print("Method 'engzeemod2012' failed:", e)
                        raise ValueError(
                            "All methods failed. Please check the input signal."
                        )

    # Access relevant outputs from the tuple
    rpeaks = info["ECG_R_Peaks"]
    heart_rate = np.array(signals["ECG_Rate"][info["ECG_R_Peaks"]])

    # Feature summary stats vectors:
    # - peaks
    # - differences of peaks
    # - cleaned signals of peaks
    # - cleaned signals of differences of peaks
    # - cleaned differences of signals of peaks
    for key in ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]:
        idx = np.array(info[key])[~np.isnan(info[key])]
        features += summary_stats(idx)
        features += summary_stats(np.diff(idx))
        features += summary_stats(np.array(signals["ECG_Clean"][idx.astype(int)]))
        features += summary_stats(
            np.array(signals["ECG_Clean"][np.diff(idx).astype(int)])
        )
        features += summary_stats(np.diff(signals["ECG_Clean"][idx.astype(int)]))

    # Feature vectors: PR interval, PR segment, QT interval, ST segment
    p_r_int = np.array(
        signals["ECG_Clean"] * signals["ECG_P_Onsets"]
        - signals["ECG_Clean"] * signals["ECG_R_Onsets"]
    )
    p_r_seg = np.array(
        signals["ECG_Clean"] * signals["ECG_P_Offsets"]
        - signals["ECG_Clean"] * signals["ECG_R_Onsets"]
    )
    q_t_int = np.array(
        signals["ECG_Clean"] * signals["ECG_Q_Peaks"]
        - signals["ECG_Clean"] * signals["ECG_T_Offsets"]
    )
    s_t_seg = np.array(
        signals["ECG_Clean"] * signals["ECG_S_Peaks"]
        - signals["ECG_Clean"] * signals["ECG_T_Onsets"]
    )
    qrs_cpx = np.array(
        signals["ECG_Clean"] * signals["ECG_Q_Peaks"]
        - signals["ECG_Clean"] * signals["ECG_S_Peaks"]
    )
    for item in [p_r_int, p_r_seg, q_t_int, s_t_seg, qrs_cpx]:
        features += summary_stats(item[item != 0])

    # Spectral features using FFT (same as before)
    clip = signal[rpeaks[0] : rpeaks[-1]]  # Clip signal around R-peaks
    freq = np.fft.rfftfreq(
        len(clip), 1 / 300
    )  # Frequency bin edges (sampling rate assumed as 300 Hz)
    spec = np.abs(np.fft.rfft(clip)) / len(clip)  # Spectral magnitude
    freq, spec = binned(freq, spec, 50.0, 100, np.max)  # Bin and apply max function
    features += list(spec)

    # Autocorrelation of the signal
    autocorr = np.correlate(clip, clip, mode="full") / len(clip)
    autocorr = autocorr[autocorr.size // 2 :]
    time = np.linspace(0, len(clip) / 300, len(clip))
    time, autocorr = binned(time, autocorr, 1.0, 100, np.mean)
    features += list(autocorr)

    # Heart rate features (mean, std, median, variance)
    features += summary_stats(heart_rate)

    # HRV (Heart Rate Variability) - difference between successive R-peaks (R-R intervals)
    rr_intervals = np.diff(rpeaks) / 300  # R-R intervals in seconds
    features += summary_stats(
        rr_intervals
    )  # HRV: mean, std, median, variance of R-R intervals

    # Signal Entropy: Measure of signal complexity (Shannon Entropy)
    entropy = signal_entropy(clip)
    features.append(entropy)

    # Time-domain features: Mean, Standard deviation, Skewness, Kurtosis
    features += time_domain_features(clip)

        # New Features
    # 1. Normalized RR intervals
    rr_normalized = (rr_intervals - np.min(rr_intervals)) / (
        np.max(rr_intervals) - np.min(rr_intervals)
    )
    features += summary_stats(rr_normalized)

    # 2. Skewness and Kurtosis of RR intervals
    features.append(skew(rr_intervals))
    features.append(kurtosis(rr_intervals))

    # 3. Peak-to-peak amplitude variability
    r_amplitudes = signal[rpeaks]
    features.append(np.std(r_amplitudes))

    # 4. Median Absolute Deviation (MAD) of Clean ECG
    mad_ecg_clean = np.median(
        np.abs(signals["ECG_Clean"] - np.median(signals["ECG_Clean"]))
    )
    features.append(mad_ecg_clean)

    # 5. Fraction of low-amplitude peaks
    low_amp_fraction = np.sum(r_amplitudes < 0.5 * np.mean(r_amplitudes)) / len(
        r_amplitudes
    )
    features.append(low_amp_fraction)

    # 6. Energy of the signal
    features.append(np.sum(np.square(signal)))

    # 7. Normalized power in frequency bands
    vlf_power = np.sum(spec[(freq >= 0.003) & (freq < 0.04)])
    lf_power = np.sum(spec[(freq >= 0.04) & (freq < 0.15)])
    hf_power = np.sum(spec[(freq >= 0.15) & (freq < 0.4)])
    total_power = vlf_power + lf_power + hf_power
    features += [
        vlf_power / total_power if total_power != 0 else 0,
        lf_power / total_power if total_power != 0 else 0,
        hf_power / total_power if total_power != 0 else 0,
    ]

    # 8. Harmonic mean of RR intervals
    rr_harmonic_mean = len(rr_intervals) / np.sum(1.0 / rr_intervals)
    features.append(rr_harmonic_mean)

    # 9. Baseline wander amplitude
    baseline = nk.signal_filter(signal, sampling_rate=300, lowcut=0.5, method="butter")
    baseline_wander_amplitude = np.max(baseline) - np.min(baseline)
    features.append(baseline_wander_amplitude)

    # # 10. Feature extraction using hrvanalysis library
    # hrv_features  = extract_hrv_features(rpeaks)   # Convert RR intervals to milliseconds
    # features += hrv_features
    
    # # 11. Extract template features using ECG module from biosppy
    # templates = ecg.ecg(signal=signal, sampling_rate=300, show=False)["templates"]
    # template_features = extract_template_features(templates)
    # features += template_features

    # Return the extracted feature vector
    return features


    # Return the extracted feature vector
    return features


# Function to calculate Shannon entropy of a signal
def signal_entropy(signal):
    prob_density, _ = np.histogram(signal, bins=10, density=True)
    prob_density = prob_density[prob_density > 0]  # Remove zero probabilities
    entropy = -np.sum(prob_density * np.log(prob_density))  # Shannon entropy
    return entropy


# Time-domain statistical features (mean, std, skewness, kurtosis)
def time_domain_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return [mean, std, skewness, kurt]

def extract_template_features(templates):
    
    med_template = np.median(templates, axis=0)
    med_std = np.std(med_template)
    med_mean = np.mean(med_template)
    med_med = np.median(med_template)

    mean_template = np.mean(templates, axis=0)
    mean_std = np.std(mean_template)
    mean_mean = np.mean(mean_template)
    mean_med = np.median(mean_template)

    std_template = np.std(templates, axis = 0)
    std_std = np.std(std_template)
    std_mean = np.mean(std_template)
    std_med = np.median(std_template)

    return [med_std, med_mean, med_med, mean_std, mean_mean, mean_med, std_std, std_mean, std_med]

def extract_hrv_features(r_peaks):
    tdf_names = [
        "mean_nni",
        "sdnn",
        "sdsd",
        "nni_50",
        "pnni_50",
        "nni_20",
        "pnni_20",
        "rmssd",
        "median_nni",
        "range_nni",
        "cvsd",
        "cvnni",
        "mean_hr",
        "max_hr",
        "min_hr",
        "std_hr",
    ]

    gf_names = ["triangular_index"]

    fdf_names = ["lf", "hf", "lf_hf_ratio", "lfnu", "hfnu", "total_power", "vlf"]

    cscv_names = [
        "csi",
        "cvi",
        "Modified_csi",
    ]

    pcp_names = ["sd1", "sd2", "ratio_sd2_sd1"]
    features = np.ndarray((len(tdf_names) + len(gf_names) + len(fdf_names) + len(cscv_names) + len(pcp_names),))
    features[:] = 0
    features = list(features)
    
    try:
        tdf = hrvanalysis.get_time_domain_features(r_peaks)
        gf = hrvanalysis.get_geometrical_features(r_peaks)
        fdf = hrvanalysis.get_frequency_domain_features(r_peaks)
        cscv = hrvanalysis.get_csi_cvi_features(r_peaks)
        pcp = hrvanalysis.get_poincare_plot_features(r_peaks)
        samp = hrvanalysis.get_sampen(r_peaks)
    except:
        return []

    for name in tdf_names:
        features.append(tdf[name])

    for name in gf_names:
        features.append(gf[name])

    for name in fdf_names:
        features.append(fdf[name])

    for name in cscv_names:
        features.append(cscv[name])

    for name in pcp_names:
        features.append(pcp[name])

    features.append(samp["sampen"])

    return features


# Binned function for downsampling and applying a function over binned data
def binned(x, y, xend, nbins, func):
    bx = np.linspace(x[0], xend, nbins + 1)
    idx = np.digitize(x, bx) - 1  # Bin assignments
    bx = (bx[1:] + bx[:-1]) / 2  # Bin centers
    by = np.array(
        [func(y[idx == i]) for i in range(nbins)]
    )  # Apply function to each bin
    return bx, by


# summary_stats function to compute statistical measures
def summary_stats(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [0, 0, 0, 0]
    else:
        return [np.mean(x), np.std(x), np.median(x), np.var(x)]


# Parallelize using multiprocessing (avoid excessive memory consumption by processing in chunks)
def feature_engineering(X, inverse=False):
    with mp.Pool(mp.cpu_count()) as pool:
        Xn = list(tqdm(pool.imap(_feature_engineering, X), total=len(X)))
    return np.array(Xn)

# Do feature engineering
X_train = feature_engineering(X_train_raw)
X_test = feature_engineering(X_test_raw)
print(
    "X_train shape: ",
    X_train.shape,
    "\nX_test shape",
    X_test.shape,
)

# Save the data as npy to save time for later use
np.save(f"{fpath}X_train.npy", X_train)
np.save(f"{fpath}X_test.npy", X_test)
np.save(f"{fpath}y_train.npy", y_train_raw)