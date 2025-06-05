# IMPORTS
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class SignalGenerator:
    """
    Class that generates 2 types of signals:
    - Random signals with polynomial and cosine components
    - ECG signals from the MIT-BIH database, the signals are segmented and the dominant frequencies are extracted in order to generate the dataset.
    """
    def __init__(self):
        self.time = None

    def generate_signal(self, a, b, w, phi, nb_points, Fe, noise_std, normalize=False):
        """
        Genrate a signal with polynomial and cosine components.

        args:
            a: list of polynomial coefficients
            b: list of cosine amplitudes
            w: list of cosine frequencies
            phi: list of cosine phases
            nb_points: number of points in the signal
            Fe: sampling frequency
            noise_std: standard deviation of the noise
            normalize: whether to normalize the signal or not
        return:
            signal: the generated signal
        """
        self.time = np.linspace(0, nb_points/Fe, nb_points)
        poly_part = sum(a[i] * (self.time ** i) for i in range(len(a)))
        cosine_part = sum(b[i] * np.cos(2 * np.pi * w[i] * self.time + phi[i]) for i in range(len(w)))
        noise = np.random.normal(0, noise_std, size=nb_points)
        signal = poly_part + cosine_part + noise

        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

        return signal

    def generate_dataset(
            self, 
            num_signals=None, 
            a_range=None, 
            b_range=None, 
            w_range=None, 
            phi_range=None, 
            nb_points=None, 
            noise_std=None, 
            Fe=None, 
            N_p=None,
            N_c=None,
            ecg=False,
            n_param=20,
            normalize=False,
            verbose=False):
        """
        Generate a dataset of signals with random parameters contained in the parameters range.
        If ecg is True then generate a dataset of ecg signals from the open source WaveForm DataBase open source.

        args:
            num_signals: number of signals to generate
            a_range: range of the polynomial coefficients
            b_range: range of the cosine amplitudes
            w_range: range of the cosine frequencies
            phi_range: range of the cosine phases
            nb_points: number of points in the signal
            noise_std: standard deviation of the noise
            Fe: sampling frequency
            N_p: degree of the polynomial
            N_c: number of cosines
            ecg: whether to generate ecg signals or not
            n_param: number of dominant frequencies to extract from the ecg signals
            normalize: whether to normalize the signals or not
            verbose: whether to print the parameters or not
        return:
            dataset: the generated dataset
        """
        dataset = []

        if verbose:
            print("=" * 60)
            print(" " * 20 + "Dataset Generation Parameters")
            print("=" * 60)
            print(f"{'Parameter':<25} | {'Value':<30}")
            print("-" * 60)
            print(f"{'Number of Signals':<25} | {num_signals:<30}")
            print(f"{'Polynomial Coefficients':<25} | {a_range!s:<30}") 
            print(f"{'Cosine Amplitudes':<25} | {b_range!s:<30}")
            print(f"{'Frequencies':<25} | {w_range!s:<30}")
            print(f"{'Phases':<25} | {phi_range!s:<30}")
            print(f"{'Number of Points':<25} | {nb_points:<30}")
            print(f"{'Noise Std Dev':<25} | {noise_std:<30}")
            print(f"{'Sampling Frequency (Fe)':<25} | {Fe:<30}")
            print(f"{'Polynomial Degree (N_p)':<25} | {N_p:<30}")
            print(f"{'Number of Cosines (N_c)':<25} | {N_c:<30}")
            print(f"{'Normalization':<25} | {str(normalize):<30}")
            print(f"{'ECG':<25} | {ecg:<30}")
            print("=" * 60)

        if ecg:
        
            dataset = []

            data_path = '../dataset/mitdb'
            if not os.path.exists(data_path):
                print(f"Downloading dataset in {data_path}")
                wfdb.dl_database('mitdb', dl_dir=data_path)
    
            record_names = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.hea')]


            for record_name in tqdm(record_names, desc="Processing WFDB records"):
       
                record = wfdb.rdrecord(f'../dataset/mitdb/{record_name}')
                ecg_signal = record.p_signal[:, 0] 

                num_segments = len(ecg_signal) // nb_points
                for i in range(num_segments):

                    segment = ecg_signal[i * nb_points : (i + 1) * nb_points]
                    fft_result = np.fft.fft(segment)
                    frequencies = np.fft.fftfreq(len(segment), d=1/record.fs)  
                    amplitudes = np.abs(fft_result)  
                    phases = np.angle(fft_result) 
                    positive_frequencies = frequencies[frequencies >= 0]
                    positive_amplitudes = amplitudes[frequencies >= 0]
                    positive_phases = phases[frequencies >= 0]

                    dominant_indices = np.argsort(positive_amplitudes)[-n_param:] 
                    dominant_frequencies = positive_frequencies[dominant_indices]
                    dominant_amplitudes = positive_amplitudes[dominant_indices]
                    dominant_phases = positive_phases[dominant_indices]

                    params = {
                        "a": dominant_amplitudes.tolist(),
                        "w": dominant_frequencies.tolist(),  
                        "phi": dominant_phases.tolist() 
                    }

                    dataset.append((segment, params))

            return dataset
            
        else:

            for _ in tqdm(range(num_signals), desc='generating signals'):
                a = [np.random.uniform(a_range[0], a_range[1]) for _ in range(N_p)]
                b = [np.random.uniform(b_range[0], b_range[1]) for _ in range(N_c)]
                w = [np.random.uniform(w_range[0], w_range[1]) for _ in range(N_c)]

                w.sort()
                
                phi = [np.random.uniform(phi_range[0], phi_range[1]) for _ in range(N_c)]

                signal = self.generate_signal(a, b, w, phi, nb_points=nb_points, Fe=Fe, noise_std=noise_std, normalize=normalize)

                dataset.append((signal, {"a": a, "b": b, "w": [float(np.log1p(w)) for w in w], "phi": [float(np.cos(phi)) for phi in phi]}))

            return dataset