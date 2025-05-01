#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

"""
Run Seismic Event Classifier

This script classifies seismic events using multiple deep learning and machine learning models.
It downloads waveform data from IRIS, processes it, and runs it through classification models
to determine the probability of different event types (earthquake, explosion, noise, or surface event).

The script takes an event ID (evid) as input and produces both console output and a text file
containing probability values and classification statistics for each model.

Event types:
- EQ: Earthquake
- EX: Explosion
- NO: Noise
- SU: Surface event (e.g., quarry blast, mining activity)

Models used:
- SeismicCNN_1d: 1D Convolutional Neural Network for seismic classification
- SeismicCNN_2d: 2D Convolutional Neural Network for seismic classification
- QuakeXNet_1d: 1D implementation of QuakeXNet architecture
- QuakeXNet_2d: 2D implementation of QuakeXNet architecture
- ML40sec: Traditional machine learning model using 40-second windows

Usage:
    python script_name.py evid

    where:
        evid: Valid integer event ID in the database

Output:
    - Console output of processing status and probabilities
    - Text file in RESULTS directory with detailed model outputs and statistics

Author: Alex Hutko
Last Modified: Apr 7, 2025
"""

# import std packages
import os
import sys
import random
from time import time
import logging  # Added for better error handling
from typing import List, Tuple, Optional, Dict, Any  # Added type hints

# Check command line arguments and provide usage information
try:
    evid = sys.argv[1]
except IndexError:
    logger.error("Missing required event ID argument")
    print("Usage: thisscript.py evid")
    print("evid must be a valid int event_id")
    sys.exit(1)

# Output file configuration and quick check
results_dir = "RESULTS_SU2"
os.makedirs(results_dir, exist_ok=True)  # Ensure the output directory exists
outfile = os.path.join(results_dir, f"{evid}_output.txt")
if os.path.exists(outfile):
    sys.exit("File already exists; exiting.")

Timport = time()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings # to silence the torch.load warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import h5py
import obspy
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from scipy import stats, signal
from tqdm import tqdm
from joblib import dump, load

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="AutoDateLocator was unable to pick an appropriate interval"
)

# Add custom module path
module_path = os.path.abspath(os.path.join('..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import custom utilities and models
import seis_feature
from utils import apply_cosine_taper, butterworth_filter, resample_array
from neural_network_architectures import (
    QuakeXNet_1d, QuakeXNet_2d, SeismicCNN_1d, SeismicCNN_2d
)
from get_volcano_stations import get_volcano_stations

print("DONE IMPORTING.  Elapsed time: ",time()-Timport)
Tzero = time()

# import third party packages
from obspy.clients.fdsn import Client
client = Client('IRIS')

# import event classifier specific packges
from db.get_event_info2 import unix_to_true_time
from db.get_event_info2 import get_event_info

# Import classification functions
from all_models_classification import (
    compute_window_probs, plot_single_model_probs, plot_all_model_probs
)

# ====================
# 0. Parameters
# ====================

# Open output file in append mode
try:
    f = open(outfile, "a")
except IOError as e:
    logger.error(f"Error opening output file: {e}")
    sys.exit(f"Error opening output file: {e}")

# Set device to GPU if available, else use CPU
device = "cpu"     #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters for signal processing
orig_sr = 100  # Original sampling rate in Hz
new_sr = 50    # New sampling rate in Hz
stride = 5 * orig_sr  # Window moving increment for processing windows (X seconds times original sampling rate)
lowpass = 1  # Lowpass filter cutoff frequency in Hz
highpass = 20  # Highpass filter cutoff frequency in Hz
window_length = 100  # Length of the window for processing in samples
channel_patterns = ["HH", "BH", "EH", "HN", "EN"]  # Channel patterns to filter
trace_window_length = 100.  + 30. + 11.  # Total length of data downloaded (100 sec analysis window, 30 before P, 6 after P.  Note: DL models were trained w windows starting from -20 to -5 sec relative to the P pick time.

# Get event parameters from database using evid
try:
    orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class = get_event_info(evid)
    event_info = [evid, orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class]
    # Calculate time window for data retrieval
    # Start 20 seconds before origin time plus buffer for window processing
    start_time = UTCDateTime(ordate) - 30.  # 30 sec before P is the first analysis window
    if analyst_class == 'su' and len(netstas) < 3:
        netstas, dists_km = get_volcano_stations(lat, lon, netstas, start_time)
except Exception as e:
    logger.error(f"Error retrieving event information: {e}")
    f.close()
    sys.exit(f"Error retrieving event information: {e}")

# Display event information
print('')
print("-------------- ", evid, " --------------")
print("EVENT INFO evid, orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class: ")
print(evid, orid, ordate, lat, lon, dep, mag, mindist, maxdist, netstas, dists_km, analyst_class)

dists_km = [round(x*10)/10. for x in dists_km]
mindist = min(dists_km)

end_time = start_time + trace_window_length
strordate = ordate.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]
strordate2 = ordate.strftime("%Y_%m_%dT%H_%M_%S")
stations_id = netstas

print("ORIGIN DATE  start_time   end_time: ",evid,ordate, start_time, end_time )
f.write(f"ORDATE START END: {evid} {ordate} {start_time} {end_time}\n")
print("STATIONS_ID: ",stations_id)

# Location wildcard, used to match all available locations for the given stations
location = "*"

# Initialize client for accessing IRIS web services
client = Client("IRIS")

# Function to process and compute probabilities for seismic models
def process_model(model, stations_id, dists_km, location, start_time, end_time, one_d, model_type, filename, remember_st=True, st_all=None, integrate_SM=False):
    """
    This function computes the probabilities for a given model, station data, and time window.
    
    Args:
        model: The model used for classification (e.g., deep learning or machine learning).
        stations_id: List of station IDs for data collection.
        dists_km: Distances of stations in km to be used to get approx P travel times.
        st_all: (don
        location: Location of the station (wildcard "*" to match all locations).
        start_time: Start time of the seismic event (obspy.UTCDateTime object).
        end_time: End time of the seismic event (obspy.UTCDateTime object).
        one_d: Boolean indicating whether the model is 1D or 2D.
        model_type: Type of the model, either 'dl' for deep learning or 'ml' for machine learning.
        filename: Name of the output file to save results.
        remember_st (bool, optional): If True, retains the cumulative st_all stream to only download data once per station.
        st_all (Stream, optional): Existing cumulative stream; if None, a new stream is created.
        integrate_SM (optional, default = False): integrate SM data to velocity (with pre-highpass filter above 0.3Hz).

    Returns:
        tuple:
            Processed probabilities and output data.
            st_all (Stream): Updated cumulative waveform stream.
    """
    if st_all is None or not remember_st:
        st_all = Stream()
        
    try:
        return compute_window_probs(
            stations_id=stations_id, dists_km=dists_km, st_all=st_all, location=location, start_time=start_time,
            end_time=end_time, channel_patterns=channel_patterns, client=client,
            stride=stride, orig_sr=orig_sr, new_sr=new_sr, window_length=window_length,
            lowpass=lowpass, highpass=highpass, one_d=one_d, model=model,
            model_type=model_type, filename=filename, remember_st=remember_st, 
            integrate_SM=integrate_SM
        )
    except Exception as e:
        logger.error(f"Error processing model {model_type}: {e}")
        # Return empty results but don't crash the entire script
        return [], None, stations_id, st_all, []

#----- Loading the machine learning models
# ====================
# 1. Model Setup Parameters
# ====================
# Fixed parameters for the models (do not change)
num_channels = 3        # Number of input channels (seismic data)
dropout = 0.9           # Dropout rate to prevent overfitting

# ============================
# 2. Model Initialization
# ============================
# Initialize models with the number of classes, channels, and dropout rate
model_SeismicCNN_1d = SeismicCNN_1d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_SeismicCNN_2d = SeismicCNN_2d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_QuakeXNet_1d = QuakeXNet_1d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)
model_QuakeXNet_2d = QuakeXNet_2d(num_classes=4, num_channels=num_channels, dropout_rate=dropout).to(device)

# ============================
# 3. Load Pretrained Weights
# ============================
# Load the pretrained model state dictionaries from saved files
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

# Define model paths
model_dir = '../trained_deep_learning_models'
models_paths = {
    'SeismicCNN_1d': os.path.join(model_dir, 'best_model_SeismicCNN_1d.pth'),
    'SeismicCNN_2d': os.path.join(model_dir, 'best_model_SeismicCNN_2d.pth'),
    'QuakeXNet_1d': os.path.join(model_dir, 'best_model_QuakeXNet_1d.pth'),
    'QuakeXNet_2d': os.path.join(model_dir, 'best_model_QuakeXNet_2d.pth')
}

# Load model weights with error handling
try:
    saved_model_SeismicCNN_2d = torch.load(models_paths['SeismicCNN_2d'], map_location=device)
    saved_model_QuakeXNet_2d = torch.load(models_paths['QuakeXNet_2d'], map_location=device)
    saved_model_QuakeXNet_1d = torch.load(models_paths['QuakeXNet_1d'], map_location=device)
    saved_model_SeismicCNN_1d = torch.load(models_paths['SeismicCNN_1d'], map_location=device)
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    f.close()
    sys.exit(f"Model file not found: {e}")
except Exception as e:
    logger.error(f"Error loading model weights: {e}")
    f.close()
    sys.exit(f"Error loading model weights: {e}")

# ============================
# 4. Load Weights into Models
# ============================
# Load the state dictionaries into the corresponding models
model_SeismicCNN_1d.load_state_dict(saved_model_SeismicCNN_1d)
model_SeismicCNN_2d.load_state_dict(saved_model_SeismicCNN_2d)
model_QuakeXNet_1d.load_state_dict(saved_model_QuakeXNet_1d)
model_QuakeXNet_2d.load_state_dict(saved_model_QuakeXNet_2d)

# ============================
# 5. Set Models to Evaluation Mode
# ============================
# Move models to evaluation mode (important for layers like dropout and batch norm)
model_SeismicCNN_1d.eval()
model_SeismicCNN_2d.eval()
model_QuakeXNet_1d.eval()
model_QuakeXNet_2d.eval()

# ============================
# 6. Move Models to Correct Device
# ============================
# Ensure all models are on the correct device (GPU or CPU)
model_SeismicCNN_1d.to(device)
model_SeismicCNN_2d.to(device)
model_QuakeXNet_1d.to(device)
model_QuakeXNet_2d.to(device)

print("DONE LOADING.  Elapsed time: ",time()-Timport)

# Compute probabilities for different deep learning & 1 classical ML models

Tzero = time()
# 1D model: QuakeXNet_1d
stn_probs_QuakeXNet_1d, _, big_station_ids, st_all, snrs = process_model(
    model_QuakeXNet_1d, stations_id, dists_km, location, start_time, end_time,
    one_d=True, model_type='dl', filename='P_10_30_F_05_15_50', 
)
print("ELAPSED TIME (includes data download) 1d QuakeXNet: ",(time() - Tzero))
Tzero = time()

# 2D model: QuakeXNet_2d
stn_probs_QuakeXNet_2d, big_reshaped_data, big_station_ids, st_all, _  = process_model(
    model_QuakeXNet_2d, stations_id, dists_km, location, start_time, end_time,
    one_d=False, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 2d QuakeXNet: ",(time() - Tzero))
Tzero = time()
print('')

# 1D model: SeismicCNN_1d
stn_probs_SeismicCNN_1d, _, big_station_ids, st_all, _  = process_model(
    model_SeismicCNN_1d, stations_id, dists_km, location, start_time, end_time,
    one_d=True, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 1d SeismicCNN: ",(time() - Tzero))
Tzero = time()
print('')

# 2D model: SeismicCNN_2d
stn_probs_SeismicCNN_2d, _, big_station_ids, st_all, _ = process_model(
    model_SeismicCNN_2d, stations_id, dists_km, location, start_time, end_time,
    one_d=False, model_type='dl', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME 2d SeismicCNN: ",(time() - Tzero))
Tzero = time()
print('') 

# 40sec model (ML)
model = model_QuakeXNet_1d  # dummy name, not important according to Akash
stn_probs_ml_40, _, big_station_ids, st_all, _ = process_model(
    model, stations_id, dists_km, location, start_time, end_time,
    one_d=False, model_type='ml', filename='P_10_30_F_05_15_50',
    st_all=st_all
)
print("ELAPSED TIME ML40sec: ",(time() - Tzero))
Tzero = time()
print('')

def print_stats(probs, name, evid, snrs, analyst_class, mag):
    """
    Calculate and print statistics for the event classification probabilities.
    
    For the event classes EQ (earthquake), EX (explosion), and SU (surface event),
    calculate and print:
      - The overall mean probability of each event class.
      - The mean probability for each event class, computed over only those stations
        where the maximum probability for that station is greater than a threshold
        AND the probability distance (difference between the top two probabilities)
        is greater than a given threshold.
        
    Parameters:
    -----------
    probs : array-like
        Probability values that can be reshaped to (number_of_stations, n_windows, 4).
        The 4 classes are: [EQ, EX, NO, SU] (Earthquake, Explosion, Noise, Surface)
    name : str
        Base name of the model (e.g., "SeismicCNN_1d")
    evid : str
        Event ID being processed
    snrs : array-like
        Signal-to-noise ratios for each station
    analyst_class : str
        Analyst classification of the event (ground truth if available)
    mag : str or float
        Magnitude of the event
    """
    # Convert probabilities to a NumPy array and reshape.
    probs = np.array(probs)
    nstations = len(big_station_ids)  # Ensure big_station_ids is defined globally
    probs = probs.reshape(nstations, -1, 4)
    
    # Handle potential missing values for analyst_class and mag
    try:
        if len(analyst_class) == 2:
            pass
    except:
        analyst_class = 'na'
    try:
        if float(mag[2:]) > -6:
            pass
    except:
        mag = 'na'

    # Initialize lists to hold the maximum probability values per station.
    eqprobs, exprobs, suprobs, pdistances = [], [], [], []
    
    # Loop over stations and compute the maximum probability for each event class.
    for k in range(nstations):
        # For each station, assume:
        # index 0: EQ, index 1: EX, index 3: SU.
        eqprob = np.max(probs[k][:, 0])
        exprob = np.max(probs[k][:, 1])
        suprob = np.max(probs[k][:, 3])
        eqprobs.append(eqprob)
        exprobs.append(exprob)
        suprobs.append(suprob)
        
        # Compute probability distance: difference between the highest and second-highest
        pvals = [eqprob, exprob, suprob]
        sorted_pvals = sorted(pvals, reverse=True)
        pdistance = sorted_pvals[0] - sorted_pvals[1]
        pdistances.append(pdistance)
    
    # Convert lists to NumPy arrays for easier indexing.
    eqprobs = np.array(eqprobs)
    exprobs = np.array(exprobs)
    suprobs = np.array(suprobs)
    pdistances = np.array(pdistances)
    snrs = np.array(snrs)

    # Print overall means for each event class.
    overall_eq = np.mean(eqprobs)
    overall_ex = np.mean(exprobs)
    overall_su = np.mean(suprobs)
    overall_snr = np.mean(snrs)
    
    composite_name = f"{name}_mean_all"
    overall_means = [overall_eq, overall_ex, overall_su]
    sorted_means = sorted(overall_means, reverse=True)
    mean_pd = sorted_means[0] - sorted_means[1]
    event_classes = ['EQ', 'EX', 'SU']
    max_idx = np.argmax(overall_means)
    max_class = event_classes[max_idx]
    output_line = (f"{str(evid):<10s} {composite_name:<32s}    EQ: {overall_eq:5.3f}  {k+1:<3d}  "
        f"EX: {overall_ex:5.3f}  {k+1:<3d}  SU: {overall_su:5.3f}  {k+1:<3d}    ProbDist: {mean_pd:5.3f}    "
        f"Pred: {max_class}  {sorted_means[0]:5.3f}  {k+1:<3d}  Analyst: {analyst_class}  Mag: {mag}")
    f.write(output_line + "\n")  
 
    # Loop over the desired probability thresholds, probability distance thresholds, and SNR thresholds.
    for probthreshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
        for probdistance in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for snrthreshold in [0, 1, 2, 3, 4, 10]:
                # Create separate masks for each event class:
                eq_mask = (eqprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                ex_mask = (exprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                su_mask = (suprobs > probthreshold) & (pdistances > probdistance) & (snrs > snrthreshold)
                
                # Calculate the means for each event class using its own mask.
                mean_eq = np.mean(eqprobs[eq_mask]) if np.any(eq_mask) else 0.0
                mean_ex = np.mean(exprobs[ex_mask]) if np.any(ex_mask) else 0.0
                mean_su = np.mean(suprobs[su_mask]) if np.any(su_mask) else 0.0
                
                # Calculate counts for each event class.
                count_eq = np.sum(eq_mask)
                count_ex = np.sum(ex_mask)
                count_su = np.sum(su_mask)
                
                # Compute the probability distance based on the means:
                # This is the difference between the highest and second-highest mean probability.
                pvals_mean = [mean_eq, mean_ex, mean_su]
                sorted_means = sorted(pvals_mean, reverse=True)
                mean_pd = sorted_means[0] - sorted_means[1]
                
                # Determine the event class with the highest mean.
                event_classes = ['EQ', 'EX', 'SU']
                max_idx = np.argmax(pvals_mean)
                max_class = event_classes[max_idx]
                counts = [count_eq, count_ex, count_su]
                pred_count = counts[max_idx]
                if sorted_means[0] < 0.01 or pred_count == 0:
                    max_class = 'NO'
                
                # Create the composite name including SNR threshold and left-justify in a 25-character field.
                composite_name = f"{name}_p{probthreshold:.2f}_d{probdistance:.2f}_snr{snrthreshold:02d}"
                
                # Print the event id, composite name, the means, counts, probability distance, and predicted class.
                output_line = (f"{str(evid):<10s} {composite_name:<32s}    EQ: {mean_eq:5.3f}  {count_eq:<3d}  "
                    f"EX: {mean_ex:5.3f}  {count_ex:<3d}  SU: {mean_su:5.3f}  {count_su:<3d}    "
                    f"ProbDist: {mean_pd:5.3f}    Pred: {max_class}  {sorted_means[0]:5.3f}  {pred_count:<3d}  "
                    f"Analyst: {analyst_class}  Mag: {mag}")
                f.write(output_line + "\n")


def process_model_probs(model_probs, model_name, evid, snrs, big_station_ids, f, analyst_class=None, mag=None):
    """
    Process probability data for a seismic model and write results to output.
    
    Parameters:
    -----------
    model_probs : array-like
        Probability values from the model
    model_name : str
        Name of the model (display name for output)
    evid : str
        Event ID
    snrs : array-like
        Signal-to-noise ratios
    big_station_ids : list
        Station identifiers
    f : file object
        File to write results to
    analyst_class : optional
        Analyst classification
    mag : optional
        Magnitude information
    """
    # Skip processing if model_probs is empty (indicates error in processing)
    if len(model_probs) == 0:
        logger.warning(f"No probability data available for model {model_name}")
        print(f"WARNING: No probability data available for model {model_name}")
        return
        
    try:
        prob_stns = np.array(model_probs)
        prob_stns = prob_stns.reshape(len(big_station_ids), -1, 4)
        
        for k in range(len(prob_stns)):
            eqprob = np.max(prob_stns[k][:,0])
            exprob = np.max(prob_stns[k][:,1])
            noprob = np.max(prob_stns[k][:,2])
            suprob = np.max(prob_stns[k][:,3])

            pvals = [eqprob, exprob, suprob]
            sorted_pvals = sorted(pvals, reverse=True)
            pdistance = sorted_pvals[0] - sorted_pvals[1]  # Calculated but not used
            
            output_line = f"PROBS: {evid} {model_name:<15s} {k:2d} {eqprob:.7f} {exprob:.7f} {noprob:.7f} {suprob:.7f} {snrs[k]:7.2f}  {big_station_ids[k]} "
            print(output_line)
            #for ij in range(0,len(prob_stns[k][:,0])):
            #    print(ij, prob_stns[k][ij,0], prob_stns[k][ij,1], prob_stns[k][ij,2], prob_stns[k][ij,3])

            f.write(output_line + "\n")
        
        print_stats(model_probs, model_name, evid, snrs, analyst_class, mag)
        print('')
    except Exception as e:
        logger.error(f"Error processing model {model_name} probabilities: {e}")
        print(f"ERROR processing model {model_name}: {e}")


# Define models and their display names
models_and_names = [
    (stn_probs_SeismicCNN_1d, 'SeismicCNN_1d'),
    (stn_probs_SeismicCNN_2d, 'SeismicCNN_2d'),
    (stn_probs_QuakeXNet_1d, 'QuakeXNet_1d'),
    (stn_probs_QuakeXNet_2d, 'QuakeXNet_2d'),
    (stn_probs_ml_40, 'ML40sec')
]

# Process each model's results and print/write output
for model_probs, model_name in models_and_names:
    process_model_probs(model_probs, model_name, evid, snrs, big_station_ids, f, analyst_class, mag)

# Close the output file
f.close()

# Print total elapsed time
total_time = time() - Timport
print("TOTAL ELAPSED TIME: ", total_time)
logger.info(f"Processing completed for event {evid}, total time: {total_time:.2f} seconds")

