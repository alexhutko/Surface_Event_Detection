#!/home/ahutko/miniconda3/envs/surface_dl/bin/python
#  Fix the issue of old way used predicted event totals rather than true/label events for total count.

"""
Seismic Event Classification Analysis with Magnitude Distribution

This script analyzes seismic event classification results from multiple models,
evaluates their performance, and analyzes the relationship between event magnitude
and prediction accuracy.

Features:
- Identifies the best parameter configurations for each model and event type
- Calculates performance metrics (precision, recall, F1, accuracy)
- Creates magnitude-based histograms for correct vs. incorrect predictions
- Generates probability distance histograms and scatter plots
- Creates trace-wise and event-wise analysis visualizations
- Generates confusion matrices and performance metric visualizations
- Writes detailed performance results to output files
- Handles empty files and detects/stops at repeated lines

Usage:
  python seismic_analysis.py

Inputs:
  - RESULTS/*ou*.txt files: Model prediction results
  - earthquake_data.txt: Contains event magnitudes and analyst classifications

Outputs:
  - Text reports of model performance
  - Visualizations of model performance metrics
  - Magnitude histograms for correct/incorrect predictions by model
  - Probability distance histograms and scatter plots
  - Trace-wise analysis visualizations

Functions to create advanced visualizations combining SNR, distance_km, and probability data.
Includes multiple visualization types:
1. Small multiples grid (model × event type)
2. Heatmap with bubble size
3. Contour plot overlay
4. Binned statistics plot
5. SNR histograms by channel type
6. Classification report heatmap
7. Updated probability vs SNR plots with correct trace counting
8. Distance vs probability/probability distance plots
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle, Patch
from matplotlib.cm import ScalarMappable

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------
RESULTS_PATTERN = 'RESULTS/6*ou*.txt'  # Pattern for model output files
OUTPUT_DIR = 'output'  # Directory for output files
BIN_WIDTH = 0.2  # Width of histogram bins
COLORS = {'eq': 'blue', 'ex': 'red', 'su': 'green'}  # Colors for each event type
OUTPUT_DPI = 300  # Resolution of output images
MIN_MAGNITUDE = -2.0  # Cap for extremely negative magnitudes
SNR_SATURATE = 20

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
SNR_SATURATE = 20.0  # Cap SNR values at this level
DISTKM_SATURATE = 150.0  # Cap distance values at this level (km)
DISTANCE_BIN_WIDTH = 2.0  # Width of distance bins (km)
PROB_BIN_WIDTH = 0.025  # Width of probability bins
SNR_BIN_WIDTH = 0.25  # Width of SNR bins
DISTANCE_FILE = "evid_netsta_distances.txt"  # Filename for distance data

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def extract_netsta(channel_list):
    """
    Extract network.station from a list of NSLC identifiers.
    
    Args:
        channel_list: List of strings in NET.STA.LOC.CHA format
    
    Returns:
        String with 'NET.STA' or None if not available
    """
    if not channel_list or len(channel_list) == 0:
        return None
    
    # Take the first channel in the list
    channel = channel_list[0]
    
    # Extract NET.STA part
    parts = channel.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    
    return None

def count_unique_stations(traces):
    """
    Count unique network.station pairs in trace data.
    
    Args:
        traces: List of trace data tuples
    
    Returns:
        Number of unique network.station pairs
    """
    unique_stations = set()
    
    for trace_data in traces:
        if len(trace_data) > 2 and isinstance(trace_data[2], list):
            netsta = extract_netsta(trace_data[2])
            if netsta:
                unique_stations.add(netsta)
    
    return len(unique_stations)

def load_distances(filename=DISTANCE_FILE):
    """
    Load station distances from file.
    
    Args:
        filename: Path to distance file
    
    Returns:
        Dictionary mapping (evid, netsta) to distance
    """
    distances = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    evid = int(parts[0])
                    netsta = parts[1]
                    distance_km = float(parts[2])
                    distances[(evid, netsta)] = min(distance_km, DISTKM_SATURATE)
    except FileNotFoundError:
        print(f"Warning: Distance file '{filename}' not found.")
    
    return distances

def add_colorbar(fig, im, label, position=None):
    """
    Add a colorbar to a figure.
    
    Args:
        fig: Figure object
        im: Image object
        label: Label for colorbar
        position: Optional position [left, bottom, width, height]
    
    Returns:
        Colorbar object
    """
    if position is None:
        position = [0.92, 0.15, 0.02, 0.7]
    
    cbar_ax = fig.add_axes(position)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label)
    return cbar

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def init_nested_dict():
    """Helper function to create nested defaultdicts."""
    return defaultdict(int)

def ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def calculate_prob_distance(probabilities):
    """
    Calculate the probability distance (difference between highest and second highest).
    
    Args:
        probabilities: List of probability values
    
    Returns:
        Probability distance
    """
    if len(probabilities) < 2:
        return 0.0
    
    sorted_probs = sorted(probabilities, reverse=True)
    return sorted_probs[0] - sorted_probs[1]

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
# Model evaluation data structures
correct_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # model -> params -> event_type -> count
total_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))    # model -> params -> event_type -> count
overall_correct = defaultdict(lambda: defaultdict(int))                           # model -> params -> count
overall_total = defaultdict(lambda: defaultdict(int))                             # model -> params -> count
paramcount = defaultdict(lambda: defaultdict(int))                                # model -> params -> count
confusion_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))  # model -> params -> true -> pred -> count
seismogram_counts = defaultdict(lambda: defaultdict(list))                        # model -> params -> [counts]

# Magnitude data structures
model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [magnitudes]

# Probability distance data structures (event-wise)
event_probdist_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [prob_dist]
event_scatter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [(prob_dist, magnitude, trace_count)]

# Trace-wise data structures
trace_results = defaultdict(lambda: defaultdict(list))  # model -> event_type -> [probabilities]
trace_probdist_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [prob_dist]
trace_scatter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [(prob_dist, magnitude)]

# Add a new data structure to track trace predictions by predicted class
trace_predictions = {
    'strong_motion': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    '4_channel': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    'short_period_3c': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    'short_period_vertical': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    'broadband': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    'all': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
}

# Mean_all results for reference
mean_all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # model -> event_type -> metric -> value

# List of event types for iteration
event_types = ['eq', 'ex', 'su']

# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

def process_model_outputs():
    """
    Process all model output files.
    
    This function processes all output files, but doesn't build the histogram data yet.
    That will be done after we identify the single best parameter set for each model.
    
    Modified to properly extract SNR from PROBS line.
    """
    print(f"Processing model output files from {RESULTS_PATTERN}...")
    
    file_count = 0
    empty_count = 0
    repetition_count = 0
    
    # Save raw events data for later histogram creation
    raw_event_data = {}  # {evid: {"model": model, "params": params, "pred": pred, "analyst": analyst, "magnitude": mag, "prob_dist": prob_dist, "trace_count": trace_count}}
    
    # Save trace-level data
    raw_trace_data = {}  # {evid: {model: [(trace_index, [eq_prob, ex_prob, no_prob, su_prob], channels, snr)]}}
    
    for outfile in glob.glob(RESULTS_PATTERN):
        file_count += 1
        if file_count % 100 == 0:
            print(f"Processing file {file_count}...")
        
        # Read all lines from the file
        with open(outfile, 'r') as f:
            lines = f.readlines()
        
        # Skip empty files
        if not lines:
            empty_count += 1
            continue
        
        # Track processed lines to detect repetitions
        processed_lines = set()
        
        # First pass: collect trace-level data from PROBS lines
        current_evid = None
        
        for line in lines:
            # Extract event ID from ORDATE line if present
            if line.startswith("ORDATE START END:"):
                parts = line.split()
                if len(parts) >= 3:
                    current_evid = int(parts[3])
                    
            # Process PROBS lines
            if line.startswith("PROBS:") and current_evid is not None:
                try:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    evid = int(parts[1])
                    model_name = parts[2]  # Model name without parameters
                    trace_index = int(parts[3])
                    
                    # Extract model name properly, preserving dimensionality
                    name_parts = model_name.split('_')
                    if len(name_parts) > 1 and (name_parts[1] == '1d' or name_parts[1] == '2d'):
                        model = f"{name_parts[0]}_{name_parts[1]}"  # e.g. "SeismicCNN_1d"
                    else:
                        model = name_parts[0]  # For models without dimensionality like ML40sec
                    
                    # Extract probabilities (eq, ex, no, su)
                    eq_prob = float(parts[4])
                    ex_prob = float(parts[5])
                    no_prob = float(parts[6])
                    su_prob = float(parts[7])
                    
                    # Extract SNR value (9th column)
                    snr = min(SNR_SATURATE,float(parts[8]))
                    
                    # Extract channel information
                    channels = []
                    try:
                        # Find the portion of the line with channel info
                        channel_str = ' '.join(parts[9:])
                        # Extract the content inside the double brackets
                        channels_match = re.search(r"\[\['(.+?)'\]\]", channel_str)
                        if channels_match:
                            # Split the channels by ', '
                            channels = channels_match.group(1).split("', '")
                    except Exception:
                        # If any error occurs, just use empty list
                        channels = []
                    
                    # Initialize trace data for this event and model if not exists
                    if evid not in raw_trace_data:
                        raw_trace_data[evid] = {}
                    
                    if model not in raw_trace_data[evid]:
                        raw_trace_data[evid][model] = []
                    
                    # Add trace data including SNR
                    raw_trace_data[evid][model].append((trace_index, [eq_prob, ex_prob, no_prob, su_prob], channels, snr))
                    
                except (IndexError, ValueError) as e:
                    # Skip lines that don't match expected format
                    continue
        
        # Second pass: process event prediction lines
        for line in lines:
            # Skip lines that don't contain "Analyst:"
            if "Analyst:" not in line:
                continue
                
            # Check for repeated lines
            if line in processed_lines:
                repetition_count += 1
                break  # Stop processing this file when repetition detected
            
            # Add line to processed set
            processed_lines.add(line)
            
            try:
                parts = line.split()
                if len(parts) < 21:
                    continue
                    
                evid = int(parts[0])
                name = parts[1]
                
                # Extract model name properly, preserving dimensionality
                name_parts = name.split('_')
                if len(name_parts) > 1 and (name_parts[1] == '1d' or name_parts[1] == '2d'):
                    model = f"{name_parts[0]}_{name_parts[1]}"  # e.g. "SeismicCNN_1d"
                else:
                    model = name_parts[0]  # For models without dimensionality like ML40sec

                # Extracting params
                params = re.split(r'sec_|1d_|2d_', name)[1] if len(re.split(r'sec_|1d_|2d_', name)) > 1 else ""

                # Extract the predicted class (index 14) and convert to lowercase
                pred = parts[14].lower()
                
                # Extract trace count (index 16)
                trace_count = int(parts[16])
                
                # Extract seismogram count (index 16)
                seismogramcount = trace_count
                
                # Extract analyst label (index 18) and convert px to ex
                analyst = parts[18].lower()
                if analyst == 'px':
                    analyst = 'ex'

                # Extract magnitude
                magstr = parts[20]
                magnitude = float(magstr[2:])
                # Cap extremely negative magnitudes at MIN_MAGNITUDE
                if magnitude < MIN_MAGNITUDE:
                    magnitude = MIN_MAGNITUDE
                
                # Extract probability distance
                prob_dist = float(parts[12])
                
                # Store seismogram count
                seismogram_counts[model][params].append(seismogramcount)
                
                # Update paramcount (total evaluations for this model+params)
                paramcount[model][params] += 1
                
                # Save the raw event data for later histogram creation
                if evid not in raw_event_data:
                    raw_event_data[evid] = []
                raw_event_data[evid].append({
                    "model": model,
                    "params": params,
                    "pred": pred,
                    "analyst": analyst,
                    "magnitude": magnitude,
                    "prob_dist": prob_dist,
                    "trace_count": trace_count
                })
                
                # Skip if prediction is "no" (noise) as it's always wrong per instructions
                if pred == "no":
                    # Still count it as a prediction for the true class
                    total_predictions[model][params][analyst] += 1
                    overall_total[model][params] += 1
                    # Update confusion matrix
                    confusion_matrices[model][params][analyst][pred] += 1
                    continue
                
                # Update prediction counts
                total_predictions[model][params][pred] += 1
                overall_total[model][params] += 1
                
                # Update correct prediction counts if prediction matches analyst
                if pred == analyst:
                    correct_predictions[model][params][pred] += 1
                    overall_correct[model][params] += 1
                
                # Update confusion matrix
                confusion_matrices[model][params][analyst][pred] += 1
                
                # Save mean_all results for reference
                if params == "mean_all":
                    correct = overall_correct[model][params]
                    total = overall_total[model][params]
                    accuracy = correct / total if total > 0 else 0
                    
                    # Store accuracy for this model and parameters
                    mean_all_results[model]["overall"]["accuracy"] = accuracy
                    
                    # Store event type specific metrics
                    for event_type in event_types:
                        event_correct = correct_predictions[model][params][event_type]
                        event_total = total_predictions[model][params][event_type]
                        event_accuracy = event_correct / event_total if event_total > 0 else 0
                        mean_all_results[model][event_type]["accuracy"] = event_accuracy
                        mean_all_results[model][event_type]["correct"] = event_correct
                        mean_all_results[model][event_type]["total"] = event_total
                
            except (IndexError, ValueError) as e:
                # Skip lines that don't match expected format
                continue

    print(f"Processed {file_count} files.")
    print(f"  - Skipped {empty_count} empty files")
    print(f"  - Detected and handled {repetition_count} files with repeated lines")
    
    return raw_event_data, raw_trace_data

# -----------------------------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------------------------
def find_best_parameters():
    """
    Find the best parameters for each model and event type using
    the confusion matrix for accurate event counts.
    
    Returns:
    - Dictionary of best parameters by model and event type
    - Dictionary of best overall parameters by model
    """
    # Find best params for each model and event type
    best_params_by_event = {}
    
    for model in confusion_matrices:
        best_params_by_event[model] = {}
        
        for event_type in event_types:
            best_params = []
            best_accuracy = 0.0
            
            for params in confusion_matrices[model]:
                # Skip if this event type doesn't exist in the confusion matrix
                if event_type not in confusion_matrices[model][params]:
                    continue
                
                # Calculate true positives (correctly predicted this class)
                true_positives = confusion_matrices[model][params][event_type][event_type] if event_type in confusion_matrices[model][params][event_type] else 0
                
                # Calculate total true events for this class (sum of all predictions for this true class)
                true_total = sum(confusion_matrices[model][params][event_type].values())
                
                # Skip if no events of this type
                if true_total == 0:
                    continue
                
                # Calculate accuracy for this event type (recall)
                accuracy = true_positives / true_total
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = [params]
                elif accuracy == best_accuracy:
                    best_params.append(params)
            
            best_params_by_event[model][event_type] = best_params

    # Find best overall params for each model
    best_overall_params = {}
    
    for model in confusion_matrices:
        best_params = []
        best_accuracy = 0.0
        
        for params in confusion_matrices[model]:
            # Calculate total correct predictions (sum of diagonal elements in confusion matrix)
            total_correct = sum(
                confusion_matrices[model][params][et][et] 
                for et in event_types 
                if et in confusion_matrices[model][params] and et in confusion_matrices[model][params][et]
            )
            
            # Calculate total events (sum of all elements in confusion matrix)
            total_events = sum(
                sum(pred_counts.values()) 
                for true_class, pred_counts in confusion_matrices[model][params].items()
            )
            
            # Skip if no events
            if total_events == 0:
                continue
            
            # Calculate overall accuracy
            accuracy = total_correct / total_events
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [params]
            elif accuracy == best_accuracy:
                best_params.append(params)
        
        # Break ties using median seismogram count
        if len(best_params) > 1:
            best_median = 0
            chosen_param = best_params[0]
            
            for param in best_params:
                if len(seismogram_counts[model][param]) > 0:
                    median_count = np.median(seismogram_counts[model][param])
                    if median_count > best_median:
                        best_median = median_count
                        chosen_param = param
            
            best_overall_params[model] = chosen_param
        else:
            best_overall_params[model] = best_params[0] if best_params else None

    return best_params_by_event, best_overall_params

def calculate_metrics(best_overall_params):
   """
   Calculate performance metrics for each model's best parameters.
   
   Returns dictionary of metrics by model.
   """
   metrics = {}
   for model in best_overall_params:
       if best_overall_params[model]:
           params = best_overall_params[model]
           
           # Calculate correct predictions (sum of true positives)
           correct = sum(confusion_matrices[model][params][et][et] 
                        for et in event_types 
                        if et in confusion_matrices[model][params] 
                        and et in confusion_matrices[model][params][et])
           
           # Calculate total events (sum of all confusion matrix entries)
           total = sum(sum(pred_counts.values()) 
                      for analyst, pred_counts in confusion_matrices[model][params].items())
           
           metrics[model] = {
               'params': params,
               'correct': correct,
               'total': total,
               'accuracy': correct / total if total > 0 else 0,
               'confusion_matrix': confusion_matrices[model][params],
               'mean_seismogram_count': np.mean(seismogram_counts[model][params]) if seismogram_counts[model][params] else 0,
               'median_seismogram_count': np.median(seismogram_counts[model][params]) if seismogram_counts[model][params] else 0
           }

           # Calculate precision, recall, F1 for each class
           precision = {}
           recall = {}
           f1 = {}
           
           cm = confusion_matrices[model][params]
           for event_type in event_types:
               # True positives: Predicted this class correctly
               tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
               
               # False positives: Predicted this class but was wrong
               fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                        for true in cm if true != event_type)
               
               # False negatives: Should have predicted this class but didn't
               fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                        for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                        if pred != event_type)
               
               # Calculate metrics
               if tp + fp > 0:
                   precision[event_type] = tp / (tp + fp)
               else:
                   precision[event_type] = 0
                   
               if tp + fn > 0:
                   recall[event_type] = tp / (tp + fn)
               else:
                   recall[event_type] = 0
                   
               if precision[event_type] + recall[event_type] > 0:
                   f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
               else:
                   f1[event_type] = 0
           
           # Calculate macro average metrics
           metrics[model]['precision'] = precision
           metrics[model]['recall'] = recall
           metrics[model]['f1'] = f1
           metrics[model]['macro_precision'] = sum(precision.values()) / len(precision) if precision else 0
           metrics[model]['macro_recall'] = sum(recall.values()) / len(recall) if recall else 0
           metrics[model]['macro_f1'] = sum(f1.values()) / len(f1) if f1 else 0

   return metrics

# -----------------------------------------------------------------------------
# Output and Visualization Functions
# -----------------------------------------------------------------------------
"""
Fixed version of the print_and_write_results function that counts unique events properly.
"""

def print_and_write_results_old(best_params_by_event, metrics, raw_event_data, output_file="model_results.txt"):
    """
    Print results to console and write to file.
    Uses corrected event counting that only counts each unique event once.
    
    Args:
        best_params_by_event: Dictionary of best parameters by model and event type
        metrics: Dictionary of metrics by model
        raw_event_data: Dictionary of raw event data used for counting
        output_file: Name of output file for results
    """

    print('')
    print("best_params_by_event: ", type(best_params_by_event), len(best_params_by_event) )
    print(best_params_by_event)
    print('metrics: ', type(metrics), len(metrics) )
    print(metrics)
    print('raw_event_data: ',type(raw_event_data), len(raw_event_data) )
    key = next(iter(raw_event_data))
    print('key: ',key)
    print(len(raw_event_data[key]), raw_event_data[key][0])
    print('')

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    # Calculate total event counts by event type - counting each unique event ID only once
    event_class_totals = defaultdict(int)
    counted_events = set()
    
    # Count events directly from raw data
    for evid, event_entries in raw_event_data.items():
        if event_entries and evid not in counted_events:
            counted_events.add(evid)
            analyst = event_entries[0]["analyst"]
            event_class_totals[analyst] += 1
    
    # Model-specific event counts (still counting unique events only)
    model_event_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Count events for each model/params combination
    for model in confusion_matrices:
        for params in confusion_matrices[model]:
            model_counted_events = set()  # Track events per model+params
            
            for evid, event_entries in raw_event_data.items():
                # Skip if already processed for this model+params
                if evid in model_counted_events:
                    continue
                
                # Check if this event was evaluated with this model+params
                found = False
                for entry in event_entries:
                    if entry["model"] == model and entry["params"] == params:
                        found = True
                        model_counted_events.add(evid)
                        analyst = entry["analyst"]
                        model_event_counts[model][params][analyst] += 1
                        break

    #print("DONE w profile section")
    #profiler.disable()
    #profiler.dump_stats("print_and_write.prof")
    #print("Wrote profiling stats to print_and_write.prof")

    # Open file for writing
    with open(output_path, 'w') as f:
        # Function to write both to console and file
        def write_line(line):
            print(line)
            f.write(line + '\n')
        
        # Print total events across all models and parameters
        total_all_events = sum(event_class_totals.values())
        write_line(f"\n=== Total Events: {total_all_events} ===")
        for event_type in event_types:
            write_line(f"Event type: {event_type.upper()}: {event_class_totals[event_type]} total events")
        write_line("")
            
        # Print results by model and event type
        write_line("\n=== Best Parameters by Model and Event Type ===")
        for model in sorted(best_params_by_event.keys()):
            for event_type in event_types:
                # First include mean_all as reference
                if model in mean_all_results and event_type in mean_all_results[model]:
                    correct = mean_all_results[model][event_type]["correct"]
                    # Use true total for this event class, not predictions
                    total = model_event_counts[model]["mean_all"][event_type]
                    percent = (correct / total * 100) if total > 0 else 0
                    write_line(f"Model: {model}  event type: {event_type}   params: mean_all  "
                              f"correct predictions: {correct}  total events: {total}  "
                              f"percent correct: {percent:.1f}%. (REFERENCE)")
                
                # Then show the best parameters
                if event_type in best_params_by_event[model] and best_params_by_event[model][event_type]:
                    for params in best_params_by_event[model][event_type]:
                        if params == "mean_all":
                            continue  # Skip mean_all here as it's already shown
                        
                        # Calculate true positives from confusion matrix
                        true_pos = confusion_matrices[model][params][event_type][event_type] if event_type in confusion_matrices[model][params] and event_type in confusion_matrices[model][params][event_type] else 0
                        
                        # Calculate total true events for this class (using analyst label)
                        total = model_event_counts[model][params][event_type]
                        
                        percent = (true_pos / total * 100) if total > 0 else 0
                        write_line(f"Model: {model}  event type: {event_type}   params: {params}  "
                                  f"correct predictions: {true_pos}  total events: {total}  "
                                  f"percent correct: {percent:.1f}%.")
            write_line("")

        # Print best overall params for each model
        write_line("\n=== Best Overall Parameters for Each Model ===")
        for model in sorted(metrics.keys()):
            # First include mean_all as reference if available
            if model in mean_all_results and "overall" in mean_all_results[model]:
                # Get total correct predictions (all true positives)
                correct = sum(confusion_matrices[model]["mean_all"][et][et] 
                             for et in event_types 
                             if et in confusion_matrices[model]["mean_all"] 
                             and et in confusion_matrices[model]["mean_all"][et])
                
                # Get total events (sum of all confusion matrix entries for this model+params)
                total = sum(sum(pred_counts.values()) 
                           for analyst, pred_counts in confusion_matrices[model]["mean_all"].items())
                
                percent = (correct / total * 100) if total > 0 else 0
                write_line(f"Model: {model}  params: mean_all  "
                          f"correct predictions: {correct}  total events: {total}  "
                          f"percent correct: {percent:.1f}%. (REFERENCE)")
            
            # Then show the best parameters
            m = metrics[model]
            write_line(f"Model: {model}  params: {m['params']}  "
                      f"correct predictions: {m['correct']}  total events: {m['total']}  "
                      f"percent correct: {m['accuracy']*100:.1f}%.")
            write_line(f"  Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                      f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}")
            write_line(f"  Mean seismogram count: {m['mean_seismogram_count']:.1f}, "
                      f"Median seismogram count: {m['median_seismogram_count']:.1f}")
            
            # Print per-class metrics with total true event counts for each class
            for event_type in event_types:
                if event_type in m['precision']:
                    # Get true event count for this model, parameter set, and event type
                    events_count = model_event_counts[model][m['params']][event_type]
                    write_line(f"  - {event_type.upper()}: Precision: {m['precision'][event_type]:.3f}, "
                              f"Recall: {m['recall'][event_type]:.3f}, F1: {m['f1'][event_type]:.3f}, "
                              f"Total events: {events_count}")
            write_line("")

    print(f"Results written to {output_path}")

def print_and_write_results_chatgpt(best_params_by_event, metrics, raw_event_data,
                            output_file="model_results.txt"):
    """
    Print results to console and write to file.
    Uses corrected event counting that only counts each unique event once.
    """
    output_path = os.path.join(OUTPUT_DIR, output_file)

    # 1) Total event counts by class (still just 252 items → fast)
    event_class_totals = {
        et: sum(1 for entries in raw_event_data.values()
                if entries and entries[0]["analyst"] == et)
        for et in event_types
    }

    # 2) Precompute model+params → event counts from confusion_matrices (very fast)
    model_event_counts = {
        model: {
            params: {
                et: sum(confusion_matrices[model][params].get(et, {}).values())
                for et in event_types
            }
            for params in confusion_matrices[model]
        }
        for model in confusion_matrices
    }

    with open(output_path, 'w') as f:
        def write_line(line):
            print(line)
            f.write(line + "\n")

        # Total events
        total_all_events = sum(event_class_totals.values())
        write_line(f"\n=== Total Events: {total_all_events} ===")
        for et in event_types:
            write_line(f"Event type: {et.upper()}: {event_class_totals[et]} total events")
        write_line("")

        # Best by model/event type
        write_line("\n=== Best Parameters by Model and Event Type ===")
        for model in sorted(best_params_by_event):
            # reference “mean_all”
            if model in mean_all_results:
                for et in event_types:
                    if et in mean_all_results[model]:
                        corr = mean_all_results[model][et]["correct"]
                        tot  = model_event_counts[model]["mean_all"][et]
                        pct  = corr / tot * 100 if tot else 0
                        write_line(
                            f"Model: {model}  event type: {et}  params: mean_all  "
                            f"correct: {corr}  total: {tot}  {pct:.1f}%. (REF)")
            # then your best_params
            for et in event_types:
                for params in best_params_by_event[model].get(et, []):
                    if params == "mean_all":
                        continue
                    tp    = confusion_matrices[model][params].get(et, {}).get(et, 0)
                    tot   = model_event_counts[model][params][et]
                    pct   = tp / tot * 100 if tot else 0
                    write_line(
                        f"Model: {model}  event type: {et}  params: {params}  "
                        f"correct: {tp}  total: {tot}  {pct:.1f}%.")
            write_line("")

        # Best overall per model
        write_line("\n=== Best Overall Parameters for Each Model ===")
        for model in sorted(metrics):
            # reference
            if model in mean_all_results.get(model, {}):
                corr = sum(
                    confusion_matrices[model]["mean_all"][et].get(et, 0)
                    for et in event_types
                )
                tot = sum(model_event_counts[model]["mean_all"].values())
                pct = corr / tot * 100 if tot else 0
                write_line(
                    f"Model: {model}  params: mean_all  correct: {corr}  "
                    f"total: {tot}  {pct:.1f}%. (REF)")

            m = metrics[model]
            write_line(
                f"Model: {model}  params: {m['params']}  "
                f"correct: {m['correct']}  total: {m['total']}  "
                f"pct: {m['accuracy']*100:.1f}%.")
            write_line(
                f"  Precision: {m['macro_precision']:.3f}, "
                f"Recall: {m['macro_recall']:.3f}, "
                f"F1: {m['macro_f1']:.3f}, "
                f"Accuracy: {m['accuracy']:.3f}, "
                f"Total events: {m['total']}")
            write_line(
                f"  Mean traces: {m['mean_seismogram_count']:.1f}, "
                f"Median traces: {m['median_seismogram_count']:.1f}")
            for et in event_types:
                if et in m['precision']:
                    cnt = model_event_counts[model][m['params']][et]
                    write_line(
                        f"  - {et.upper()}: Precision: {m['precision'][et]:.3f}, "
                        f"Recall: {m['recall'][et]:.3f}, "
                        f"F1: {m['f1'][et]:.3f}, "
                        f"Total events: {cnt}")
            write_line("")

    print(f"Results written to {output_path}")


def print_and_write_results_claude(best_params_by_event, metrics, raw_event_data, output_file="model_results_claude.txt"):
    """
    Print results to console and write to file.
    Uses corrected event counting that only counts each unique event once.
    
    Args:
        best_params_by_event: Dictionary of best parameters by model and event type
        metrics: Dictionary of metrics by model
        raw_event_data: Dictionary of raw event data used for counting
        output_file: Name of output file for results
    """
    # Remove debug printing which slows down processing
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    # Pre-compute event counts once, instead of recalculating 
    # Calculate total event counts by event type - counting each unique event ID only once
    event_class_totals = defaultdict(int)
    counted_events = set()
    
    # Create a mapping of events to analysts to avoid repeated lookups
    event_to_analyst = {}
    
    # First pass: build event_to_analyst mapping
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            event_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Now count unique events
    for evid, analyst in event_to_analyst.items():
        if evid not in counted_events:
            counted_events.add(evid)
            event_class_totals[analyst] += 1
    
    # Pre-compute model event counts once
    model_event_counts = {}
    
    # Process each model only once
    for model in confusion_matrices:
        model_event_counts[model] = {}
        
        for params in confusion_matrices[model]:
            # Create a mapping for this model and parameter set
            model_params_analysts = {}
            model_counted_events = set()
            
            # Find all events evaluated with this model+params combo
            for evid, event_entries in raw_event_data.items():
                for entry in event_entries:
                    if entry["model"] == model and entry["params"] == params:
                        if evid not in model_counted_events:
                            model_counted_events.add(evid)
                            analyst = event_to_analyst[evid]
                            if params not in model_event_counts[model]:
                                model_event_counts[model][params] = defaultdict(int)
                            model_event_counts[model][params][analyst] += 1
                        break  # No need to check other entries for this event
    
    # Open file for writing and use string concatenation to build output
    with open(output_path, 'w') as f:
        # Build string in memory and write once
        output_lines = []
        
        # Function to add line to both output and console
        def add_line(line):
            print(line)
            output_lines.append(line)
        
        # Print total events across all models and parameters
        total_all_events = sum(event_class_totals.values())
        add_line(f"\n=== Total Events: {total_all_events} ===")
        for event_type in event_types:
            add_line(f"Event type: {event_type.upper()}: {event_class_totals[event_type]} total events")
        add_line("")
        
        # Print results by model and event type
        add_line("\n=== Best Parameters by Model and Event Type ===")
        for model in sorted(best_params_by_event.keys()):
            for event_type in event_types:
                # First include mean_all as reference
                if model in mean_all_results and event_type in mean_all_results[model]:
                    correct = mean_all_results[model][event_type]["correct"]
                    # Use pre-computed counts
                    total = model_event_counts[model]["mean_all"].get(event_type, 0)
                    percent = (correct / total * 100) if total > 0 else 0
                    add_line(f"Model: {model}  event type: {event_type}   params: mean_all  "
                              f"correct predictions: {correct}  total events: {total}  "
                              f"percent correct: {percent:.1f}%. (REFERENCE)")
                
                # Then show the best parameters
                if event_type in best_params_by_event[model] and best_params_by_event[model][event_type]:
                    for params in best_params_by_event[model][event_type]:
                        if params == "mean_all":
                            continue  # Skip mean_all here as it's already shown
                        
                        # Use pre-computed matrix values
                        true_pos = confusion_matrices[model][params].get(event_type, {}).get(event_type, 0)
                        
                        # Use pre-computed counts
                        total = model_event_counts[model][params].get(event_type, 0)
                        
                        percent = (true_pos / total * 100) if total > 0 else 0
                        add_line(f"Model: {model}  event type: {event_type}   params: {params}  "
                                  f"correct predictions: {true_pos}  total events: {total}  "
                                  f"percent correct: {percent:.1f}%.")
            add_line("")
            
        # Print best overall params for each model
        add_line("\n=== Best Overall Parameters for Each Model ===")
        for model in sorted(metrics.keys()):
            # First include mean_all as reference if available
            if model in mean_all_results and "overall" in mean_all_results[model]:
                # Pre-compute sums once
                correct = sum(confusion_matrices[model]["mean_all"].get(et, {}).get(et, 0)
                             for et in event_types)
                
                total = sum(sum(pred_counts.values())
                           for analyst, pred_counts in confusion_matrices[model]["mean_all"].items())
                
                percent = (correct / total * 100) if total > 0 else 0
                add_line(f"Model: {model}  params: mean_all  "
                          f"correct predictions: {correct}  total events: {total}  "
                          f"percent correct: {percent:.1f}%. (REFERENCE)")
            
            # Then show the best parameters - these are pre-computed in metrics
            m = metrics[model]
            add_line(f"Model: {model}  params: {m['params']}  "
                      f"correct predictions: {m['correct']}  total events: {m['total']}  "
                      f"percent correct: {m['accuracy']*100:.1f}%.")
            add_line(f"  Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                      f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}")
            add_line(f"  Mean seismogram count: {m['mean_seismogram_count']:.1f}, "
                      f"Median seismogram count: {m['median_seismogram_count']:.1f}")
            
            # Print per-class metrics
            for event_type in event_types:
                if event_type in m['precision']:
                    # Get pre-computed event count
                    events_count = model_event_counts[model][m['params']].get(event_type, 0)
                    add_line(f"  - {event_type.upper()}: Precision: {m['precision'][event_type]:.3f}, "
                              f"Recall: {m['recall'][event_type]:.3f}, F1: {m['f1'][event_type]:.3f}, "
                              f"Total events: {events_count}")
            add_line("")
        
        # Write all lines at once
        f.write('\n'.join(output_lines))
    
    print(f"Results written to {output_path}")


def generate_confusion_matrices(metrics):
    """
    Generate and save confusion matrix plots for each model.
    
    Args:
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating confusion matrices...")
    
    for model in sorted(metrics.keys()):
        # Create the confusion matrix figure
        plt.figure(figsize=(10, 8))
        
        # Extract confusion matrix data
        cm = metrics[model]['confusion_matrix']
        
        # Determine all classes present in the confusion matrix
        all_classes = set(event_types + ['no'])
        for true_label in cm:
            all_classes.add(true_label)
            for pred_label in cm[true_label]:
                all_classes.add(pred_label)
        
        all_classes = sorted(list(all_classes))
        matrix_size = len(all_classes)
        
        # Create numpy array for the confusion matrix
        confusion_array = np.zeros((matrix_size, matrix_size))
        
        # Fill the confusion matrix
        for i, true_label in enumerate(all_classes):
            for j, pred_label in enumerate(all_classes):
                if true_label in cm and pred_label in cm[true_label]:
                    confusion_array[i, j] = cm[true_label][pred_label]
        
        # Plot the confusion matrix
        plt.imshow(confusion_array, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {model} with {metrics[model]["params"]}')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(matrix_size)
        plt.xticks(tick_marks, all_classes, rotation=45)
        plt.yticks(tick_marks, all_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add the numbers
        thresh = confusion_array.max() / 2.
        for i in range(matrix_size):
            for j in range(matrix_size):
                plt.text(j, i, format(int(confusion_array[i, j]), 'd'),
                         ha="center", va="center",
                         color="white" if confusion_array[i, j] > thresh else "black")
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved confusion matrix for {model} to {output_path}")

def generate_performance_plots(metrics):
    """
    Generate and save performance metric plots (precision, recall, F1) with both event-wise and trace-wise results.
    Uses corrected metrics based on true event counts.
    
    Args:
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating performance metric plots...")
    #print('trace predictions: ',trace_predictions)
    # Extract data for plotting
    models = sorted(metrics.keys())
    
    # Set up a figure with 3 subplots for precision, recall, and F1 - make it taller
    fig, axes = plt.subplots(3, 1, figsize=(14, 24), sharex=True)
    
    # Set bar width and positions
    bar_width = 0.15
    x = np.arange(len(models))
    
    # Colors for each event type
    colors = {'eq': 'blue', 'ex': 'red', 'su': 'green'}
    
    # Create hatching pattern for trace-wise bars
    trace_hatch = '////'
    
    # Function to add formatted labels to bars
    def add_labels_f2(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0.02:  # Only add label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Calculate trace-level metrics directly here for each model
    trace_metrics = {}
    # grab the "all" channel predictions:   # Next line from chatgpt
    overall_preds = trace_predictions.get('all', {})
    for model in models:
        # Skip models with no trace data
        if model not in trace_probdist_results:
            continue
            
        # Initialize metrics dictionary
        trace_metrics[model] = {'precision': {}, 'recall': {}, 'f1': {}}
        
        # For each event type, calculate trace-level metrics from trace_predictions 
        # which stores true positives, false positives, and false negatives
        for event_type in event_types:
            """   # Was claude:
            # True positives: Traces of this type correctly classified
            tp = len(trace_predictions[model]["true_positive"].get(event_type, []))
            
            # False positives: Traces of other types classified as this type
            fp = len(trace_predictions[model]["false_positive"].get(event_type, []))
            
            # False negatives: Traces of this type classified as something else
            fn = len(trace_predictions[model]["false_negative"].get(event_type, []))
            
            # Calculate precision
            precision = tp / max(tp + fp, 1)
            
            # Calculate recall
            recall = tp / max(tp + fn, 1)

            # Calculate F1 score
            f1 = 2 * precision * recall / max(precision + recall, 0.001)
            """
            # chatgpt
            tp = len(overall_preds[model]["true_positive"].get(event_type, []))
            fp = len(overall_preds[model]["false_positive"].get(event_type, []))
            fn = len(overall_preds[model]["false_negative"].get(event_type, []))

            precision = tp / (tp + fp or 1)
            recall    = tp / (tp + fn or 1)
            f1        = 2 * precision * recall / (precision + recall or 0.001)
            
            # Store metrics
            trace_metrics[model]['precision'][event_type] = precision
            trace_metrics[model]['recall'][event_type] = recall
            trace_metrics[model]['f1'][event_type] = f1
    
    # Plot precision (top subplot)
    ax = axes[0]
    for i, event_type in enumerate(event_types):
        # Event-wise bars (solid)
        event_values = [metrics[model]['precision'][event_type] if model in metrics and event_type in metrics[model]['precision'] else 0 
                  for model in models]
        event_bars = ax.bar(x + (i-1.5)*bar_width, event_values, bar_width, label=f"{event_type.upper()} (Event)", 
                color=colors[event_type])
        # Add labels to event bars
        add_labels_f2(event_bars, ax)
        
        # Trace-wise bars (hatched)
        trace_values = [trace_metrics[model]['precision'][event_type] if model in trace_metrics and event_type in trace_metrics[model]['precision'] else 0 
                   for model in models]
        trace_bars = ax.bar(x + (i-0.5)*bar_width, trace_values, bar_width, label=f"{event_type.upper()} (Trace)", 
                 color=colors[event_type], hatch=trace_hatch)
        # Add labels to trace bars
        add_labels_f2(trace_bars, ax)
    
    ax.set_ylabel('Precision')
    ax.set_title('Precision by Model and Event Type')
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.grid(axis='y', alpha=0.3)
    # Remove vertical grid lines
    ax.grid(axis='x', alpha=0)
    
    # Plot recall (middle subplot)
    ax = axes[1]
    for i, event_type in enumerate(event_types):
        # Event-wise bars (solid)
        event_values = [metrics[model]['recall'][event_type] if model in metrics and event_type in metrics[model]['recall'] else 0 
                  for model in models]
        event_bars = ax.bar(x + (i-1.5)*bar_width, event_values, bar_width, label=f"{event_type.upper()} (Event)", 
                color=colors[event_type])
        # Add labels to event bars
        add_labels_f2(event_bars, ax)
        
        # Trace-wise bars (hatched)
        trace_values = [trace_metrics[model]['recall'][event_type] if model in trace_metrics and event_type in trace_metrics[model]['recall'] else 0 
                   for model in models]
        trace_bars = ax.bar(x + (i-0.5)*bar_width, trace_values, bar_width, label=f"{event_type.upper()} (Trace)", 
                 color=colors[event_type], hatch=trace_hatch)
        # Add labels to trace bars
        add_labels_f2(trace_bars, ax)
    
    ax.set_ylabel('Recall')
    ax.set_title('Recall by Model and Event Type')
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.grid(axis='y', alpha=0.3)
    # Remove vertical grid lines
    ax.grid(axis='x', alpha=0)
    
    # Plot F1 (bottom subplot)
    ax = axes[2]
    for i, event_type in enumerate(event_types):
        # Event-wise bars (solid)
        event_values = [metrics[model]['f1'][event_type] if model in metrics and event_type in metrics[model]['f1'] else 0 
                  for model in models]
        event_bars = ax.bar(x + (i-1.5)*bar_width, event_values, bar_width, label=f"{event_type.upper()} (Event)", 
                color=colors[event_type])
        # Add labels to event bars
        add_labels_f2(event_bars, ax)
        
        # Trace-wise bars (hatched) - ensure consistent bar positioning with other subplots
        trace_values = [trace_metrics[model]['f1'][event_type] if model in trace_metrics and event_type in trace_metrics[model]['f1'] else 0 
                   for model in models]
        # Fix the inconsistent bar position
        trace_bars = ax.bar(x + (i-0.5)*bar_width, trace_values, bar_width, label=f"{event_type.upper()} (Trace)", 
                 color=colors[event_type], hatch=trace_hatch)
        # Add labels to trace bars
        add_labels_f2(trace_bars, ax)
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Model and Event Type')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.grid(axis='y', alpha=0.3)
    # Remove vertical grid lines
    ax.grid(axis='x', alpha=0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for x labels
    output_path = os.path.join(OUTPUT_DIR, 'performance_metrics.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved performance metrics plot to {output_path}")

def generate_magnitude_histograms_v1(best_overall_params):
    """
    Generate magnitude histograms for correct vs. incorrect predictions by model.
    
    Creates histograms for each model with:
    - The same x-axis limits across all models
    - The same y-axis limits across ALL histograms
    - Includes performance metrics on each plot
    """
    print("\nGenerating magnitude histograms (v1)...")
    
    # First determine global min and max magnitude across all models
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    
    # Also track max count for all histograms to standardize y-axes
    global_max_count = 0
    
    # Compute temporary histograms to find the max counts
    temp_hists = {}
    
    # Find the global min and max magnitude across all models
    for model in sorted(model_results.keys()):
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    mags = model_results[model][status][event_type]
                    if mags:
                        global_min_mag = min(global_min_mag, min(mags))
                        global_max_mag = max(global_max_mag, max(mags))
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No magnitude data found for any model. Skipping histogram generation.")
        return
    
    # Calculate global bin parameters
    bins = np.arange(global_min_mag - BIN_WIDTH/2, global_max_mag + BIN_WIDTH, BIN_WIDTH)
    
    # Pre-compute histograms to find maximum y values across ALL histograms
    for model in sorted(model_results.keys()):
        temp_hists[model] = {"correct": {}, "incorrect": {}}
        
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            continue
        
        # Calculate histograms for each event type
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    hist, _ = np.histogram(model_results[model][status][event_type], bins=bins)
                    temp_hists[model][status][event_type] = hist
                    
                    # Update global max count for ALL histograms
                    global_max_count = max(global_max_count, np.max(hist) if hist.size > 0 else 0)
    
    # Add a small buffer to max count to prevent bars from touching the top
    global_max_count = int(global_max_count * 1.1) + 1
    
    print(f"Global y-axis maximum for ALL histograms: {global_max_count}")
    
    # Find best parameter metrics for each model
    model_metrics = {}
    for model in sorted(model_results.keys()):
        if model not in best_overall_params:
            continue
        
        params = best_overall_params[model]
        if not params:
            continue
            
        # Calculate metrics
        correct = overall_correct[model][params]
        total = overall_total[model][params]
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, F1 for each class
        precision = {}
        recall = {}
        f1 = {}
        class_totals = {}
        
        cm = confusion_matrices[model][params]
        for event_type in event_types:
            # True positives: Predicted this class correctly
            tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
            
            # False positives: Predicted this class but was wrong
            fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                     for true in cm if true != event_type)
            
            # False negatives: Should have predicted this class but didn't
            fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                     for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                     if pred != event_type)
            
            # Get total events for this class
            class_totals[event_type] = total_predictions[model][params][event_type]
            
            # Calculate metrics
            if tp + fp > 0:
                precision[event_type] = tp / (tp + fp)
            else:
                precision[event_type] = 0
                
            if tp + fn > 0:
                recall[event_type] = tp / (tp + fn)
            else:
                recall[event_type] = 0
                
            if precision[event_type] + recall[event_type] > 0:
                f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
            else:
                f1[event_type] = 0
        
        # Calculate macro average metrics
        macro_precision = sum(precision.values()) / len(precision) if precision else 0
        macro_recall = sum(recall.values()) / len(recall) if recall else 0
        macro_f1 = sum(f1.values()) / len(f1) if f1 else 0
        
        # Store metrics for this model
        model_metrics[model] = {
            'params': params,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total': total,
            'class_precision': precision,
            'class_recall': recall, 
            'class_f1': f1,
            'class_totals': class_totals
        }
    
    # Now generate histograms for each model using the global ranges
    for model in sorted(model_results.keys()):
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            print(f"Skipping {model} - no magnitude data available")
            continue
        
        # Generate v1 plots (shared y-axis scaling)
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot correct predictions (left subplot)
        ax1.set_title(f'Correct Predictions - {model}')
        for event_type in event_types:
            if event_type in model_results[model]["correct"] and model_results[model]["correct"][event_type]:
                ax1.hist(model_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['correct'][event_type])})")
        
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax1.set_xlim(global_min_mag, global_max_mag)
        # Set consistent y-axis limits for ALL histograms
        ax1.set_ylim(0, global_max_count)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot incorrect predictions (right subplot)
        ax2.set_title(f'Incorrect Predictions - {model}')
        for event_type in event_types:
            if event_type in model_results[model]["incorrect"] and model_results[model]["incorrect"][event_type]:
                ax2.hist(model_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['incorrect'][event_type])})")
        
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax2.set_xlim(global_min_mag, global_max_mag)
        # Set consistent y-axis limits for ALL histograms (same as correct predictions panel)
        ax2.set_ylim(0, global_max_count)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add the performance metrics as text
        if model in model_metrics:
            m = model_metrics[model]
            metrics_text = (
                f"Model: {model}  params: {m['params']}\n"
                f"Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}\n"
            )
            
            # Add per-class metrics
            class_metrics = []
            for event_type in event_types:
                if event_type in m['class_precision']:
                    class_metrics.append(
                        f"{event_type.upper()}: P={m['class_precision'][event_type]:.3f}, "
                        f"R={m['class_recall'][event_type]:.3f}, F1={m['class_f1'][event_type]:.3f}, "
                        f"Events={m['class_totals'][event_type]}"
                    )
            
            if class_metrics:
                metrics_text += "\n" + "\n".join(class_metrics)
            
            # Add metrics text to the bottom of the figure
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.suptitle(f'Magnitude Distribution for {model}')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
        
        output_path = os.path.join(OUTPUT_DIR, f'magnitude_histograms_{model}_v1.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved magnitude histogram for {model} (v1) to {output_path}")

def generate_magnitude_histograms_v2(best_overall_params):
    """
    Generate magnitude histograms for correct vs. incorrect predictions by model.
    
    Creates histograms for each model with:
    - The same x-axis limits across all models
    - The same y-axis limits for correct predictions across all models
    - Independent y-axis limits for incorrect predictions (but consistent across models)
    - Includes performance metrics on each plot
    """
    print("\nGenerating magnitude histograms (v2)...")
    
    # First determine global min and max magnitude across all models
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    
    # Also track max count for correct histograms to standardize y-axes
    global_max_count_correct = 0
    
    # Also track max count for incorrect histograms to standardize y-axes
    global_max_count_incorrect = 0
    
    # Compute temporary histograms to find the max counts
    temp_hists = {}
    
    # Find the global min and max magnitude across all models
    for model in sorted(model_results.keys()):
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    mags = model_results[model][status][event_type]
                    if mags:
                        global_min_mag = min(global_min_mag, min(mags))
                        global_max_mag = max(global_max_mag, max(mags))
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No magnitude data found for any model. Skipping histogram generation.")
        return
    
    # Calculate global bin parameters
    bins = np.arange(global_min_mag - BIN_WIDTH/2, global_max_mag + BIN_WIDTH, BIN_WIDTH)
    
    # Pre-compute histograms to find maximum y values for each panel
    for model in sorted(model_results.keys()):
        temp_hists[model] = {"correct": {}, "incorrect": {}}
        
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            continue
        
        # Calculate histograms for each event type
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    hist, _ = np.histogram(model_results[model][status][event_type], bins=bins)
                    temp_hists[model][status][event_type] = hist
                    
                    # Update global max count for each panel type
                    if status == "correct":
                        global_max_count_correct = max(global_max_count_correct, np.max(hist) if hist.size > 0 else 0)
                    else:  # incorrect
                        global_max_count_incorrect = max(global_max_count_incorrect, np.max(hist) if hist.size > 0 else 0)
    
    # Add a small buffer to max counts to prevent bars from touching the top
    global_max_count_correct = int(global_max_count_correct * 1.1) + 1
    global_max_count_incorrect = int(global_max_count_incorrect * 1.1) + 1
    
    print(f"Global y-axis maximum for correct histograms: {global_max_count_correct}")
    print(f"Global y-axis maximum for incorrect histograms: {global_max_count_incorrect}")
    
    # Find best parameter metrics for each model
    model_metrics = {}
    for model in sorted(model_results.keys()):
        if model not in best_overall_params:
            continue
        
        params = best_overall_params[model]
        if not params:
            continue
            
        # Calculate metrics
        correct = overall_correct[model][params]
        total = overall_total[model][params]
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, F1 for each class
        precision = {}
        recall = {}
        f1 = {}
        class_totals = {}
        
        cm = confusion_matrices[model][params]
        for event_type in event_types:
            # True positives: Predicted this class correctly
            tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
            
            # False positives: Predicted this class but was wrong
            fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                     for true in cm if true != event_type)
            
            # False negatives: Should have predicted this class but didn't
            fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                     for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                     if pred != event_type)
            
            # Get total events for this class
            class_totals[event_type] = total_predictions[model][params][event_type]
            
            # Calculate metrics
            if tp + fp > 0:
                precision[event_type] = tp / (tp + fp)
            else:
                precision[event_type] = 0
                
            if tp + fn > 0:
                recall[event_type] = tp / (tp + fn)
            else:
                recall[event_type] = 0
                
            if precision[event_type] + recall[event_type] > 0:
                f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
            else:
                f1[event_type] = 0
        
        # Calculate macro average metrics
        macro_precision = sum(precision.values()) / len(precision) if precision else 0
        macro_recall = sum(recall.values()) / len(recall) if recall else 0
        macro_f1 = sum(f1.values()) / len(f1) if f1 else 0
        
        # Store metrics for this model
        model_metrics[model] = {
            'params': params,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total': total,
            'class_precision': precision,
            'class_recall': recall, 
            'class_f1': f1,
            'class_totals': class_totals
        }
    
    # Now generate histograms for each model using the global ranges
    for model in sorted(model_results.keys()):
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            print(f"Skipping {model} - no magnitude data available")
            continue
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot correct predictions (left subplot)
        ax1.set_title(f'Correct Predictions - {model}')
        for event_type in event_types:
            if event_type in model_results[model]["correct"] and model_results[model]["correct"][event_type]:
                ax1.hist(model_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['correct'][event_type])})")
        
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax1.set_xlim(global_min_mag, global_max_mag)
        # Set consistent y-axis limits for correct panel
        ax1.set_ylim(0, global_max_count_correct)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot incorrect predictions (right subplot)
        ax2.set_title(f'Incorrect Predictions - {model}')
        for event_type in event_types:
            if event_type in model_results[model]["incorrect"] and model_results[model]["incorrect"][event_type]:
                ax2.hist(model_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['incorrect'][event_type])})")
        
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax2.set_xlim(global_min_mag, global_max_mag)
        # Set y-axis limits for incorrect panel (consistent across models but different from correct panel)
        ###### ax2.set_ylim(0, global_max_count_incorrect)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add the performance metrics as text
        if model in model_metrics:
            m = model_metrics[model]
            metrics_text = (
                f"Model: {model}  params: {m['params']}\n"
                f"Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}\n"
            )
            
            # Add per-class metrics
            class_metrics = []
            for event_type in event_types:
                if event_type in m['class_precision']:
                    class_metrics.append(
                        f"{event_type.upper()}: P={m['class_precision'][event_type]:.3f}, "
                        f"R={m['class_recall'][event_type]:.3f}, F1={m['class_f1'][event_type]:.3f}, "
                        f"Events={m['class_totals'][event_type]}"
                    )
            
            if class_metrics:
                metrics_text += "\n" + "\n".join(class_metrics)
            
            # Add metrics text to the bottom of the figure
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.suptitle(f'Magnitude Distribution for {model}')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
        
        output_path = os.path.join(OUTPUT_DIR, f'magnitude_histograms_{model}_v2.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved magnitude histogram for {model} (v2) to {output_path}")

def generate_probdist_histograms_trace(best_overall_params):
   """
   Generate probability distance histograms for correct vs. incorrect predictions by model (trace-wise).
   
   Creates histograms for each model with:
   - The same x-axis limits across all models
   - The same y-axis limits across all histograms
   - Includes trace-level statistics on each plot
   """
   print("\nGenerating probability distance histograms (trace-wise)...")
   
   # First determine global max probability distance across all models
   global_max_probdist = 0.0
   
   # Also track max count for all histograms to standardize y-axes
   global_max_count = 0
   
   # Compute temporary histograms to find the max counts
   temp_hists = {}
   
   # Find the global max probability distance across all models
   for model in sorted(trace_probdist_results.keys()):
       for status in ["correct", "incorrect"]:
           for event_type in event_types:
               if event_type in trace_probdist_results[model][status] and trace_probdist_results[model][status][event_type]:
                   probdists = trace_probdist_results[model][status][event_type]
                   if probdists:
                       global_max_probdist = max(global_max_probdist, max(probdists))
   
   # If no data found, exit
   if global_max_probdist == 0.0:
       print("No trace-wise probability distance data found for any model. Skipping histogram generation.")
       return
   
   # Calculate global bin parameters
   bins = np.linspace(0.0, global_max_probdist, 20)
   
   # Pre-compute histograms to find maximum y values across ALL histograms
   for model in sorted(trace_probdist_results.keys()):
       temp_hists[model] = {"correct": {}, "incorrect": {}}
       
       # Skip if no probability distance data for this model
       if not any(trace_probdist_results[model]["correct"].values()) and not any(trace_probdist_results[model]["incorrect"].values()):
           continue
       
       # Calculate histograms for each event type
       for status in ["correct", "incorrect"]:
           for event_type in event_types:
               if event_type in trace_probdist_results[model][status] and trace_probdist_results[model][status][event_type]:
                   hist, _ = np.histogram(trace_probdist_results[model][status][event_type], bins=bins)
                   temp_hists[model][status][event_type] = hist
                   
                   # Update global max count for ALL histograms
                   global_max_count = max(global_max_count, np.max(hist) if hist.size > 0 else 0)
   
   # Add a small buffer to max count to prevent bars from touching the top
   global_max_count = int(global_max_count * 1.1) + 1
   
   print(f"Global y-axis maximum for ALL trace-wise probability distance histograms: {global_max_count}")
   
   # Now generate histograms for each model using the global ranges
   for model in sorted(trace_probdist_results.keys()):
       # Skip if no probability distance data for this model
       if not any(trace_probdist_results[model]["correct"].values()) and not any(trace_probdist_results[model]["incorrect"].values()):
           print(f"Skipping {model} - no trace-wise probability distance data available")
           continue
       
       # Create a figure with two subplots side by side
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
       
       # Plot correct predictions (left subplot)
       ax1.set_title(f'Correct Predictions - {model}')
       total_correct_traces = 0
       for event_type in event_types:
           if event_type in trace_probdist_results[model]["correct"] and trace_probdist_results[model]["correct"][event_type]:
               count = len(trace_probdist_results[model]["correct"][event_type])
               total_correct_traces += count
               ax1.hist(trace_probdist_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                       color=COLORS[event_type], label=f"{event_type.upper()} (n={count})")
       
       ax1.set_xlabel('Probability Distance')
       ax1.set_ylabel('Frequency')
       # Set consistent x-axis limits
       ax1.set_xlim(0, global_max_probdist)
       # Set consistent y-axis limits for ALL histograms
       ax1.set_ylim(0, global_max_count)
       ax1.legend()
       ax1.grid(alpha=0.3)
       
       # Plot incorrect predictions (right subplot)
       ax2.set_title(f'Incorrect Predictions - {model}')
       total_incorrect_traces = 0
       class_incorrect_counts = {}
       for event_type in event_types:
           if event_type in trace_probdist_results[model]["incorrect"] and trace_probdist_results[model]["incorrect"][event_type]:
               count = len(trace_probdist_results[model]["incorrect"][event_type])
               total_incorrect_traces += count
               class_incorrect_counts[event_type] = count
               ax2.hist(trace_probdist_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                       color=COLORS[event_type], label=f"{event_type.upper()} (n={count})")
       
       ax2.set_xlabel('Probability Distance')
       ax2.set_ylabel('Frequency')
       # Set consistent x-axis limits
       ax2.set_xlim(0, global_max_probdist)
       # Set consistent y-axis limits for ALL histograms
       ax2.set_ylim(0, global_max_count)
       ax2.legend()
       ax2.grid(alpha=0.3)
       
       # Calculate trace-level statistics
       total_traces = total_correct_traces + total_incorrect_traces
       accuracy = total_correct_traces / total_traces if total_traces > 0 else 0
       
       # Add trace-level statistics as text
       trace_stats_text = (
           f"Model: {model} - Trace-Level Statistics\n"
           f"Total traces: {total_traces}, Correct: {total_correct_traces}, "
           f"Incorrect: {total_incorrect_traces}, Accuracy: {accuracy:.3f}\n"
       )
       
       # Add per-class trace counts
       class_stats = []
       for event_type in event_types:
           correct_count = len(trace_probdist_results[model]["correct"].get(event_type, []))
           incorrect_count = class_incorrect_counts.get(event_type, 0)
           total_count = correct_count + incorrect_count
           class_accuracy = correct_count / total_count if total_count > 0 else 0
           class_stats.append(
               f"{event_type.upper()}: Correct: {correct_count}, Incorrect: {incorrect_count}, "
               f"Total: {total_count}, Accuracy: {class_accuracy:.3f}"
           )
       
       if class_stats:
           trace_stats_text += "\n" + "\n".join(class_stats)
       
       # Add trace stats text to the bottom of the figure
       plt.figtext(0.5, 0.01, trace_stats_text, ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
       
       plt.suptitle(f'Probability Distance Distribution for {model} (trace-wise)')
       plt.tight_layout()
       plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
       
       output_path = os.path.join(OUTPUT_DIR, f'probdist_histograms_{model}_trace.png')
       plt.savefig(output_path, dpi=OUTPUT_DPI)
       print(f"Saved probability distance histogram for {model} (trace-wise) to {output_path}")

'''   ######## CHECK THIS redundant function might be the right one
def generate_probdist_histograms_trace(best_overall_params):
    """
    Generate probability distance histograms for correct vs. incorrect predictions by model (trace-wise).
    
    Creates histograms for each model with:
    - The same x-axis limits across all models
    - The same y-axis limits across all histograms
    - Includes trace-level statistics on each plot
    """
    # [Rest of the function remains the same until the metrics text part]
    
    # Now generate histograms for each model using the global ranges
    for model in sorted(trace_probdist_results.keys()):
        # Skip if no probability distance data for this model
        if not any(trace_probdist_results[model]["correct"].values()) and not any(trace_probdist_results[model]["incorrect"].values()):
            print(f"Skipping {model} - no trace-wise probability distance data available")
            continue
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot correct predictions (left subplot)
        ax1.set_title(f'Correct Predictions - {model}')
        total_correct_traces = 0
        for event_type in event_types:
            if event_type in trace_probdist_results[model]["correct"] and trace_probdist_results[model]["correct"][event_type]:
                count = len(trace_probdist_results[model]["correct"][event_type])
                total_correct_traces += count
                ax1.hist(trace_probdist_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={count})")
        
        ax1.set_xlabel('Probability Distance')
        ax1.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax1.set_xlim(0, global_max_probdist)
        # Set consistent y-axis limits for ALL histograms
        ax1.set_ylim(0, global_max_count)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot incorrect predictions (right subplot)
        ax2.set_title(f'Incorrect Predictions - {model}')
        total_incorrect_traces = 0
        class_incorrect_counts = {}
        for event_type in event_types:
            if event_type in trace_probdist_results[model]["incorrect"] and trace_probdist_results[model]["incorrect"][event_type]:
                count = len(trace_probdist_results[model]["incorrect"][event_type])
                total_incorrect_traces += count
                class_incorrect_counts[event_type] = count
                ax2.hist(trace_probdist_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                        color=COLORS[event_type], label=f"{event_type.upper()} (n={count})")
        
        ax2.set_xlabel('Probability Distance')
        ax2.set_ylabel('Frequency')
        # Set consistent x-axis limits
        ax2.set_xlim(0, global_max_probdist)
        # Set consistent y-axis limits for ALL histograms
        ax2.set_ylim(0, global_max_count)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Calculate trace-level statistics
        total_traces = total_correct_traces + total_incorrect_traces
        accuracy = total_correct_traces / total_traces if total_traces > 0 else 0
        
        # Add trace-level statistics as text
        trace_stats_text = (
            f"Model: {model} - Trace-Level Statistics\n"
            f"Total traces: {total_traces}, Correct: {total_correct_traces}, "
            f"Incorrect: {total_incorrect_traces}, Accuracy: {accuracy:.3f}\n"
        )
        
        # Add per-class trace counts
        class_stats = []
        for event_type in event_types:
            correct_count = len(trace_probdist_results[model]["correct"].get(event_type, []))
            incorrect_count = class_incorrect_counts.get(event_type, 0)
            total_count = correct_count + incorrect_count
            class_accuracy = correct_count / total_count if total_count > 0 else 0
            class_stats.append(
                f"{event_type.upper()}: Correct: {correct_count}, Incorrect: {incorrect_count}, "
                f"Total: {total_count}, Accuracy: {class_accuracy:.3f}"
            )
        
        if class_stats:
            trace_stats_text += "\n" + "\n".join(class_stats)
        
        # Add trace stats text to the bottom of the figure
        plt.figtext(0.5, 0.01, trace_stats_text, ha='center', fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.suptitle(f'Probability Distance Distribution for {model} (trace-wise)')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
        
        output_path = os.path.join(OUTPUT_DIR, f'probdist_histograms_{model}_trace.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved probability distance histogram for {model} (trace-wise) to {output_path}")
'''

def generate_probdist_histograms_event(best_overall_params):
   """
   Generate probability distance histograms for correct vs. incorrect predictions by model (event-wise).
   
   Creates histograms for each model with:
   - The same x-axis limits across all models
   - The same y-axis limits across all histograms
   - Includes performance metrics on each plot
   """
   print("\nGenerating probability distance histograms (event-wise)...")
   
   # First determine global max probability distance across all models
   global_max_probdist = 0.0
   
   # Also track max count for all histograms to standardize y-axes
   global_max_count = 0
   
   # Compute temporary histograms to find the max counts
   temp_hists = {}
   
   # Find the global max probability distance across all models
   for model in sorted(event_probdist_results.keys()):
       for status in ["correct", "incorrect"]:
           for event_type in event_types:
               if event_type in event_probdist_results[model][status] and event_probdist_results[model][status][event_type]:
                   probdists = event_probdist_results[model][status][event_type]
                   if probdists:
                       global_max_probdist = max(global_max_probdist, max(probdists))
   
   # If no data found, exit
   if global_max_probdist == 0.0:
       print("No probability distance data found for any model. Skipping histogram generation.")
       return
   
   # Calculate global bin parameters
   bins = np.linspace(0.0, global_max_probdist, 20)
   
   # Pre-compute histograms to find maximum y values across ALL histograms
   for model in sorted(event_probdist_results.keys()):
       temp_hists[model] = {"correct": {}, "incorrect": {}}
       
       # Skip if no probability distance data for this model
       if not any(event_probdist_results[model]["correct"].values()) and not any(event_probdist_results[model]["incorrect"].values()):
           continue
       
       # Calculate histograms for each event type
       for status in ["correct", "incorrect"]:
           for event_type in event_types:
               if event_type in event_probdist_results[model][status] and event_probdist_results[model][status][event_type]:
                   hist, _ = np.histogram(event_probdist_results[model][status][event_type], bins=bins)
                   temp_hists[model][status][event_type] = hist
                   
                   # Update global max count for ALL histograms
                   global_max_count = max(global_max_count, np.max(hist) if hist.size > 0 else 0)
   
   # Add a small buffer to max count to prevent bars from touching the top
   global_max_count = int(global_max_count * 1.1) + 1
   
   print(f"Global y-axis maximum for ALL probability distance histograms: {global_max_count}")
   
   # Find best parameter metrics for each model
   model_metrics = {}
   for model in sorted(event_probdist_results.keys()):
       if model not in best_overall_params:
           continue
       
       params = best_overall_params[model]
       if not params:
           continue
           
       # Calculate metrics
       correct = overall_correct[model][params]
       total = overall_total[model][params]
       accuracy = correct / total if total > 0 else 0
       
       # Calculate precision, recall, F1 for each class
       precision = {}
       recall = {}
       f1 = {}
       class_totals = {}
       
       cm = confusion_matrices[model][params]
       for event_type in event_types:
           # True positives: Predicted this class correctly
           tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
           
           # False positives: Predicted this class but was wrong
           fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                    for true in cm if true != event_type)
           
           # False negatives: Should have predicted this class but didn't
           fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                    for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                    if pred != event_type)
           
           # Get total events for this class
           class_totals[event_type] = total_predictions[model][params][event_type]
           
           # Calculate metrics
           if tp + fp > 0:
               precision[event_type] = tp / (tp + fp)
           else:
               precision[event_type] = 0
               
           if tp + fn > 0:
               recall[event_type] = tp / (tp + fn)
           else:
               recall[event_type] = 0
               
           if precision[event_type] + recall[event_type] > 0:
               f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
           else:
               f1[event_type] = 0
       
       # Calculate macro average metrics
       macro_precision = sum(precision.values()) / len(precision) if precision else 0
       macro_recall = sum(recall.values()) / len(recall) if recall else 0
       macro_f1 = sum(f1.values()) / len(f1) if f1 else 0
       
       # Store metrics for this model
       model_metrics[model] = {
           'params': params,
           'accuracy': accuracy,
           'macro_precision': macro_precision,
           'macro_recall': macro_recall,
           'macro_f1': macro_f1,
           'total': total,
           'class_precision': precision,
           'class_recall': recall, 
           'class_f1': f1,
           'class_totals': class_totals
       }
   
   # Now generate histograms for each model using the global ranges
   for model in sorted(event_probdist_results.keys()):
       # Skip if no probability distance data for this model
       if not any(event_probdist_results[model]["correct"].values()) and not any(event_probdist_results[model]["incorrect"].values()):
           print(f"Skipping {model} - no probability distance data available")
           continue
       
       # Create a figure with two subplots side by side
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
       
       # Plot correct predictions (left subplot)
       ax1.set_title(f'Correct Predictions - {model}')
       for event_type in event_types:
           if event_type in event_probdist_results[model]["correct"] and event_probdist_results[model]["correct"][event_type]:
               ax1.hist(event_probdist_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                       color=COLORS[event_type], label=f"{event_type.upper()} (n={len(event_probdist_results[model]['correct'][event_type])})")
       
       ax1.set_xlabel('Probability Distance')
       ax1.set_ylabel('Frequency')
       # Set consistent x-axis limits
       ax1.set_xlim(0, global_max_probdist)
       # Set consistent y-axis limits for ALL histograms
       ax1.set_ylim(0, global_max_count)
       ax1.legend()
       ax1.grid(alpha=0.3)
       
       # Plot incorrect predictions (right subplot)
       ax2.set_title(f'Incorrect Predictions - {model}')
       for event_type in event_types:
           if event_type in event_probdist_results[model]["incorrect"] and event_probdist_results[model]["incorrect"][event_type]:
               ax2.hist(event_probdist_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                       color=COLORS[event_type], label=f"{event_type.upper()} (n={len(event_probdist_results[model]['incorrect'][event_type])})")
       
       ax2.set_xlabel('Probability Distance')
       ax2.set_ylabel('Frequency')
       # Set consistent x-axis limits
       ax2.set_xlim(0, global_max_probdist)
       # Set consistent y-axis limits for ALL histograms
       ax2.set_ylim(0, global_max_count)
       ax2.legend()
       ax2.grid(alpha=0.3)
       
       # Add the performance metrics as text
       if model in model_metrics:
           m = model_metrics[model]
           metrics_text = (
               f"Model: {model}  params: {m['params']}\n"
               f"Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
               f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}\n"
           )
           
           # Add per-class metrics
           class_metrics = []
           for event_type in event_types:
               if event_type in m['class_precision']:
                   class_metrics.append(
                       f"{event_type.upper()}: P={m['class_precision'][event_type]:.3f}, "
                       f"R={m['class_recall'][event_type]:.3f}, F1={m['class_f1'][event_type]:.3f}, "
                       f"Events={m['class_totals'][event_type]}"
                   )
           
           if class_metrics:
               metrics_text += "\n" + "\n".join(class_metrics)
           
           # Add metrics text to the bottom of the figure
           plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
       
       plt.suptitle(f'Probability Distance Distribution for {model} (event-wise)')
       plt.tight_layout()
       plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
       
       output_path = os.path.join(OUTPUT_DIR, f'probdist_histograms_{model}_event.png')
       plt.savefig(output_path, dpi=OUTPUT_DPI)
       print(f"Saved probability distance histogram for {model} (event-wise) to {output_path}")

def generate_max_probability_histograms(best_overall_params):
    """
    Generate histograms of maximum probability values for each model.
    
    Creates histograms showing the distribution of maximum probability values
    for each event type, split by correct and incorrect predictions.
    """
    print("\nGenerating maximum probability histograms...")
    
    # Collect maximum probability values for each event type and model
    max_probs = {
        "event": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),  # Model -> correct/incorrect -> event_type -> [probs]
        "trace": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # Model -> correct/incorrect -> event_type -> [probs]
    }
    
    # Process trace data to collect maximum probabilities
    for model in trace_probdist_results:
        # Process correct predictions
        for event_type in trace_probdist_results[model]["correct"]:
            # Get the trace IDs for this event type
            for trace_id in range(len(trace_probdist_results[model]["correct"][event_type])):
                # Assuming we can access the original probability values
                # This would need to be modified to access actual probability values
                if model in trace_channel_results["all"] and event_type in trace_channel_results["all"][model]["correct"]:
                    # Extract probability from the trace data (this is approximate)
                    # In a real implementation, you would extract the actual max probability
                    prob_data = trace_channel_results["all"][model]["correct"][event_type][trace_id]
                    if isinstance(prob_data, tuple) and len(prob_data) >= 1:
                        # Use the probability distance as a proxy (not ideal)
                        max_prob = prob_data[0]
                        max_probs["trace"][model]["correct"][event_type].append(max_prob)
        
        # Process incorrect predictions similarly
        for event_type in trace_probdist_results[model]["incorrect"]:
            for trace_id in range(len(trace_probdist_results[model]["incorrect"][event_type])):
                if model in trace_channel_results["all"] and event_type in trace_channel_results["all"][model]["incorrect"]:
                    prob_data = trace_channel_results["all"][model]["incorrect"][event_type][trace_id]
                    if isinstance(prob_data, tuple) and len(prob_data) >= 1:
                        max_prob = prob_data[0]
                        max_probs["trace"][model]["incorrect"][event_type].append(max_prob)
    
    # Process event data similarly
    for model in event_probdist_results:
        # Process correct predictions
        for event_type in event_probdist_results[model]["correct"]:
            # For event data, we use probability distance as a proxy
            probs = event_probdist_results[model]["correct"][event_type]
            max_probs["event"][model]["correct"][event_type].extend(probs)
        
        # Process incorrect predictions
        for event_type in event_probdist_results[model]["incorrect"]:
            probs = event_probdist_results[model]["incorrect"][event_type]
            max_probs["event"][model]["incorrect"][event_type].extend(probs)
    
    # Generate histograms
    for data_type in ["event", "trace"]:
        for model in max_probs[data_type]:
            # Create figure
            plt.figure(figsize=(16, 7))
            
            # Set up subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot correct predictions
            ax1.set_title(f'Correct Predictions - {model} ({data_type})')
            for event_type in event_types:
                if event_type in max_probs[data_type][model]["correct"] and max_probs[data_type][model]["correct"][event_type]:
                    values = max_probs[data_type][model]["correct"][event_type]
                    ax1.hist(values, bins=20, alpha=0.6, 
                             color=COLORS[event_type], 
                             label=f"{event_type.upper()} (n={len(values)})")
            
            ax1.set_xlabel('Maximum Probability')
            ax1.set_ylabel('Frequency')
            ax1.set_xlim(0, 1)
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot incorrect predictions
            ax2.set_title(f'Incorrect Predictions - {model} ({data_type})')
            for event_type in event_types:
                if event_type in max_probs[data_type][model]["incorrect"] and max_probs[data_type][model]["incorrect"][event_type]:
                    values = max_probs[data_type][model]["incorrect"][event_type]
                    ax2.hist(values, bins=20, alpha=0.6,
                             color=COLORS[event_type], 
                             label=f"{event_type.upper()} (n={len(values)})")
            
            ax2.set_xlabel('Maximum Probability')
            ax2.set_ylabel('Frequency')
            ax2.set_xlim(0, 1)
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.suptitle(f'Maximum Probability Distribution for {model} ({data_type})')
            plt.tight_layout()
            
            output_path = os.path.join(OUTPUT_DIR, f'max_probability_{model}_{data_type}.png')
            plt.savefig(output_path, dpi=OUTPUT_DPI)
            print(f"Saved maximum probability histogram for {model} ({data_type}) to {output_path}")

def generate_scatter_plots_event():
    """
    Generate scatter plots of probability distance vs. magnitude for incorrect predictions (event-wise).
    
    Creates scatter plots colored by event type with symbol size proportional to trace count.
    """
    print("\nGenerating scatter plots for incorrect predictions (event-wise)...")
    
    # First determine global min and max values
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    global_max_probdist = 0.0
    global_max_trace_count = 0
    
    # Find global limits
    for model in sorted(event_scatter_data.keys()):
        for event_type in event_types:
            if event_type in event_scatter_data[model]["incorrect"] and event_scatter_data[model]["incorrect"][event_type]:
                for probdist, magnitude, trace_count in event_scatter_data[model]["incorrect"][event_type]:
                    global_min_mag = min(global_min_mag, magnitude)
                    global_max_mag = max(global_max_mag, magnitude)
                    global_max_probdist = max(global_max_probdist, probdist)
                    global_max_trace_count = max(global_max_trace_count, trace_count)
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No scatter plot data found for any model. Skipping scatter plot generation.")
        return
    
    # Scale factor for marker sizes
    scale_factor = 50.0 / global_max_trace_count if global_max_trace_count > 0 else 1.0
    
    # Generate scatter plots for each model
    for model in sorted(event_scatter_data.keys()):
        # Skip if no data for this model
        if not any(event_scatter_data[model]["incorrect"].values()):
            print(f"Skipping {model} - no incorrect predictions for scatter plot")
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot data points for each event type
        for event_type in event_types:
            if event_type in event_scatter_data[model]["incorrect"] and event_scatter_data[model]["incorrect"][event_type]:
                # Extract data
                data = event_scatter_data[model]["incorrect"][event_type]
                probdists = [x[0] for x in data]
                magnitudes = [x[1] for x in data]
                trace_counts = [x[2] * scale_factor for x in data]
                
                # Plot scatter with varying sizes
                plt.scatter(probdists, magnitudes, s=trace_counts, alpha=0.7, 
                           color=COLORS[event_type], label=f"{event_type.upper()} (n={len(data)})")
        
        plt.xlabel('Probability Distance')
        plt.ylabel('Magnitude')
        plt.title(f'Incorrect Predictions: Probability Distance vs. Magnitude for {model} (event-wise)')
        plt.xlim(0, global_max_probdist * 1.05)  # Add 5% margin
        plt.ylim(global_min_mag - 0.1, global_max_mag + 0.1)  # Add small margin
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add text explaining marker size
        plt.figtext(0.15, 0.02, "Marker size proportional to number of traces", fontsize=9)
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, f'scatter_probdist_mag_{model}_event.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved event-wise scatter plot for {model} to {output_path}")

def generate_scatter_plots_trace():
    """
    Generate scatter plots of probability distance vs. magnitude for incorrect predictions (trace-wise).
    """
    print("\nGenerating scatter plots for incorrect predictions (trace-wise)...")
    
    # First determine global min and max values
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    global_max_probdist = 0.0
    
    # Find global limits
    for model in sorted(trace_scatter_data.keys()):
        for event_type in event_types:
            if event_type in trace_scatter_data[model]["incorrect"] and trace_scatter_data[model]["incorrect"][event_type]:
                for probdist, magnitude in trace_scatter_data[model]["incorrect"][event_type]:
                    global_min_mag = min(global_min_mag, magnitude)
                    global_max_mag = max(global_max_mag, magnitude)
                    global_max_probdist = max(global_max_probdist, probdist)
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No trace-wise scatter plot data found for any model. Skipping trace-wise scatter plot generation.")
        return
    
    # Generate scatter plots for each model
    for model in sorted(trace_scatter_data.keys()):
        # Skip if no data for this model
        if not any(trace_scatter_data[model]["incorrect"].values()):
            print(f"Skipping {model} - no incorrect trace predictions for scatter plot")
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot data points for each event type
        for event_type in event_types:
            if event_type in trace_scatter_data[model]["incorrect"] and trace_scatter_data[model]["incorrect"][event_type]:
                # Extract data
                data = trace_scatter_data[model]["incorrect"][event_type]
                probdists = [x[0] for x in data]
                magnitudes = [x[1] for x in data]
                
                # Plot scatter with fixed size
                plt.scatter(probdists, magnitudes, s=30, alpha=0.7, 
                           color=COLORS[event_type], label=f"{event_type.upper()} (n={len(data)})")
        
        plt.xlabel('Probability Distance')
        plt.ylabel('Magnitude')
        plt.title(f'Incorrect Predictions: Probability Distance vs. Magnitude for {model} (trace-wise)')
        plt.xlim(0, global_max_probdist * 1.05)  # Add 5% margin
        plt.ylim(global_min_mag - 0.1, global_max_mag + 0.1)  # Add small margin
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, f'scatter_probdist_mag_{model}_trace.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved trace-wise scatter plot for {model} to {output_path}")

def generate_heatmaps_event():
    """
    Generate heatmaps of probability distance vs. magnitude for incorrect predictions (event-wise).
    """
    print("\nGenerating heatmaps for incorrect predictions (event-wise)...")
    
    # First determine global min and max values
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    global_max_probdist = 0.0
    
    # Find global limits
    for model in sorted(event_scatter_data.keys()):
        for event_type in event_types:
            if event_type in event_scatter_data[model]["incorrect"] and event_scatter_data[model]["incorrect"][event_type]:
                for probdist, magnitude, _ in event_scatter_data[model]["incorrect"][event_type]:
                    global_min_mag = min(global_min_mag, magnitude)
                    global_max_mag = max(global_max_mag, magnitude)
                    global_max_probdist = max(global_max_probdist, probdist)
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No heatmap data found for any model. Skipping heatmap generation.")
        return
    
    # Generate heatmaps for each model
    for model in sorted(event_scatter_data.keys()):
        # Skip if no data for this model
        if not any(event_scatter_data[model]["incorrect"].values()):
            print(f"Skipping {model} - no incorrect predictions for heatmap")
            continue
        
        # Create figure with subplots for each event type
        fig, axes = plt.subplots(1, len(event_types), figsize=(18, 6), sharey=True)
        
        # Keep track of the last valid image for colorbar
        last_im = None
        max_count = 0  # Track the maximum count for colorbar scaling
        
        for i, event_type in enumerate(event_types):
            ax = axes[i]
            
            if event_type in event_scatter_data[model]["incorrect"] and event_scatter_data[model]["incorrect"][event_type]:
                # Extract data
                data = event_scatter_data[model]["incorrect"][event_type]
                probdists = [x[0] for x in data]
                magnitudes = [x[1] for x in data]
                
                # Create heatmap using 2D histogram
                heatmap, xedges, yedges = np.histogram2d(
                    probdists, magnitudes, 
                    bins=[20, 20], 
                    range=[[0, global_max_probdist], [global_min_mag, global_max_mag]]
                )
                
                # Transpose for proper orientation
                heatmap = heatmap.T
                
                # Update max count
                if heatmap.size > 0:
                    max_count = max(max_count, np.max(heatmap))
                
                # Plot heatmap
                im = ax.imshow(
                    heatmap, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap='YlOrRd', norm=LogNorm()
                )
                
                # Save this image for colorbar
                last_im = im
                
                ax.set_title(f'{event_type.upper()} (n={len(data)})')
            else:
                ax.text(0.5, 0.5, f"No {event_type.upper()} data", 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Probability Distance')
            if i == 0:
                ax.set_ylabel('Magnitude')
        
        # Add colorbar only if we have a valid image
        if last_im is not None:
            # Create a new axis for the colorbar to the right of all panels
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label('Count')
            
            # Format the colorbar ticks as integers
            from matplotlib.ticker import ScalarFormatter, FuncFormatter
            
            # Use integer formatter
            def int_formatter(x, pos):
                return f"{int(x)}"
            
            cbar.formatter = FuncFormatter(int_formatter)
            cbar.update_ticks()
            
            # Ensure the maximum value is included in the ticks
            if max_count > 0:
                # Get current ticks
                current_ticks = cbar.get_ticks()
                # Make sure the max value is included
                if max_count not in current_ticks:
                    new_ticks = np.append(current_ticks, max_count)
                    cbar.set_ticks(new_ticks)
            
            cbar.update_ticks()
        
        plt.suptitle(f'Heatmap: Probability Distance vs. Magnitude for {model} (event-wise)')
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
        
        output_path = os.path.join(OUTPUT_DIR, f'heatmap_probdist_mag_{model}_event.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved event-wise heatmap for {model} to {output_path}")

def generate_heatmaps_trace():
    """
    Generate heatmaps of probability distance vs. magnitude for incorrect predictions (trace-wise).
    """
    print("\nGenerating heatmaps for incorrect predictions (trace-wise)...")
    
    # First determine global min and max values
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    global_max_probdist = 0.0
    
    # Find global limits
    for model in sorted(trace_scatter_data.keys()):
        for event_type in event_types:
            if event_type in trace_scatter_data[model]["incorrect"] and trace_scatter_data[model]["incorrect"][event_type]:
                for probdist, magnitude in trace_scatter_data[model]["incorrect"][event_type]:
                    global_min_mag = min(global_min_mag, magnitude)
                    global_max_mag = max(global_max_mag, magnitude)
                    global_max_probdist = max(global_max_probdist, probdist)
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No trace-wise heatmap data found for any model. Skipping trace-wise heatmap generation.")
        return
    
    # Generate heatmaps for each model
    for model in sorted(trace_scatter_data.keys()):
        # Skip if no data for this model
        if not any(trace_scatter_data[model]["incorrect"].values()):
            print(f"Skipping {model} - no incorrect trace predictions for heatmap")
            continue
        
        # Create figure with subplots for each event type
        fig, axes = plt.subplots(1, len(event_types), figsize=(18, 6), sharey=True)
        
        # Keep track of the last valid image for colorbar
        last_im = None
        max_count = 0  # Track the maximum count for colorbar scaling
        
        for i, event_type in enumerate(event_types):
            ax = axes[i]
            
            if event_type in trace_scatter_data[model]["incorrect"] and trace_scatter_data[model]["incorrect"][event_type]:
                # Extract data
                data = trace_scatter_data[model]["incorrect"][event_type]
                probdists = [x[0] for x in data]
                magnitudes = [x[1] for x in data]
                
                # Create heatmap using 2D histogram
                heatmap, xedges, yedges = np.histogram2d(
                    probdists, magnitudes, 
                    bins=[20, 20], 
                    range=[[0, global_max_probdist], [global_min_mag, global_max_mag]]
                )
                
                # Transpose for proper orientation
                heatmap = heatmap.T
                
                # Update max count
                if heatmap.size > 0:
                    max_count = max(max_count, np.max(heatmap))
                
                # Plot heatmap
                im = ax.imshow(
                    heatmap, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap='YlOrRd', norm=LogNorm()
                )
                
                # Save this image for colorbar
                last_im = im
                
                ax.set_title(f'{event_type.upper()} (n={len(data)})')
            else:
                ax.text(0.5, 0.5, f"No {event_type.upper()} data", 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Probability Distance')
            if i == 0:
                ax.set_ylabel('Magnitude')
        
        # Add colorbar only if we have a valid image
        if last_im is not None:
            # Create a new axis for the colorbar to the right of all panels
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label('Count')
            
            # Format the colorbar ticks as integers
            from matplotlib.ticker import ScalarFormatter, FuncFormatter
            
            # Use integer formatter
            def int_formatter(x, pos):
                return f"{int(x)}"
            
            # Ensure the maximum value is included in the ticks
            if max_count > 0:
                # Get current ticks
                current_ticks = cbar.get_ticks()
                # Make sure the max value is included
                if max_count not in current_ticks:
                    new_ticks = np.append(current_ticks, max_count)
                    cbar.set_ticks(new_ticks)

            cbar.formatter = FuncFormatter(int_formatter)
            cbar.update_ticks()
        
        plt.suptitle(f'Heatmap: Probability Distance vs. Magnitude for {model} (trace-wise)')
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
        
        output_path = os.path.join(OUTPUT_DIR, f'heatmap_probdist_mag_{model}_trace.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved trace-wise heatmap for {model} to {output_path}")

def build_event_histograms(raw_event_data, best_overall_params):
    """
    Build magnitude and probability distance histograms using only data from the best parameter set for each model.
    
    Args:
        raw_event_data: Dictionary of raw event data
        best_overall_params: Dictionary mapping model -> best parameter set
    """
    print("\nBuilding event-wise histograms for best parameter sets...")

    # Clear any existing model results
    global model_results, event_probdist_results, event_scatter_data
    model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    event_probdist_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    event_scatter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Process each event's data
    for evid, event_entries in raw_event_data.items():
        for entry in event_entries:
            model = entry["model"]
            params = entry["params"]
            pred = entry["pred"]
            analyst = entry["analyst"] 
            magnitude = entry["magnitude"]
            prob_dist = entry["prob_dist"]
            trace_count = entry["trace_count"]

            # Only use data from the best parameter set for each model
            if model in best_overall_params and params == best_overall_params[model]:
                # Handle "no" predictions as incorrect
                if pred == "no":
                    model_results[model]["incorrect"][analyst].append(magnitude)
                    event_probdist_results[model]["incorrect"][analyst].append(prob_dist)
                    event_scatter_data[model]["incorrect"][analyst].append((prob_dist, magnitude, trace_count))
                # Handle normal predictions
                elif pred == analyst:
                    model_results[model]["correct"][pred].append(magnitude)
                    event_probdist_results[model]["correct"][pred].append(prob_dist)
                    event_scatter_data[model]["correct"][pred].append((prob_dist, magnitude, trace_count))
                else:
                    model_results[model]["incorrect"][analyst].append(magnitude)
                    event_probdist_results[model]["incorrect"][analyst].append(prob_dist)
                    event_scatter_data[model]["incorrect"][analyst].append((prob_dist, magnitude, trace_count))

    # Print summary of histogram data
    for model in model_results:
        correct_count = sum(len(mags) for mags in model_results[model]["correct"].values())
        incorrect_count = sum(len(mags) for mags in model_results[model]["incorrect"].values())
        print(f"  {model}: {correct_count} correct predictions, {incorrect_count} incorrect predictions for histograms")

def build_trace_histograms(raw_trace_data, raw_event_data):
    """
    Build trace-wise probability distance histograms and classify traces by channel type.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nBuilding trace-wise histograms...")

    # Clear any existing trace results
    global trace_probdist_results, trace_scatter_data, trace_predictions
    trace_probdist_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    trace_scatter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Initialize channel-type data structures
    global trace_channel_results
    trace_channel_results = {
        'strong_motion': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        '4_channel': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'short_period_3c': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'short_period_vertical': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'broadband': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'all': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    # Initialize prediction tracking structure
    trace_predictions = {
        'strong_motion': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        '4_channel': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'short_period_3c': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'short_period_vertical': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'broadband': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'all': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }

    # First create mapping from evid to analyst label and magnitude
    evid_to_metadata = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            # Just use the first entry's analyst and magnitude (should be the same for all entries)
            evid_to_metadata[evid] = {
                "analyst": event_entries[0]["analyst"],
                "magnitude": event_entries[0]["magnitude"]
            }

    # Pre-process: Create channel type cache to avoid repeated classifications
    channel_cache = {}  # (evid, netsta) -> channel_type
    
    # Process trace data efficiently
    channel_type_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    
    for evid, model_data in raw_trace_data.items():
        # Skip if we don't have metadata for this event
        if evid not in evid_to_metadata:
            continue

        analyst = evid_to_metadata[evid]["analyst"]
        magnitude = evid_to_metadata[evid]["magnitude"]

        for model, traces in model_data.items():
            # Track processed stations per model and evid to avoid duplicates
            processed_stations = set()
            
            for trace_data in traces:
                # Skip if trace data is incomplete
                if len(trace_data) < 3:
                    continue
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                channels = trace_data[2] if len(trace_data) > 2 else []

                # Skip if no channel info
                if not isinstance(channels, list) or not channels:
                    continue
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue
                
                # Skip if already processed this station for this event and model
                if (evid, netsta, model) in processed_stations:
                    continue
                processed_stations.add((evid, netsta, model))
                
                # Extract probabilities for each class (excluding noise)
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skip noise probability
                
                # Use only EQ, EX, SU probabilities for calculating distance
                class_probs = [eq_prob, ex_prob, su_prob]
                
                # Find highest and second highest probabilities
                sorted_probs = sorted(class_probs, reverse=True)
                prob_dist = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
                
                # Find predicted class (index of max probability)
                pred_index = class_probs.index(max(class_probs))
                pred_class = event_types[pred_index]
                
                # Get or classify channel type (using cache)
                if (evid, netsta) in channel_cache:
                    channel_type = channel_cache[(evid, netsta)]
                else:
                    channel_type = classify_channel_type(channels)
                    channel_cache[(evid, netsta)] = channel_type
                
                # Determine if prediction is correct
                is_correct = (pred_class == analyst)
                status = "correct" if is_correct else "incorrect"
                
                # Update overall trace results
                trace_probdist_results[model][status][pred_class if is_correct else analyst].append(prob_dist)
                trace_scatter_data[model][status][pred_class if is_correct else analyst].append((prob_dist, magnitude))
                
                # Update channel-specific results
                trace_channel_results['all'][model][status][pred_class if is_correct else analyst].append((prob_dist, magnitude))
                trace_channel_results[channel_type][model][status][pred_class if is_correct else analyst].append((prob_dist, magnitude))
                
                # Update prediction tracking
                if is_correct:
                    trace_predictions['all'][model]["true_positive"][pred_class].append((prob_dist, magnitude))
                    trace_predictions[channel_type][model]["true_positive"][pred_class].append((prob_dist, magnitude))
                else:
                    trace_predictions['all'][model]["false_negative"][analyst].append((prob_dist, magnitude))
                    trace_predictions[channel_type][model]["false_negative"][analyst].append((prob_dist, magnitude))
                    
                    trace_predictions['all'][model]["false_positive"][pred_class].append((prob_dist, magnitude))
                    trace_predictions[channel_type][model]["false_positive"][pred_class].append((prob_dist, magnitude))
                
                # Update channel type counts
                channel_type_counts[channel_type][status] += 1

    # Print channel-type breakdown (more efficient)
    print("Channel Type Statistics:")
    for channel_type, counts in channel_type_counts.items():
        if channel_type == 'all':
            continue  # Skip 'all' since it's derivative
            
        correct = counts['correct']
        incorrect = counts['incorrect']
        total = correct + incorrect
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"  - {channel_type}: {correct} correct, {incorrect} incorrect, "
                  f"{total} total, {accuracy:.1f}% accuracy")

def classify_channel_type(channels):
    """
    Classify a set of channels into one of the predefined types.
    
    Args:
        channels: List of channel codes in NSLC format ['NET.STA.LOC.CHA', ...]
    
    Returns:
        Channel type classification
    """
    # Extract just the channel codes (last 3 chars of each entry)
    channel_codes = [ch.split('.')[-1] for ch in channels]
    
    # Convert to set for easy checking
    code_set = set(channel_codes)
    
    # Check for Strong Motion (HNE/HNN/HNZ or ENE/ENN/ENZ)
    if ({'HNE', 'HNN', 'HNZ'}.issubset(code_set) or 
        {'ENE', 'ENN', 'ENZ'}.issubset(code_set)):
        return 'strong_motion'
    
    # Check for 4 Channel ((ENE+ENN or HNE+HNN) + EHZ)
    if ('EHZ' in code_set and 
        ({'ENE', 'ENN'}.issubset(code_set) or {'HNE', 'HNN'}.issubset(code_set))):
        return '4_channel'
    
    # Check for Short Period 3 Component (EHE, EHN, EHZ)
    if {'EHE', 'EHN', 'EHZ'}.issubset(code_set):
        return 'short_period_3c'
    
    # Check for Short Period Vertical (EHZ repeated)
    if code_set == {'EHZ'} and len(channel_codes) == 3:
        return 'short_period_vertical'
    
    # Check for Broadband (BHE/BHN/BHZ or HHE/HHN/HHZ)
    if ({'BHE', 'BHN', 'BHZ'}.issubset(code_set) or 
        {'HHE', 'HHN', 'HHZ'}.issubset(code_set)):
        return 'broadband'
    
    # Default to 'other' if no match
    return 'other'

def generate_channel_performance_plots():
    """
    Generate performance metric plots broken down by channel type.
    
    Creates a 2x3 grid of plots showing performance metrics for each channel type.
    Uses the updated data structure to calculate precision correctly.
    Adds trace counts to channel type labels.
    """
    print("\nGenerating channel performance plots...")
    
    # Setup figure with 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey=True)
    
    # Channel types and their display names
    channel_types = [
        ('strong_motion', 'Strong Motion\n(HNE/HNN/HNZ or ENE/ENN/ENZ)'),
        ('4_channel', '4 Channel\n((ENE+ENN or HNE+HNN) + EHZ)'),
        ('short_period_3c', 'Short Period 3 Component\n(EHE/EHN/EHZ)'),
        ('short_period_vertical', 'Short Period Vertical\n(EHZ Only)'),
        ('broadband', 'Broadband\n(BHE/BHN/BHZ or HHE/HHN/HHZ)'),
        ('all', 'All Channels\nCombined')
    ]
    
    # Find global y-max for consistent scaling
    y_max = 0
    
    # Calculate metrics for each channel type and find max value
    channel_metrics = {}
    for channel_type, _ in channel_types:
        channel_metrics[channel_type] = {}
        
        # Get all models that have data for this channel type
        models = []
        for model in sorted(trace_channel_results[channel_type].keys()):
            # Check if there's any data for this model
            if (any(trace_channel_results[channel_type][model]["correct"].values()) or 
                any(trace_channel_results[channel_type][model]["incorrect"].values())):
                models.append(model)
        
        if not models:
            continue  # Skip if no models have data for this channel type
        
        # Calculate metrics for each model
        for model in models:
            # Initialize metrics dictionary for this model
            channel_metrics[channel_type][model] = {
                'precision': {},
                'recall': {},
                'f1': {},
                'accuracy': 0,
                'macro_precision': 0,
                'macro_recall': 0,
                'macro_f1': 0
            }
            
            # Calculate per-class metrics
            for event_type in event_types:
                # Get counts for precision calculation (true positives and false positives)
                true_positives = len(trace_predictions[channel_type][model]["true_positive"].get(event_type, []))
                false_positives = len(trace_predictions[channel_type][model]["false_positive"].get(event_type, []))
                
                # Get counts for recall calculation (true positives and false negatives)
                false_negatives = len(trace_predictions[channel_type][model]["false_negative"].get(event_type, []))
                
                # Calculate total predictions of this class
                total_actual = true_positives + false_negatives
                
                # Calculate precision
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate recall
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store metrics
                channel_metrics[channel_type][model]['precision'][event_type] = precision
                channel_metrics[channel_type][model]['recall'][event_type] = recall
                channel_metrics[channel_type][model]['f1'][event_type] = f1
            
            # Calculate overall accuracy
            total_correct = sum(len(data) for data in trace_channel_results[channel_type][model]["correct"].values())
            total_incorrect = sum(len(data) for data in trace_channel_results[channel_type][model]["incorrect"].values())
            total = total_correct + total_incorrect
            
            accuracy = total_correct / total if total > 0 else 0
            channel_metrics[channel_type][model]['accuracy'] = accuracy
            
            # Calculate macro-averaged metrics (event-class weighted)
            channel_metrics[channel_type][model]['macro_precision'] = (
                sum(channel_metrics[channel_type][model]['precision'].values()) / 
                len(channel_metrics[channel_type][model]['precision']) 
                if channel_metrics[channel_type][model]['precision'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_recall'] = (
                sum(channel_metrics[channel_type][model]['recall'].values()) / 
                len(channel_metrics[channel_type][model]['recall']) 
                if channel_metrics[channel_type][model]['recall'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_f1'] = (
                sum(channel_metrics[channel_type][model]['f1'].values()) / 
                len(channel_metrics[channel_type][model]['f1']) 
                if channel_metrics[channel_type][model]['f1'] else 0
            )
            
            # Update global y-max
            y_max = max(y_max, 
                        channel_metrics[channel_type][model]['macro_precision'],
                        channel_metrics[channel_type][model]['macro_recall'],
                        channel_metrics[channel_type][model]['macro_f1'],
                        channel_metrics[channel_type][model]['accuracy'])
    
    # Add a small buffer to y_max
    y_max = min(1.0, y_max * 1.1)

    # Now plot each channel type in its grid position
    for i, (channel_type, display_name) in enumerate(channel_types):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        models = list(sorted(channel_metrics.get(channel_type, {}).keys()))
        
        if not models:
            ax.text(0.5, 0.5, f"No data for {display_name}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name)
            continue
        
        # Calculate total trace count for this channel type
        total_trace_count = 0
        for model in models:
            # Count unique network.station pairs
            count = calculate_channel_trace_count(trace_channel_results, channel_type, model)
            total_trace_count += count
        
        # Set up bar positions
        x = np.arange(len(models))
        bar_width = 0.2
        
        # Plot bars for each metric
        precision_bars = ax.bar(x - 1.5*bar_width, 
                                [channel_metrics[channel_type][model]['macro_precision'] for model in models],
                                bar_width, label='Precision (Event-class weighted)', color='skyblue')
        
        recall_bars = ax.bar(x - 0.5*bar_width, 
                            [channel_metrics[channel_type][model]['macro_recall'] for model in models],
                            bar_width, label='Recall (Event-class weighted)', color='lightgreen')
        
        f1_bars = ax.bar(x + 0.5*bar_width, 
                        [channel_metrics[channel_type][model]['macro_f1'] for model in models],
                        bar_width, label='F1 (Event-class weighted)', color='salmon')
        
        accuracy_bars = ax.bar(x + 1.5*bar_width, 
                              [channel_metrics[channel_type][model]['accuracy'] for model in models],
                              bar_width, label='Accuracy', color='mediumpurple')
        
        # Add text labels on top of the bars
        bar_label_fontsize = 7
        
        # Function to add formatted labels to bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only add label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        # Add labels to all bar sets
        add_labels(precision_bars)
        add_labels(recall_bars)
        add_labels(f1_bars)
        add_labels(accuracy_bars)
        
        # Customize the plot
        # Add trace count to the channel type name
        title_with_count = f"{display_name}\n(n={total_trace_count})"
        ax.set_title(title_with_count)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        if col == 0:
            ax.set_ylabel('Score')
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set main title and adjust layout
    plt.suptitle('Performance Metrics by Channel Type (Event-class weighted)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'channel_performance_fixed.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved fixed channel performance plot to {output_path}")

## Custom legend handler to help with left-alignment
#class HandlerRectangle(HandlerBase):
#    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
#        # Modify text alignment
#        legend._loc = 3  # Left alignment
#        return super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

# Modified HandlerRectangle class that doesn't try to set _loc property
class HandlerRectangle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create a Rectangle patch without attempting to modify legend._loc
        p = Rectangle(xy=(-xdescent, -ydescent), width=width, height=height,
                     facecolor=orig_handle.get_facecolor(), 
                     edgecolor=orig_handle.get_edgecolor(),
                     lw=orig_handle.get_linewidth())
        return [p]

def calculate_channel_trace_count(trace_channel_results, channel_type, model):
    """
    Calculate total trace count for a specific channel type and model.
    Counts unique stations, not individual channels.
    
    Args:
        trace_channel_results (dict): The full trace results dictionary
        channel_type (str): The channel type to calculate traces for
        model (str): The model to calculate traces for
    
    Returns:
        int: Total number of unique stations for the given channel type and model
    """
    try:
        # Get all traces
        all_traces = []
        
        # Add traces from correct results
        for event_type in trace_channel_results[channel_type][model]["correct"]:
            all_traces.extend(trace_channel_results[channel_type][model]["correct"][event_type])
        
        # Add traces from incorrect results
        for event_type in trace_channel_results[channel_type][model]["incorrect"]:
            all_traces.extend(trace_channel_results[channel_type][model]["incorrect"][event_type])
        
        # Extract unique NET.STA pairs
        unique_stations = set()
        for trace_data in all_traces:
            # Check if trace_data is a tuple with at least length 2
            if isinstance(trace_data, tuple) and len(trace_data) >= 2:
                # Check if the second element is a list (channels)
                if len(trace_data) > 2 and isinstance(trace_data[2], list):
                    channels = trace_data[2]
                    netsta = extract_netsta(channels)
                    if netsta:
                        unique_stations.add(netsta)
                # If we don't have channel info, try to use any station identifier in the data
                elif hasattr(trace_data[0], '__iter__') and len(trace_data[0]) > 0:
                    try:
                        unique_stations.add(str(trace_data[0]))
                    except:
                        pass
        
        return len(unique_stations)
    except KeyError:
        return 0

def generate_channel_performance_plots():
    """
    Generate performance metric plots broken down by channel type.
    
    Creates a 2x3 grid of plots showing performance metrics for each channel type.
    Uses the updated data structure to calculate precision correctly.
    """
    print("\nGenerating channel performance plots...")
    
    # Setup figure with 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey=True)
    
    # Channel types and their display names
    channel_types = [
        ('strong_motion', 'Strong Motion\n(HNE/HNN/HNZ or ENE/ENN/ENZ)'),
        ('4_channel', '4 Channel\n((ENE+ENN or HNE+HNN) + EHZ)'),
        ('short_period_3c', 'Short Period 3 Component\n(EHE/EHN/EHZ)'),
        ('short_period_vertical', 'Short Period Vertical\n(EHZ Only)'),
        ('broadband', 'Broadband\n(BHE/BHN/BHZ or HHE/HHN/HHZ)'),
        ('all', 'All Channels\nCombined')
    ]
    
    # Find global y-max for consistent scaling
    y_max = 0
    
    # Calculate metrics for each channel type and find max value
    channel_metrics = {}
    for channel_type, _ in channel_types:
        channel_metrics[channel_type] = {}
        
        # Get all models that have data for this channel type
        models = []
        for model in sorted(trace_channel_results[channel_type].keys()):
            # Check if there's any data for this model
            if (any(trace_channel_results[channel_type][model]["correct"].values()) or 
                any(trace_channel_results[channel_type][model]["incorrect"].values())):
                models.append(model)
        
        if not models:
            continue  # Skip if no models have data for this channel type
        
        # Calculate metrics for each model
        for model in models:
            # Initialize metrics dictionary for this model
            channel_metrics[channel_type][model] = {
                'precision': {},
                'recall': {},
                'f1': {},
                'accuracy': 0,
                'macro_precision': 0,
                'macro_recall': 0,
                'macro_f1': 0
            }
            
            # Calculate per-class metrics
            for event_type in event_types:
                # Get counts for precision calculation (true positives and false positives)
                true_positives = len(trace_predictions[channel_type][model]["true_positive"].get(event_type, []))
                false_positives = len(trace_predictions[channel_type][model]["false_positive"].get(event_type, []))
                
                # Get counts for recall calculation (true positives and false negatives)
                false_negatives = len(trace_predictions[channel_type][model]["false_negative"].get(event_type, []))
                
                # Calculate total predictions of this class
                total_actual = true_positives + false_negatives
                
                # Calculate precision
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate recall
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store metrics
                channel_metrics[channel_type][model]['precision'][event_type] = precision
                channel_metrics[channel_type][model]['recall'][event_type] = recall
                channel_metrics[channel_type][model]['f1'][event_type] = f1
            
            # Calculate overall accuracy
            total_correct = sum(len(data) for data in trace_channel_results[channel_type][model]["correct"].values())
            total_incorrect = sum(len(data) for data in trace_channel_results[channel_type][model]["incorrect"].values())
            total = total_correct + total_incorrect
            
            accuracy = total_correct / total if total > 0 else 0
            channel_metrics[channel_type][model]['accuracy'] = accuracy
            
            # Calculate macro-averaged metrics
            channel_metrics[channel_type][model]['macro_precision'] = (
                sum(channel_metrics[channel_type][model]['precision'].values()) / 
                len(channel_metrics[channel_type][model]['precision']) 
                if channel_metrics[channel_type][model]['precision'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_recall'] = (
                sum(channel_metrics[channel_type][model]['recall'].values()) / 
                len(channel_metrics[channel_type][model]['recall']) 
                if channel_metrics[channel_type][model]['recall'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_f1'] = (
                sum(channel_metrics[channel_type][model]['f1'].values()) / 
                len(channel_metrics[channel_type][model]['f1']) 
                if channel_metrics[channel_type][model]['f1'] else 0
            )
            
            # Update global y-max
            y_max = max(y_max, 
                        channel_metrics[channel_type][model]['macro_precision'],
                        channel_metrics[channel_type][model]['macro_recall'],
                        channel_metrics[channel_type][model]['macro_f1'],
                        channel_metrics[channel_type][model]['accuracy'])
    
    # Add a small buffer to y_max
    y_max = min(1.0, y_max * 1.1)

    # Now plot each channel type in its grid position
    for i, (channel_type, display_name) in enumerate(channel_types):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        models = list(sorted(channel_metrics.get(channel_type, {}).keys()))
        
        if not models:
            ax.text(0.5, 0.5, f"No data for {display_name}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name)
            continue
        
        # Calculate total trace count for this channel type
        total_trace_count = 0
        for model in models:
            # Get the count of unique stations
            model_trace_count = calculate_channel_trace_count(trace_channel_results, channel_type, model)
            total_trace_count += model_trace_count
        
        # Set up bar positions
        x = np.arange(len(models))
        bar_width = 0.2
        
        # Plot bars for each metric
        precision_bars = ax.bar(x - 1.5*bar_width, 
                                [channel_metrics[channel_type][model]['macro_precision'] for model in models],
                                bar_width, label='Precision', color='skyblue')
        
        recall_bars = ax.bar(x - 0.5*bar_width, 
                            [channel_metrics[channel_type][model]['macro_recall'] for model in models],
                            bar_width, label='Recall', color='lightgreen')
        
        f1_bars = ax.bar(x + 0.5*bar_width, 
                        [channel_metrics[channel_type][model]['macro_f1'] for model in models],
                        bar_width, label='F1', color='salmon')
        
        accuracy_bars = ax.bar(x + 1.5*bar_width, 
                              [channel_metrics[channel_type][model]['accuracy'] for model in models],
                              bar_width, label='Accuracy', color='mediumpurple')
        
        # Add text labels on top of the bars
        bar_label_fontsize = 7
        
        # Function to add formatted labels to bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only add label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        # Add labels to all bar sets
        add_labels(precision_bars)
        add_labels(recall_bars)
        add_labels(f1_bars)
        add_labels(accuracy_bars)
        
        # Customize the plot
        ax.set_title(f"{display_name}\nn={total_trace_count} stations")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        if col == 0:
            ax.set_ylabel('Score')
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set main title and adjust layout
    plt.suptitle('Performance Metrics by Channel Type', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'channel_performance_fixed.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved fixed channel performance plot to {output_path}")




"""
Functions to create probability vs. SNR scatter plots and heatmaps for each model.

Each figure will have 4 subplots:
1. Trace-wise probability of event class with highest probability vs SNR (colored by event class)
2-4. Trace-wise probability of each of the 3 event classes vs SNR (color scheme the same as the first subplot)

The functions will generate both scatter plots and heatmaps.
"""

def generate_probability_vs_snr_scatter(raw_trace_data, raw_event_data):
    """
    Generate scatter plots of probability vs. SNR for each model.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nGenerating probability vs. SNR scatter plots...")
    
    # Create a data structure to hold probabilities and SNR values
    prob_snr_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> event_type -> "prob"/"snr" -> list of values
    
    # Create a mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Process trace data to extract probabilities and SNR
    for evid, model_data in raw_trace_data.items():
        # Skip if we don't have metadata for this event
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            for trace_data in traces:
                # Extract data safely, handling potential missing pieces
                if len(trace_data) >= 3:
                    trace_index = trace_data[0]
                    probs = trace_data[1]
                    
                    # Extract SNR value - it's in the trace_data structure
                    # In the original data, SNR is the 9th column in the PROBS line
                    # At this point, SNR would be included in the trace_data if it was parsed correctly
                    snr = None
                    if len(trace_data) > 2 and isinstance(trace_data[2], list) and len(trace_data[2]) > 0:
                        # If the third element is already parsed channels
                        channels = trace_data[2]
                        # Try to get SNR from elsewhere in the structure if available
                        # This might need to be adjusted based on exactly how the data is structured
                        if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                            snr = float(trace_data[3])
                    elif len(trace_data) > 2 and isinstance(trace_data[2], (int, float)):
                        # If the third element is the SNR value directly
                        snr = float(trace_data[2])
                    
                    # If we couldn't find SNR, try another approach or skip this trace
                    if snr is None:
                        continue
                    
                    # Extract probabilities for each class
                    eq_prob = probs[0]
                    ex_prob = probs[1]
                    su_prob = probs[3]  # Skipping index 2 which is noise
                    
                    # Find class with highest probability
                    class_probs = [eq_prob, ex_prob, su_prob]
                    highest_prob = max(class_probs)
                    highest_prob_index = class_probs.index(highest_prob)
                    highest_prob_class = event_types[highest_prob_index]
                    
                    # Store data for plotting
                    prob_snr_data[model]["eq"]["prob"].append(eq_prob)
                    prob_snr_data[model]["eq"]["snr"].append(snr)
                    prob_snr_data[model]["ex"]["prob"].append(ex_prob)
                    prob_snr_data[model]["ex"]["snr"].append(snr)
                    prob_snr_data[model]["su"]["prob"].append(su_prob)
                    prob_snr_data[model]["su"]["snr"].append(snr)
                    
                    # Also store the highest probability data separately
                    prob_snr_data[model]["highest"]["prob"].append(highest_prob)
                    prob_snr_data[model]["highest"]["snr"].append(snr)
                    prob_snr_data[model]["highest"]["class"].append(highest_prob_class)
    
    # Generate scatter plots for each model
    for model in sorted(prob_snr_data.keys()):
        # Create figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        plt.suptitle(f'Probability vs. SNR Scatter Plots for {model}')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot highest probability vs SNR (first subplot)
        ax = axes[0]
        for event_type in event_types:
            # Extract data points where this event type had the highest probability
            indices = [i for i, cls in enumerate(prob_snr_data[model]["highest"]["class"]) if cls == event_type]
            snr_values = [prob_snr_data[model]["highest"]["snr"][i] for i in indices]
            prob_values = [prob_snr_data[model]["highest"]["prob"][i] for i in indices]
            
            ax.scatter(snr_values, prob_values, alpha=0.6, color=COLORS[event_type], 
                     label=f"{event_type.upper()} (n={len(indices)})")
        
        ax.set_title('Highest Probability vs. SNR')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot individual event type probabilities vs SNR (subplots 2-4)
        for i, event_type in enumerate(event_types):
            ax = axes[i+1]
            
            snr_values = prob_snr_data[model][event_type]["snr"]
            prob_values = prob_snr_data[model][event_type]["prob"]
            
            ax.scatter(snr_values, prob_values, alpha=0.6, color=COLORS[event_type], 
                     label=f"{event_type.upper()} (n={len(snr_values)})")
            
            ax.set_title(f'{event_type.upper()} Probability vs. SNR')
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for title
        
        output_path = os.path.join(OUTPUT_DIR, f'probability_vs_snr_scatter_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved probability vs. SNR scatter plot for {model} to {output_path}")

def generate_probability_vs_snr_heatmap(raw_trace_data, raw_event_data):
    """
    Generate heatmaps of probability vs. SNR for each model.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nGenerating probability vs. SNR heatmaps...")
    
    # Create a data structure to hold probabilities and SNR values
    prob_snr_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Create a mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Process trace data to extract probabilities and SNR
    for evid, model_data in raw_trace_data.items():
        # Skip if we don't have metadata for this event
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            for trace_data in traces:
                # Extract data safely, handling potential missing pieces
                if len(trace_data) >= 3:
                    trace_index = trace_data[0]
                    probs = trace_data[1]
                    
                    # Extract SNR value
                    snr = None
                    if len(trace_data) > 2 and isinstance(trace_data[2], list) and len(trace_data[2]) > 0:
                        # If the third element is already parsed channels
                        channels = trace_data[2]
                        # Try to get SNR from elsewhere in the structure if available
                        if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                            snr = float(trace_data[3])
                    elif len(trace_data) > 2 and isinstance(trace_data[2], (int, float)):
                        # If the third element is the SNR value directly
                        snr = float(trace_data[2])
                    
                    # If we couldn't find SNR, try another approach or skip this trace
                    if snr is None:
                        continue
                    
                    # Extract probabilities for each class
                    eq_prob = probs[0]
                    ex_prob = probs[1]
                    su_prob = probs[3]  # Skipping noise
                    
                    # Find class with highest probability
                    class_probs = [eq_prob, ex_prob, su_prob]
                    highest_prob = max(class_probs)
                    highest_prob_index = class_probs.index(highest_prob)
                    highest_prob_class = event_types[highest_prob_index]
                    
                    # Store data for plotting
                    prob_snr_data[model]["eq"]["prob"].append(eq_prob)
                    prob_snr_data[model]["eq"]["snr"].append(snr)
                    prob_snr_data[model]["ex"]["prob"].append(ex_prob)
                    prob_snr_data[model]["ex"]["snr"].append(snr)
                    prob_snr_data[model]["su"]["prob"].append(su_prob)
                    prob_snr_data[model]["su"]["snr"].append(snr)
                    
                    # Also store the highest probability data separately
                    prob_snr_data[model]["highest"]["prob"].append(highest_prob)
                    prob_snr_data[model]["highest"]["snr"].append(snr)
                    prob_snr_data[model]["highest"]["class"].append(highest_prob_class)
    
    # For each model, generate a heatmap
    for model in sorted(prob_snr_data.keys()):
        # Calculate global min and max SNR for consistent binning
        all_snr_values = []
        for event_type in list(event_types) + ["highest"]:
            all_snr_values.extend(prob_snr_data[model][event_type]["snr"])
        
        if not all_snr_values:  # Skip if no data for this model
            print(f"No SNR data available for {model}. Skipping heatmap.")
            continue
        
        min_snr = min(all_snr_values)
        max_snr = max(all_snr_values)
        
        # Create bins with 0.1 increments for SNR (as requested)
        snr_bins = np.arange(min_snr, max_snr + 0.1, 0.1)
        
        # Create probability bins with 0.01 increments (as requested)
        prob_bins = np.arange(0, 1.01, 0.01)
        
        # Create figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        plt.suptitle(f'Probability vs. SNR Heatmaps for {model}')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Track max count for colorbar scaling
        max_count = 0
        im = None  # Reference to the last image for colorbar
        
        # Create heatmaps for each subplot
        for i, event_type in enumerate(list(event_types) + ["highest"]):
            ax = axes[i]
            
            if event_type == "highest":
                # For highest probability subplot, process by event class
                for et in event_types:
                    # Extract data points where this event type had the highest probability
                    indices = [i for i, cls in enumerate(prob_snr_data[model]["highest"]["class"]) if cls == et]
                    if not indices:  # Skip if no data for this event type
                        continue
                        
                    snr_values = [prob_snr_data[model]["highest"]["snr"][i] for i in indices]
                    prob_values = [prob_snr_data[model]["highest"]["prob"][i] for i in indices]
                    
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(
                        snr_values, prob_values, 
                        bins=[snr_bins, prob_bins]
                    )
                    
                    # Update max count for colorbar scaling
                    max_count = max(max_count, np.max(H)) if H.size > 0 else max_count
                    
                    # Plot heatmap
                    im = ax.imshow(
                        H.T, interpolation='nearest', origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        aspect='auto', cmap=plt.cm.YlOrRd, alpha=0.7,
                        norm=LogNorm(vmin=1, vmax=max(1, max_count))  # Using log scale for better visualization
                    )
                    
                    # Add contour lines to identify event type regions
                    if np.sum(H) > 0:  # Only add contours if there's data
                        ax.contour(
                            H.T, levels=[0.5*np.max(H)], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            colors=[COLORS[et]], linewidths=2, alpha=0.8
                        )
                
                ax.set_title('Highest Probability vs. SNR')
            else:
                # For individual event type subplots
                snr_values = prob_snr_data[model][event_type]["snr"]
                prob_values = prob_snr_data[model][event_type]["prob"]
                
                if not snr_values:  # Skip if no data for this event type
                    ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{event_type.upper()} Probability vs. SNR')
                    continue
                
                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(
                    snr_values, prob_values, 
                    bins=[snr_bins, prob_bins]
                )
                
                # Update max count for colorbar scaling
                max_count = max(max_count, np.max(H)) if H.size > 0 else max_count
                
                # Plot heatmap
                im = ax.imshow(
                    H.T, interpolation='nearest', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap=plt.cm.YlOrRd,
                    norm=LogNorm(vmin=1, vmax=max(1, max_count))  # Using log scale for better visualization
                )
                
                ax.set_title(f'{event_type.upper()} Probability vs. SNR')
            
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('Probability')
            ax.grid(alpha=0.3)
            
            # Add legend for the highest probability subplot
            if event_type == "highest":
                handles = [plt.Line2D([0], [0], color=COLORS[et], lw=2) for et in event_types]
                labels = [f"{et.upper()}" for et in event_types]
                ax.legend(handles, labels, loc='upper right')
        
        # Add colorbar if we have valid data
        if im is not None:
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Count')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.92, top=0.92)  # Make room for colorbar and title
        
        output_path = os.path.join(OUTPUT_DIR, f'probability_vs_snr_heatmap_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved probability vs. SNR heatmap for {model} to {output_path}")

def generate_probability_vs_snr_heatmap_fixed(raw_trace_data, raw_event_data):
    """
    Generate heatmaps of probability vs. SNR for each model.
    Fixed to only show highest probability class for each trace and use correct spacing.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nGenerating updated probability vs. SNR heatmaps...")
    
    # Create a data structure to hold probabilities and SNR values
    prob_snr_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Process trace data and count unique stations
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        for model, traces in model_data.items():
            # Count unique stations in traces
            unique_station_count = count_unique_stations(traces)
            
            # Process each trace
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Extract probabilities for each class
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Find class with highest probability
                class_probs = [eq_prob, ex_prob, su_prob]
                highest_prob = max(class_probs)
                highest_prob_index = class_probs.index(highest_prob)
                highest_prob_class = event_types[highest_prob_index]
                
                # Get channel information
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                    # Extract NET.STA
                    netsta = extract_netsta(channels)
                    if not netsta:
                        continue  # Skip if couldn't extract NET.STA
                else:
                    continue  # Skip if no channel info
                
                # Store data only for highest probability class
                prob_snr_data[model]["highest"]["prob"].append(highest_prob)
                prob_snr_data[model]["highest"]["snr"].append(snr)
                prob_snr_data[model]["highest"]["class"].append(highest_prob_class)
                
                # Store data for the highest probability class subplot
                prob_snr_data[model][highest_prob_class]["prob"].append(highest_prob)
                prob_snr_data[model][highest_prob_class]["snr"].append(snr)
    
    # For each model, generate a heatmap
    for model in sorted(prob_snr_data.keys()):
        # Skip if no highest probability data
        if "highest" not in prob_snr_data[model] or not prob_snr_data[model]["highest"]["prob"]:
            print(f"Skipping probability vs. SNR heatmap for {model} - no data available")
            continue
        
        # Count total traces
        total_trace_count = len(prob_snr_data[model]["highest"]["prob"])
        
        # Calculate global min and max SNR for consistent binning
        all_snr_values = prob_snr_data[model]["highest"]["snr"]
        
        min_snr = min(all_snr_values) if all_snr_values else 0
        max_snr = max(all_snr_values) if all_snr_values else 30
        
        # Create bins with 0.25 increments for SNR (as requested)
        snr_bins = np.arange(0, SNR_SATURATE + SNR_BIN_WIDTH, SNR_BIN_WIDTH)
        
        # Create probability bins with 0.025 increments (as requested)
        prob_bins = np.arange(0, 1.01, PROB_BIN_WIDTH)
        
        # Create figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        plt.suptitle(f'Probability vs. SNR Heatmaps for {model} (n={total_trace_count})')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Track max count for colorbar scaling
        max_count = 0
        im = None  # Reference to the last image for colorbar
        
        # Create heatmaps for each subplot
        for i, event_type in enumerate(list(event_types) + ["highest"]):
            ax = axes[i]
            
            if event_type == "highest":
                # For highest probability subplot, separate by event class
                heatmap_data = {}
                
                for et in event_types:
                    # Extract data points where this event type had the highest probability
                    indices = [i for i, cls in enumerate(prob_snr_data[model]["highest"]["class"]) if cls == et]
                    if not indices:  # Skip if no data for this event type
                        continue
                        
                    snr_values = [prob_snr_data[model]["highest"]["snr"][i] for i in indices]
                    prob_values = [prob_snr_data[model]["highest"]["prob"][i] for i in indices]
                    
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(
                        snr_values, prob_values, 
                        bins=[snr_bins, prob_bins]
                    )
                    
                    # Store histogram data
                    heatmap_data[et] = {
                        'H': H,
                        'xedges': xedges,
                        'yedges': yedges
                    }
                    
                    # Update max count for colorbar scaling
                    max_count = max(max_count, np.max(H)) if H.size > 0 else max_count
                
                # Plot each event type with its own color
                for et in event_types:
                    if et in heatmap_data:
                        H = heatmap_data[et]['H']
                        xedges = heatmap_data[et]['xedges']
                        yedges = heatmap_data[et]['yedges']
                        
                        im = ax.pcolormesh(
                            xedges, yedges, H.T, 
                            cmap='hot', alpha=0.7,
                            norm=LogNorm(vmin=1, vmax=max(1, max_count))
                        )
                
                ax.set_title(f'Highest Probability vs. SNR (n={total_trace_count})')
            else:
                # For individual event type subplots
                if event_type in prob_snr_data[model] and prob_snr_data[model][event_type]["snr"]:
                    snr_values = prob_snr_data[model][event_type]["snr"]
                    prob_values = prob_snr_data[model][event_type]["prob"]
                    
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(
                        snr_values, prob_values, 
                        bins=[snr_bins, prob_bins]
                    )
                    
                    # Update max count for colorbar scaling
                    max_count = max(max_count, np.max(H)) if H.size > 0 else max_count
                    
                    # Plot heatmap
                    im = ax.pcolormesh(
                        xedges, yedges, H.T, 
                        cmap='hot',
                        norm=LogNorm(vmin=1, vmax=max(1, max_count))
                    )
                    
                    ax.set_title(f'{event_type.upper()} Probability vs. SNR (n={len(snr_values)})')
                else:
                    ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{event_type.upper()} Probability vs. SNR (n=0)')
            
            ax.set_xlabel('SNR')
            ax.set_ylabel('Probability')
            ax.grid(alpha=0.3)
        
        # Add colorbar if we have valid data
        if im is not None:
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Count')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.92, top=0.92)  # Make room for colorbar and title
        
        output_path = os.path.join(OUTPUT_DIR, f'probability_vs_snr_heatmap_fixed_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved updated probability vs. SNR heatmap for {model} to {output_path}")

# -----------------------------------------------------------------------------
# Distance vs Probability/Probability Distance Plots
# -----------------------------------------------------------------------------
def generate_distance_vs_probability_plots(raw_trace_data, raw_event_data, distances=None):
    """
    Create heatmaps of distance vs probability and distance vs probability distance.
    Use 2x2 subplots (EQ, EX, SU, All) and reverse colorbar.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating distance vs probability plots...")
    
    if distances is None:
        distances = load_distances()
    
    if not distances:
        print("No distance data available. Skipping distance vs probability plots.")
        return
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Collect data for distance vs probability and distance vs probability distance
    dist_prob_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    # model -> "max_prob"/"prob_dist" -> event_type -> "distance"/"value" -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            # Count unique stations in traces
            processed_stations = set()
            
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if (evid, netsta) in processed_stations:
                    continue
                
                processed_stations.add((evid, netsta))
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Extract probabilities for event types
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Get max probability and probability distance
                class_probs = [eq_prob, ex_prob, su_prob]
                sorted_probs = sorted(class_probs, reverse=True)
                max_prob = sorted_probs[0]
                prob_dist = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                
                # Store data by event type (analyst label)
                dist_prob_data[model]["max_prob"][analyst]["distance"].append(distance)
                dist_prob_data[model]["max_prob"][analyst]["value"].append(max_prob)
                dist_prob_data[model]["prob_dist"][analyst]["distance"].append(distance)
                dist_prob_data[model]["prob_dist"][analyst]["value"].append(prob_dist)
                
                # Store for "all" category
                dist_prob_data[model]["max_prob"]["all"]["distance"].append(distance)
                dist_prob_data[model]["max_prob"]["all"]["value"].append(max_prob)
                dist_prob_data[model]["prob_dist"]["all"]["distance"].append(distance)
                dist_prob_data[model]["prob_dist"]["all"]["value"].append(prob_dist)
    
    # Get list of models for visualization
    models = sorted(dist_prob_data.keys())
    
    if not models:
        print("No data with both distance and probability information. Skipping distance vs probability plots.")
        return
    
    # Event categories for subplots
    event_categories = list(event_types) + ["all"]
    
    # Create heatmaps for each model
    for model in models:
        # Create figures for max probability and probability distance
        for plot_type in ["max_prob", "prob_dist"]:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
            axes = axes.flatten()
            
            if plot_type not in dist_prob_data[model]:
                plt.close(fig)
                continue
            
            # Define bins
            distance_bins = np.arange(0, DISTKM_SATURATE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH)
            value_bins = np.arange(0, 1.01, PROB_BIN_WIDTH)
            
            # Track max count for colorbar
            max_count = 0
            
            # Plot each event type in its own subplot
            for i, event_type in enumerate(event_categories):
                ax = axes[i]
                
                if event_type not in dist_prob_data[model][plot_type] or not dist_prob_data[model][plot_type][event_type]["distance"]:
                    ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                           ha='center', va='center', transform=ax.transAxes)
                    if plot_type == "max_prob":
                        ax.set_title(f"{event_type.upper()} Maximum Probability")
                    else:  # prob_dist
                        ax.set_title(f"{event_type.upper()} Probability Distance")
                    continue
                
                # Get data
                distances = dist_prob_data[model][plot_type][event_type]["distance"]
                values = dist_prob_data[model][plot_type][event_type]["value"]
                
                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(
                    distances, values, 
                    bins=[distance_bins, value_bins]
                )
                
                # Update max count
                max_count = max(max_count, np.max(H)) if H.size > 0 else max_count
                
                # Plot heatmap with reversed colormap (white=1, red=max)
                im = ax.pcolormesh(
                    xedges, yedges, H.T, 
                    cmap=plt.cm.hot_r,
                    norm=LogNorm(vmin=1, vmax=max(1, np.max(H)))
                )
                
                # Add title
                if plot_type == "max_prob":
                    ax.set_title(f"{event_type.upper()} Maximum Probability (n={len(distances)})")
                else:  # prob_dist
                    ax.set_title(f"{event_type.upper()} Probability Distance (n={len(distances)})")
                
                # Add labels
                ax.set_xlabel("Distance (km)")
                ax.set_ylabel("Value")
                
                # Add grid
                ax.grid(alpha=0.3)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=axes)
            
            # Format colorbar tick labels for logarithmic scale
            if isinstance(im.norm, LogNorm):
                import matplotlib.ticker as ticker
                cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                cbar.ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
            
            cbar.set_label("Count")
            
            # Add overall title
            if plot_type == "max_prob":
                plt.suptitle(f"Distance vs Maximum Probability for {model}", fontsize=16)
                output_path = os.path.join(OUTPUT_DIR, f"distance_vs_maxprob_{model}.png")
            else:  # prob_dist
                plt.suptitle(f"Distance vs Probability Distance for {model}", fontsize=16)
                output_path = os.path.join(OUTPUT_DIR, f"distance_vs_probdist_{model}.png")
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure
            plt.savefig(output_path, dpi=OUTPUT_DPI)
            plt.close(fig)
            print(f"Saved {plot_type} vs distance plot for {model} to {output_path}")

"""
Implementation of the small multiples grid as a heatmap using median probability and probability distance.
Only showing grid points with at least 10 traces.
"""

def generate_small_multiples_heatmap(raw_trace_data, raw_event_data, distances=None):
    """
    Create small multiples grid as a heatmap showing median probability and probability distance.
    Only shows cells with at least 10 traces and ignores correctness of predictions.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating small multiples heatmap visualization...")
    
    if distances is None:
        distances = load_distances()
    
    if not distances:
        print("No distance data available. Skipping small multiples heatmap.")
        return
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Collect data for each model and event type
    # For both probability and probability distance
    grid_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    # model -> "prob"/"prob_dist" -> event_type -> (snr_bin, dist_bin) -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            # Process each unique station only once
            processed_stations = set()
            
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if (evid, netsta) in processed_stations:
                    continue
                
                processed_stations.add((evid, netsta))
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Extract probabilities for event types
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Find class with highest probability
                class_probs = [eq_prob, ex_prob, su_prob]
                sorted_probs = sorted(class_probs, reverse=True)
                highest_prob = sorted_probs[0]
                prob_dist = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                highest_prob_index = class_probs.index(highest_prob)
                highest_prob_class = event_types[highest_prob_index]
                
                # Bin SNR and distance
                snr_bin = int(snr / 2) * 2  # 2 dB bins
                dist_bin = int(distance / 10) * 10  # 10 km bins
                bin_key = (snr_bin, dist_bin)
                
                # Store data by true event type (analyst label)
                # For both probability and probability distance
                # This is agnostic to whether the prediction is correct or incorrect
                grid_data[model]["prob"][analyst][bin_key].append(highest_prob)
                grid_data[model]["prob_dist"][analyst][bin_key].append(prob_dist)
                
                # Also store for "all" category
                grid_data[model]["prob"]["all"][bin_key].append(highest_prob)
                grid_data[model]["prob_dist"]["all"][bin_key].append(prob_dist)
    
    # Get list of models for visualization
    models = sorted(grid_data.keys())
    
    if not models:
        print("No data with both SNR and distance information. Skipping small multiples heatmap.")
        return
    
    # Create two types of plots for each model
    for model in models:
        for plot_type in ["prob", "prob_dist"]:
            if plot_type not in grid_data[model]:
                continue
                
            # Create figure with subplots for each event type
            fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
            axes = axes.flatten()
            
            # Define event categories for subplots
            event_categories = list(event_types) + ["all"]
            
            # Get min/max values for y limits
            max_snr_bin = 0
            max_dist_bin = 0
            
            for event_type in event_categories:
                for bin_key in grid_data[model][plot_type][event_type]:
                    snr_bin, dist_bin = bin_key
                    max_snr_bin = max(max_snr_bin, snr_bin)
                    max_dist_bin = max(max_dist_bin, dist_bin)
            
            # Adjust for plotting
            max_snr_bin += 2
            max_dist_bin += 10
            
            # Create empty grids for each event type
            for i, event_type in enumerate(event_categories):
                ax = axes[i]
                
                if event_type not in grid_data[model][plot_type] or not grid_data[model][plot_type][event_type]:
                    ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{event_type.upper()}")
                    continue
                
                # Create grid for heatmap
                snr_values = range(0, max_snr_bin, 2)
                dist_values = range(0, max_dist_bin, 10)
                
                # Initialize grid with NaNs
                grid = np.full((len(snr_values), len(dist_values)), np.nan)
                count_grid = np.zeros((len(snr_values), len(dist_values)), dtype=int)
                
                # Fill grid with median probability or probability distance
                for bin_key, values in grid_data[model][plot_type][event_type].items():
                    if len(values) < 10:  # Skip bins with fewer than 10 traces
                        continue
                        
                    snr_bin, dist_bin = bin_key
                    snr_idx = snr_bin // 2  # Convert to index
                    dist_idx = dist_bin // 10  # Convert to index
                    
                    if 0 <= snr_idx < len(snr_values) and 0 <= dist_idx < len(dist_values):
                        # Calculate median
                        median_value = np.median(values)
                        grid[snr_idx, dist_idx] = median_value
                        count_grid[snr_idx, dist_idx] = len(values)
                
                # Create meshgrid for pcolormesh
                X, Y = np.meshgrid(dist_values, snr_values)
                
                # Plot heatmap
                im = ax.pcolormesh(X, Y, grid, cmap='viridis', vmin=0, vmax=1)
                
                # Add counts as text
                for j in range(len(snr_values)):
                    for k in range(len(dist_values)):
                        count = count_grid[j, k]
                        if count >= 10:  # Only show text for cells with data
                            value = grid[j, k]
                            ax.text(dist_values[k] + 5, snr_values[j] + 1, 
                                   f"{value:.2f}\nn={count}", 
                                   ha='center', va='center', fontsize=8,
                                   color='white' if value > 0.5 else 'black')
                
                ax.set_title(f"{event_type.upper()} (n={sum(len(values) for values in grid_data[model][plot_type][event_type].values())})")
                ax.set_xlabel("Distance (km)")
                ax.set_ylabel("SNR")
                
                # Format the highest SNR tick
                if max_snr_bin <= SNR_SATURATE:
                    yticks = ax.get_yticks()
                    if yticks[-1] >= SNR_SATURATE:
                        yticklabels = [f"{y:.0f}" for y in yticks[:-1]] + [f">{SNR_SATURATE:.0f}"]
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(yticklabels)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=axes)
            if plot_type == "prob":
                cbar.set_label("Median Probability")
                title = f"Median Maximum Probability by SNR and Distance for {model}"
            else:  # prob_dist
                cbar.set_label("Median Probability Distance")
                title = f"Median Probability Distance by SNR and Distance for {model}"
            
            # Add title
            plt.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure
            output_path = os.path.join(OUTPUT_DIR, f"small_multiples_heatmap_{plot_type}_{model}.png")
            plt.savefig(output_path, dpi=OUTPUT_DPI)
            plt.close(fig)
            print(f"Saved small multiples heatmap ({plot_type}) for {model} to {output_path}")

# -----------------------------------------------------------------------------
# Heatmap with Bubble Size
# -----------------------------------------------------------------------------
def generate_heatmap_with_bubbles(raw_trace_data, raw_event_data, distances=None):
    """
    Create heatmap of SNR vs Distance, with bubble size indicating probability.
    Use darker color scale and add legend showing circle sizes.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating heatmap with bubbles visualization...")
    
    if distances is None:
        distances = load_distances()
    
    if not distances:
        print("No distance data available. Skipping heatmap with bubbles.")
        return
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Collect data for each model and event type
    bubble_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> event_type -> "snr"/"distance"/"prob" -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            # Process each unique station only once
            processed_stations = set()
            
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if (evid, netsta) in processed_stations:
                    continue
                
                processed_stations.add((evid, netsta))
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Extract probabilities for event types
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Get highest probability for coloring
                class_probs = [eq_prob, ex_prob, su_prob]
                highest_prob = max(class_probs)
                highest_prob_index = class_probs.index(highest_prob)
                highest_prob_class = event_types[highest_prob_index]
                
                # Store data for true event type (analyst)
                bubble_data[model][analyst]["snr"].append(snr)
                bubble_data[model][analyst]["distance"].append(distance)
                bubble_data[model][analyst]["prob"].append(highest_prob)
                
                # Also store data by predicted class
                bubble_data[model][highest_prob_class]["predicted_snr"].append(snr)
                bubble_data[model][highest_prob_class]["predicted_distance"].append(distance)
                bubble_data[model][highest_prob_class]["predicted_prob"].append(highest_prob)
                
                # Store for "all" category
                bubble_data[model]["all"]["snr"].append(snr)
                bubble_data[model]["all"]["distance"].append(distance)
                bubble_data[model]["all"]["prob"].append(highest_prob)
    
    # Get list of models for visualization
    models = sorted(bubble_data.keys())
    
    if not models:
        print("No data with both SNR and distance information. Skipping heatmap with bubbles.")
        return
    
    # Create figure for each model
    for model in models:
        # Create subplots - one for each event type plus one for all combined
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Define event categories for subplots
        event_categories = list(event_types) + ["all"]
        
        # Define bins for heatmap
        distance_bins = np.arange(0, DISTKM_SATURATE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH)
        snr_bins = np.arange(0, SNR_SATURATE + 1, 1)
        
        # Define bubble scaling factor
        max_bubble_size = 300
        
        # Define example bubble sizes for legend
        example_counts = [5, 20, 50]
        example_probs = [0.5, 0.75, 0.95]
        
        # For legend
        legend_bubbles = []
        legend_labels = []
        
        # Plot each event type
        for i, event_type in enumerate(event_categories):
            ax = axes[i]
            
            if event_type not in bubble_data[model] or not bubble_data[model][event_type]["snr"]:
                ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{event_type.upper()}")
                continue
            
            distances = bubble_data[model][event_type]["distance"]
            snrs = bubble_data[model][event_type]["snr"]
            probs = bubble_data[model][event_type]["prob"]
            
            # Create 2D histogram for density coloring
            H, xedges, yedges = np.histogram2d(
                distances, snrs, 
                bins=[distance_bins, snr_bins]
            )
            
            # Calculate avg probability for each bin
            avg_probs = np.zeros_like(H)
            counts = np.zeros_like(H)
            
            for dist, snr, prob in zip(distances, snrs, probs):
                # Find the bin indices
                x_idx = np.searchsorted(distance_bins, dist) - 1
                y_idx = np.searchsorted(snr_bins, snr) - 1
                
                if 0 <= x_idx < H.shape[0] and 0 <= y_idx < H.shape[1]:
                    avg_probs[x_idx, y_idx] += prob
                    counts[x_idx, y_idx] += 1
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_probs = np.divide(avg_probs, counts, out=np.zeros_like(avg_probs), where=counts > 0)
            
            # Plot heatmap with darker Blues colormap
            im = ax.imshow(
                H.T, interpolation='nearest', origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect='auto', cmap='Blues', alpha=0.5,
                norm=LogNorm(vmin=1, vmax=max(1, np.max(H)))
            )
            
            # Plot bubbles for each bin with non-zero count
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    if counts[i, j] > 0:
                        # Bubble position (center of bin)
                        x = (distance_bins[i] + distance_bins[i+1]) / 2
                        y = (snr_bins[j] + snr_bins[j+1]) / 2
                        
                        # Bubble size (proportional to count, scaled by avg prob)
                        size = np.sqrt(counts[i, j]) * max_bubble_size / np.sqrt(np.max(counts)) * avg_probs[i, j]
                        
                        # Plot bubble
                        ax.scatter(x, y, s=size, alpha=0.7, 
                                 c=avg_probs[i, j], cmap='viridis', vmin=0, vmax=1,
                                 edgecolors='black', linewidths=0.5)
            
            total_count = np.sum(counts)
            ax.set_title(f"{event_type.upper()} (n={total_count})")
            
            # Add labels
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("SNR")
            
            # Format the highest SNR tick
            if ax.get_yticks()[-1] == SNR_SATURATE:
                yticks = ax.get_yticks()
                yticklabels = [f"{y:.0f}" for y in yticks[:-1]] + [f">{yticks[-1]:.0f}"]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            
            # Add grid
            ax.grid(alpha=0.3)
        
        # Add colorbars
        density_cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
        density_cbar = fig.colorbar(im, cax=density_cbar_ax)
        density_cbar.set_label("Trace Count")
        
        # Format the density colorbar tick labels for logarithmic scale
        if isinstance(im.norm, LogNorm):
            import matplotlib.ticker as ticker
            density_cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            density_cbar.ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        
        # Create a sample point with value 1 for the probability colorbar
        prob_sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
        prob_sm.set_array([])
        
        prob_cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        prob_cbar = fig.colorbar(prob_sm, cax=prob_cbar_ax)
        prob_cbar.set_label("Probability")
        
        # Add bubble size legend in upper right
        # Create separate axis for bubble size legend
        legend_ax = fig.add_axes([0.92, 0.85, 0.06, 0.1], frameon=True)
        legend_ax.set_title("Bubble Size", fontsize=10)
        
        # Create example bubbles
        example_sizes = [10, 30, 60]  # Size in points
        example_counts = [5, 20, 50]  # What they represent
        
        for i, (size, count) in enumerate(zip(example_sizes, example_counts)):
            y_pos = 0.7 - i * 0.2
            legend_ax.scatter(0.3, y_pos, s=size, color='gray', edgecolor='black')
            legend_ax.text(0.6, y_pos, f"n={count}", va='center', fontsize=8)
        
        # Turn off axes for legend
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis('off')
        
        # Add title
        plt.suptitle(f"Distance vs SNR with Probability Bubbles: {model}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, top=0.93)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"heatmap_bubbles_{model}.png")
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved heatmap with bubbles for {model} to {output_path}")

"""
Implementation of dataset histograms showing SNR, distance, and magnitude by event class.
Creates two versions - one with fixed y-axis scaling across columns and one without.
"""

def generate_dataset_histograms(raw_trace_data, raw_event_data, distances=None):
    """
    Create simple histograms of SNR, distance, and magnitude by event class.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating dataset histograms...")
    
    if distances is None:
        distances = load_distances()
    
    # Create mapping from evid to analyst label and magnitude
    evid_metadata = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            # Use the first entry's analyst and magnitude
            evid_metadata[evid] = {
                "analyst": event_entries[0]["analyst"],
                "magnitude": event_entries[0]["magnitude"]
            }
    
    # Collect data by event type
    dataset_data = defaultdict(lambda: defaultdict(list))
    # event_type -> "snr"/"distance"/"magnitude" -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_metadata:
            continue
        
        analyst = evid_metadata[evid]["analyst"]
        magnitude = evid_metadata[evid]["magnitude"]
        
        # Count each station only once
        processed_stations = set()
        
        for model, traces in model_data.items():
            # Just use the first model's data
            if model != sorted(model_data.keys())[0]:
                continue
                
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if (evid, netsta) in processed_stations:
                    continue
                
                processed_stations.add((evid, netsta))
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Store data by event type (analyst label)
                dataset_data[analyst]["snr"].append(snr)
                dataset_data[analyst]["distance"].append(distance)
                dataset_data[analyst]["magnitude"].append(magnitude)
                
                # Store for "all" category
                dataset_data["all"]["snr"].append(snr)
                dataset_data["all"]["distance"].append(distance)
                dataset_data["all"]["magnitude"].append(magnitude)
    
    # Define event categories for subplots
    event_categories = list(event_types) + ["all"]
    
    # Define data types to plot
    data_types = ["magnitude", "distance", "snr"]
    data_labels = ["Magnitude", "Distance (km)", "SNR"]
    
    # Create two versions of the plot (with and without fixed y-axis)
    for fixed_yaxis in [True, False]:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=fixed_yaxis)
        
        # Find global max count for each row (if using fixed y-axis)
        if fixed_yaxis:
            max_counts = []
            for i, data_type in enumerate(data_types):
                max_count = 0
                for j, event_type in enumerate(event_categories):
                    if event_type in dataset_data and data_type in dataset_data[event_type]:
                        values = dataset_data[event_type][data_type]
                        if data_type == "magnitude":
                            bins = np.arange(-2, 8, 0.5)
                        elif data_type == "distance":
                            bins = np.arange(0, DISTKM_SATURATE + 10, 10)
                        elif data_type == "snr":
                            bins = np.arange(0, SNR_SATURATE + 2, 2)
                            
                        hist, bin_edges = np.histogram(values, bins=bins)
                        max_count = max(max_count, np.max(hist) if hist.size > 0 else 0)
                max_counts.append(max_count)
        
        # Plot histograms
        for i, data_type in enumerate(data_types):
            for j, event_type in enumerate(event_categories):
                ax = axes[i, j]
                
                if event_type in dataset_data and data_type in dataset_data[event_type]:
                    values = dataset_data[event_type][data_type]
                    
                    # Define bins based on data type
                    if data_type == "magnitude":
                        bins = np.arange(-2, 8, 0.5)
                    elif data_type == "distance":
                        bins = np.arange(0, DISTKM_SATURATE + 10, 10)
                    elif data_type == "snr":
                        bins = np.arange(0, SNR_SATURATE + 2, 2)
                    
                    # Plot histogram
                    color = COLORS.get(event_type, 'blue')
                    ax.hist(values, bins=bins, alpha=0.7, color=color)
                    
                    # Set title with count
                    ax.set_title(f"{event_type.upper()} (n={len(values)})")
                else:
                    ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{event_type.upper()}")
                
                # Set labels
                ax.set_xlabel(data_labels[i])
                if j == 0:
                    ax.set_ylabel("Count")
                
                # Set y-axis limit if using fixed y-axis
                if fixed_yaxis and max_counts[i] > 0:
                    ax.set_ylim(0, max_counts[i] * 1.1)
                
                # Format the highest SNR tick for SNR plots
                if data_type == "snr" and ax.get_xticks()[-1] == SNR_SATURATE:
                    xticks = ax.get_xticks()
                    xticklabels = [f"{x:.0f}" for x in xticks[:-1]] + [f">{xticks[-1]:.0f}"]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                
                # Add grid
                ax.grid(alpha=0.3)
        
        # Add title
        if fixed_yaxis:
            plt.suptitle("Dataset Characteristics by Event Type (Fixed Y-axis)", fontsize=16)
            output_path = os.path.join(OUTPUT_DIR, "dataset_histograms_fixed_yaxis.png")
        else:
            plt.suptitle("Dataset Characteristics by Event Type", fontsize=16)
            output_path = os.path.join(OUTPUT_DIR, "dataset_histograms.png")
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save figure
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved dataset histograms to {output_path}")

# -----------------------------------------------------------------------------
# Contour Plot Overlay
# -----------------------------------------------------------------------------
def generate_contour_overlay(raw_trace_data, raw_event_data, distances=None):
    """
    Create contour plots of SNR vs Distance with probability contours.
    Split into 4 subplots (EQ, EX, SU, All)
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating contour overlay visualization...")
    
    if distances is None:
        distances = load_distances()
    
    if not distances:
        print("No distance data available. Skipping contour overlay.")
        return
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Collect data for each model and event type
    contour_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> event_type -> "snr"/"distance"/"prob" -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            # Count unique stations in traces
            unique_station_count = count_unique_stations(traces)
            
            # Process each unique station only once
            processed_stations = set()
            
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if netsta in processed_stations:
                    continue
                
                processed_stations.add(netsta)
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Extract probabilities for event types
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Store data for each event type's probability
                contour_data[model]["eq"]["snr"].append(snr)
                contour_data[model]["eq"]["distance"].append(distance)
                contour_data[model]["eq"]["prob"].append(eq_prob)
                
                contour_data[model]["ex"]["snr"].append(snr)
                contour_data[model]["ex"]["distance"].append(distance)
                contour_data[model]["ex"]["prob"].append(ex_prob)
                
                contour_data[model]["su"]["snr"].append(snr)
                contour_data[model]["su"]["distance"].append(distance)
                contour_data[model]["su"]["prob"].append(su_prob)
                
                # Store for "all" category (average of all probabilities)
                contour_data[model]["all"]["snr"].append(snr)
                contour_data[model]["all"]["distance"].append(distance)
                contour_data[model]["all"]["prob"].append((eq_prob + ex_prob + su_prob) / 3)
    
    # Get list of models for visualization
    models = sorted(contour_data.keys())
    
    if not models:
        print("No data with both SNR and distance information. Skipping contour overlay.")
        return
    
    # Create figure for each model
    for model in models:
        # Create 2x2 subplots for each event type
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Define event types for subplots
        event_categories = list(event_types) + ["all"]
        
        # Define bins for gridding
        distance_bins = np.arange(0, DISTKM_SATURATE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH)
        snr_bins = np.arange(0, SNR_SATURATE + 1, 1)
        
        # Create a grid for contour interpolation
        X, Y = np.meshgrid(distance_bins, snr_bins)
        
        # Plot each event type in its own subplot
        for i, event_type in enumerate(event_categories):
            ax = axes[i]
            
            if event_type not in contour_data[model] or not contour_data[model][event_type]["distance"]:
                ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{event_type.upper()}")
                continue
                
            distances = contour_data[model][event_type]["distance"]
            snrs = contour_data[model][event_type]["snr"]
            probs = contour_data[model][event_type]["prob"]
            
            # Create a grid of average probabilities
            Z = np.zeros((len(snr_bins), len(distance_bins)))
            counts = np.zeros((len(snr_bins), len(distance_bins)))
            
            for dist, snr, prob in zip(distances, snrs, probs):
                # Find the bin indices
                x_idx = np.searchsorted(distance_bins, dist) - 1
                y_idx = np.searchsorted(snr_bins, snr) - 1
                
                if 0 <= x_idx < len(distance_bins)-1 and 0 <= y_idx < len(snr_bins)-1:
                    Z[y_idx, x_idx] += prob
                    counts[y_idx, x_idx] += 1
            
            # Average probabilities
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = np.divide(Z, counts, out=np.zeros_like(Z), where=counts > 0)
            
            # Plot the underlying heatmap first
            im = ax.pcolormesh(X, Y, Z, shading='auto', alpha=0.3, vmin=0, vmax=1, cmap='viridis')
            
            # Add contour lines
            contour_levels = [0.3, 0.5, 0.7, 0.9]
            CS = ax.contour(X, Y, Z, levels=contour_levels, colors=[COLORS.get(event_type, 'black')], linewidths=2)
            
            # Label contours
            ax.clabel(CS, inline=True, fontsize=10, fmt='%.1f')
            
            # Add event type title
            ax.set_title(f"{event_type.upper()} Probability (n={len(distances)})")
            
            # Add labels
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("SNR")
            ax.grid(alpha=0.3)
            
            # Format the highest SNR tick
            if ax.get_yticks()[-1] == SNR_SATURATE:
                yticks = ax.get_yticks()
                yticklabels = [f"{y:.0f}" for y in yticks[:-1]] + [f">{yticks[-1]:.0f}"]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes)
        cbar.set_label("Probability")
        
        # Add overall title
        plt.suptitle(f"Probability Contours for {model}", fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"contour_overlay_{model}.png")
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved contour overlay for {model} to {output_path}")

# -----------------------------------------------------------------------------
# Binned Statistics Plot
# -----------------------------------------------------------------------------
def generate_binned_statistics(raw_trace_data, raw_event_data, distances=None):
    """
    Create binned statistics plot of SNR vs Distance, showing mean probability.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
        distances: Dictionary mapping (evid, netsta) to distance
    """
    print("\nGenerating binned statistics visualization...")
    
    if distances is None:
        distances = load_distances()
    
    if not distances:
        print("No distance data available. Skipping binned statistics.")
        return
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Collect data for each model and event type
    binned_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> event_type -> "snr"/"distance"/"prob" -> list of values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Get distance
                distance = distances.get((evid, netsta))
                if distance is None:
                    continue  # Skip if no distance data
                
                # Extract probability for true event type (analyst)
                if analyst == "eq":
                    prob = probs[0]
                elif analyst == "ex":
                    prob = probs[1]
                elif analyst == "su":
                    prob = probs[3]  # Skipping noise
                else:
                    continue  # Skip if unknown event type
                
                # Store data
                binned_data[model][analyst]["snr"].append(snr)
                binned_data[model][analyst]["distance"].append(distance)
                binned_data[model][analyst]["prob"].append(prob)
    
    # Get list of models for visualization
    models = sorted(binned_data.keys())
    
    if not models:
        print("No data with both SNR and distance information. Skipping binned statistics.")
        return
    
    # Create figure for each model
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Define bins for statistics
        distance_bins = np.arange(0, DISTKM_SATURATE + DISTANCE_BIN_WIDTH, DISTANCE_BIN_WIDTH * 5)  # Larger bins for better statistics
        snr_bins = np.arange(0, SNR_SATURATE + 1, 2)  # Larger bins for better statistics
        
        # Combined plot first
        ax = axes[0]
        all_distances = []
        all_snrs = []
        all_probs = []
        all_counts = defaultdict(int)  # (dist_bin, snr_bin) -> count
        
        for event_type in event_types:
            if event_type in binned_data[model]:
                all_distances.extend(binned_data[model][event_type]["distance"])
                all_snrs.extend(binned_data[model][event_type]["snr"])
                all_probs.extend(binned_data[model][event_type]["prob"])
        
        if all_distances:
            # Create meshgrid for plotting
            X, Y = np.meshgrid(distance_bins, snr_bins)
            
            # Calculate statistics for each bin
            Z_mean = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
            Z_count = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
            Z_std = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
            
            for dist, snr, prob in zip(all_distances, all_snrs, all_probs):
                # Find bin indices
                dist_idx = np.searchsorted(distance_bins, dist) - 1
                snr_idx = np.searchsorted(snr_bins, snr) - 1
                
                if 0 <= dist_idx < len(distance_bins)-1 and 0 <= snr_idx < len(snr_bins)-1:
                    Z_mean[snr_idx, dist_idx] += prob
                    Z_count[snr_idx, dist_idx] += 1
                    all_counts[(dist_idx, snr_idx)] += 1
            
            # Calculate mean (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                Z_mean = np.divide(Z_mean, Z_count, out=np.zeros_like(Z_mean), where=Z_count > 0)
            
            # Calculate standard deviation
            for dist, snr, prob in zip(all_distances, all_snrs, all_probs):
                dist_idx = np.searchsorted(distance_bins, dist) - 1
                snr_idx = np.searchsorted(snr_bins, snr) - 1
                
                if 0 <= dist_idx < len(distance_bins)-1 and 0 <= snr_idx < len(snr_bins)-1:
                    if Z_count[snr_idx, dist_idx] > 0:
                        Z_std[snr_idx, dist_idx] += (prob - Z_mean[snr_idx, dist_idx]) ** 2
            
            # Finalize standard deviation calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                Z_std = np.sqrt(np.divide(Z_std, Z_count, out=np.zeros_like(Z_std), where=Z_count > 0))
            
            # Plot mean probability
            im = ax.pcolormesh(X[:-1, :-1], Y[:-1, :-1], Z_mean, cmap='viridis', vmin=0, vmax=1)
            
            # Add count information
            for i in range(len(distance_bins)-1):
                for j in range(len(snr_bins)-1):
                    if Z_count[j, i] > 0:
                        # Text for bin center
                        bin_center_x = (distance_bins[i] + distance_bins[i+1]) / 2
                        bin_center_y = (snr_bins[j] + snr_bins[j+1]) / 2
                        
                        # Format count and mean
                        count_text = f"n={int(Z_count[j, i])}"
                        mean_text = f"μ={Z_mean[j, i]:.2f}"
                        
                        # Plot text with white outline for visibility
                        text = ax.text(bin_center_x, bin_center_y, f"{count_text}\n{mean_text}", 
                                     ha='center', va='center', fontsize=8,
                                     color='white', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
            
            ax.set_title(f"All Event Types Combined (n={len(all_distances)})")
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("All Event Types Combined (n=0)")
        
        # Individual event type plots
        for i, event_type in enumerate(event_types):
            ax = axes[i+1]
            
            if event_type in binned_data[model] and binned_data[model][event_type]["distance"]:
                distances = binned_data[model][event_type]["distance"]
                snrs = binned_data[model][event_type]["snr"]
                probs = binned_data[model][event_type]["prob"]
                
                # Create meshgrid for plotting
                X, Y = np.meshgrid(distance_bins, snr_bins)
                
                # Calculate statistics for each bin
                Z_mean = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
                Z_count = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
                Z_std = np.zeros((len(snr_bins)-1, len(distance_bins)-1))
                
                for dist, snr, prob in zip(distances, snrs, probs):
                    # Find bin indices
                    dist_idx = np.searchsorted(distance_bins, dist) - 1
                    snr_idx = np.searchsorted(snr_bins, snr) - 1
                    
                    if 0 <= dist_idx < len(distance_bins)-1 and 0 <= snr_idx < len(snr_bins)-1:
                        Z_mean[snr_idx, dist_idx] += prob
                        Z_count[snr_idx, dist_idx] += 1
                
                # Calculate mean (avoid division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    Z_mean = np.divide(Z_mean, Z_count, out=np.zeros_like(Z_mean), where=Z_count > 0)
                
                # Calculate standard deviation
                for dist, snr, prob in zip(distances, snrs, probs):
                    dist_idx = np.searchsorted(distance_bins, dist) - 1
                    snr_idx = np.searchsorted(snr_bins, snr) - 1
                    
                    if 0 <= dist_idx < len(distance_bins)-1 and 0 <= snr_idx < len(snr_bins)-1:
                        if Z_count[snr_idx, dist_idx] > 0:
                            Z_std[snr_idx, dist_idx] += (prob - Z_mean[snr_idx, dist_idx]) ** 2
                
                # Finalize standard deviation calculation
                with np.errstate(divide='ignore', invalid='ignore'):
                    Z_std = np.sqrt(np.divide(Z_std, Z_count, out=np.zeros_like(Z_std), where=Z_count > 0))
                
                # Plot mean probability
                im = ax.pcolormesh(X[:-1, :-1], Y[:-1, :-1], Z_mean, cmap='viridis', vmin=0, vmax=1)
                
                # Add count information
                for i in range(len(distance_bins)-1):
                    for j in range(len(snr_bins)-1):
                        if Z_count[j, i] > 0:
                            # Text for bin center
                            bin_center_x = (distance_bins[i] + distance_bins[i+1]) / 2
                            bin_center_y = (snr_bins[j] + snr_bins[j+1]) / 2
                            
                            # Format count and mean
                            count_text = f"n={int(Z_count[j, i])}"
                            mean_text = f"μ={Z_mean[j, i]:.2f}"
                            
                            # Plot text with white outline for visibility
                            text = ax.text(bin_center_x, bin_center_y, f"{count_text}\n{mean_text}", 
                                         ha='center', va='center', fontsize=8,
                                         color='white', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                
                ax.set_title(f"{event_type.upper()} (n={len(distances)})")
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{event_type.upper()} (n=0)")
        
        # Add common labels and styling
        for ax in axes:
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("SNR")
            ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Mean Probability")
        
        # Add title
        plt.suptitle(f"Binned Statistics: SNR vs Distance for {model}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, top=0.93)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"binned_stats_{model}.png")
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved binned statistics for {model} to {output_path}")

# -----------------------------------------------------------------------------
# SNR Histograms by Channel Type
# -----------------------------------------------------------------------------
def generate_snr_histograms_by_channel(raw_trace_data, raw_event_data):
    """
    Create SNR histograms by channel type, showing correct vs incorrect predictions.
    Rearranged to 4 rows (event types) x 6 columns (channel types) with lighter shades for incorrect.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nGenerating SNR histograms by channel type...")
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Define channel types
    channel_types = [
        'strong_motion',
        '4_channel',
        'short_period_3c',
        'short_period_vertical',
        'broadband',
        'all'
    ]
    
    # Collect SNR data by channel type, event type, and correctness
    snr_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    # model -> channel_type -> event_type -> "correct"/"incorrect" -> list of SNR values
    
    # Process trace data
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        analyst = evid_to_analyst[evid]
        
        for model, traces in model_data.items():
            # Process each unique station only once
            processed_stations = set()
            
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                # Get channel information
                channels = None
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                else:
                    continue  # Skip if no channel info
                
                # Extract network.station
                netsta = extract_netsta(channels)
                if not netsta:
                    continue  # Skip if couldn't extract NET.STA
                
                # Skip if already processed this station
                if (evid, netsta) in processed_stations:
                    continue
                
                processed_stations.add((evid, netsta))
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Determine channel type
                channel_type = classify_channel_type(channels)
                
                # Extract probabilities for event types
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Find predicted class
                class_probs = [eq_prob, ex_prob, su_prob]
                highest_prob = max(class_probs)
                highest_prob_index = class_probs.index(highest_prob)
                predicted_class = event_types[highest_prob_index]
                
                # Store data based on correctness
                if predicted_class == analyst:
                    snr_data[model][channel_type][analyst]["correct"].append(snr)
                    snr_data[model]["all"][analyst]["correct"].append(snr)
                else:
                    snr_data[model][channel_type][analyst]["incorrect"].append(snr)
                    snr_data[model]["all"][analyst]["incorrect"].append(snr)
                
                # Also store in "all" event types category
                if predicted_class == analyst:
                    snr_data[model][channel_type]["all"]["correct"].append(snr)
                    snr_data[model]["all"]["all"]["correct"].append(snr)
                else:
                    snr_data[model][channel_type]["all"]["incorrect"].append(snr)
                    snr_data[model]["all"]["all"]["incorrect"].append(snr)
    
    # Get list of models for visualization
    models = sorted(snr_data.keys())
    
    if not models:
        print("No SNR data available. Skipping SNR histograms by channel type.")
        return
    
    # Event categories for rows
    event_categories = list(event_types) + ["all"]
    
    # Create figure for each model with new layout: 4 rows (event types) x 6 columns (channel types)
    for model in models:
        # Create figure
        fig, axes = plt.subplots(4, 6, figsize=(20, 12), sharex=True)
        
        # Define bins for histogram
        bins = np.arange(0, SNR_SATURATE + 1, 1)
        # chatgpt
        ## Predefined color pairs for each event type
        #color_pairs = {
        #    'eq': ('blue', 'lightblue'),
        #    'ex': ('red', 'lightcoral'),
        #    'su': ('green', 'lightgreen'),
        #    'all': ('purple', 'plum')
        #}

        # replace your literal lightblue/lightcoral/etc. with a call to adjust_color_lightness
        base_colors = {
            'eq': 'blue',
            'ex': 'red',
            'su': 'green',
            'all': 'purple'
        }

        color_pairs = {
            et: (base, adjust_color_lightness(base, 1.4))
            for et, base in base_colors.items()
        }
        
        # Plot histograms for each event type (rows) and channel type (columns)
        for i, event_type in enumerate(event_categories):
            for j, channel_type in enumerate(channel_types):
                ax = axes[i, j]
                
                correct_snrs = snr_data[model][channel_type][event_type]["correct"]
                incorrect_snrs = snr_data[model][channel_type][event_type]["incorrect"]
                
                total_snrs = correct_snrs + incorrect_snrs
                total_count = len(total_snrs)
                
                if total_count > 0:
                    # Get predefined colors for this event type
                    correct_color, incorrect_color = color_pairs.get(event_type, ('blue', 'lightblue'))
                    
                    # Plot histograms (stacked)
                    # First plot incorrect with lighter color
                    if incorrect_snrs:
                        ax.hist(incorrect_snrs, bins=bins, alpha=0.7, color=incorrect_color, label="Incorrect")
                    
                    # Then plot correct with darker color
                    if correct_snrs:
                        ax.hist(correct_snrs, bins=bins, alpha=0.7, color=correct_color, label="Correct")
                    
                    # Calculate accuracy
                    accuracy = len(correct_snrs) / total_count if total_count > 0 else 0
                    
                    ax.set_title(f"{event_type.upper()} (Acc={accuracy:.2f}, n={total_count})")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{event_type.upper()}")
                
                # Add column headers (channel type) only to top row
                if i == 0:
                    title = channel_type.replace('_', ' ').title()
                    ax.text(0.5, 1.15, title, ha='center', va='center', transform=ax.transAxes,
                           fontweight='bold')
                
                # Add x and y labels only to bottom row and leftmost column
                if i == len(event_categories) - 1:
                    ax.set_xlabel("SNR")
                
                if j == 0:
                    ax.set_ylabel("Count")
                
                # Format the highest SNR tick
                if ax.get_xticks()[-1] == SNR_SATURATE:
                    xticks = ax.get_xticks()
                    xticklabels = [f"{x:.0f}" for x in xticks[:-1]] + [f">{xticks[-1]:.0f}"]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                
                # Add grid
                ax.grid(alpha=0.3)
                
                # Add legend only to the top-left plot
                if i == 0 and j == 0:
                    ax.legend(loc='upper right')
        
        # Add overall title
        plt.suptitle(f"SNR Histograms by Channel Type for {model}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"snr_histograms_channel_{model}.png")
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved SNR histograms by channel type for {model} to {output_path}")

# Helper function to adjust color lightness
def adjust_color_lightness(color, amount=1.5):
    """
    Adjust the lightness of a color, ensuring proper RGB format.
    
    Args:
        color: Color to adjust (name, hex, or RGB tuple)
        amount: Amount to lighten (>1) or darken (<1)
    
    Returns:
        Adjusted color as RGB tuple
    """
    import matplotlib.colors as mc
    import colorsys
    # chatgpt:
    """
    Lighten (>1) or darken (<1) a Matplotlib color.
    Returns an RGB tuple.
    """
    try:
        # Convert name/hex/tuple → RGB
        rgb = mcolors.to_rgb(color)
        # RGB → HLS
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # scale lightness & clamp to [0,1]
        l = max(0, min(1, l * amount))
        # HLS → RGB
        return colorsys.hls_to_rgb(h, l, s)
    except Exception:
        # fallback: return the original color as an RGB tuple
        return mcolors.to_rgb(color)

    """  # Claude:
    try:
        # Handle named colors
        if isinstance(color, str):
            if color in mc.cnames:
                c = mc.cnames[color]
            else:
                c = color
        else:
            c = color
            
        # Convert to RGB
        rgb = mc.to_rgb(c)
        
        # Convert to HLS and adjust lightness
        h, l, s = colorsys.rgb_to_hls(*rgb)
        lightness = min(1, max(0, amount * l))
        rgb = colorsys.hls_to_rgb(h, lightness, s)
        
        # Return as valid RGB tuple
        return (rgb[0], rgb[1], rgb[2])
    except:
        # If any error occurs, return the original color
        return color
    """

# -----------------------------------------------------------------------------
# Classification Report Heatmap
# -----------------------------------------------------------------------------
def generate_classification_report(best_overall_params, metrics):
    """
    Create classification report heatmaps showing precision, recall, and F1 scores.
    Use same colorbar as confusion matrix plots.
    
    Args:
        best_overall_params: Dictionary of best parameters by model
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating classification report heatmaps...")
    
    # Create a figure with subplots for each model
    models = sorted(metrics.keys())
    n_models = len(models)
    
    # Calculate number of rows and columns
    n_rows = (n_models + 1) // 2
    n_cols = min(n_models, 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows), squeeze=False)
    
    # Find global min value for consistent colorbar
    global_min = 1.0
    for model in models:
        for metric_name in ["precision", "recall", "f1"]:
            for event_type in event_types:
                if event_type in metrics[model][metric_name]:
                    global_min = min(global_min, metrics[model][metric_name][event_type])
    
    # Add a small buffer to the global min
    global_min = max(0, global_min - 0.05)
    
    # Plot each model's metrics
    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Create data matrix for heatmap
        data = np.zeros((3, 3))  # 3 metrics x 3 event types
        
        # Fill the data matrix with corrected metrics
        for i, metric_name in enumerate(["precision", "recall", "f1"]):
            for j, event_type in enumerate(event_types):
                if event_type in metrics[model][metric_name]:
                    data[i, j] = metrics[model][metric_name][event_type]
        
        # Create the heatmap using Blues colormap (same as confusion matrix)
        im = ax.imshow(data, cmap='Blues', vmin=global_min, vmax=1)
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f"{data[i, j]:.2f}", 
                             ha="center", va="center", 
                             color="white" if data[i, j] < 0.5 else "black",
                             fontweight='bold')
        
        # Add labels
        ax.set_yticks(np.arange(3))
        ax.set_xticks(np.arange(3))
        ax.set_yticklabels(["Precision", "Recall", "F1"])
        ax.set_xticklabels([et.upper() for et in event_types])
        
        # Add title
        ax.set_title(f"{model} (params: {metrics[model]['params']})")
        
        # Add grid
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    
    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Score Value")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "classification_report.png")
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Saved classification report to {output_path}")

# -----------------------------------------------------------------------------
# Updated Probability vs SNR Plots
# -----------------------------------------------------------------------------
def generate_probability_vs_snr_scatter_fixed(raw_trace_data, raw_event_data):
    """
    Generate scatter plots of probability vs. SNR for each model.
    Fixed to only show highest probability class for each trace.
    
    Args:
        raw_trace_data: Dictionary of raw trace data
        raw_event_data: Dictionary of raw event data
    """
    print("\nGenerating updated probability vs. SNR scatter plots...")
    
    # Create a data structure to hold probabilities and SNR values
    prob_snr_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # model -> event_type -> "prob"/"snr" -> list of values
    
    # Create mapping from evid to analyst label
    evid_to_analyst = {}
    for evid, event_entries in raw_event_data.items():
        if event_entries:
            evid_to_analyst[evid] = event_entries[0]["analyst"]
    
    # Process trace data and count unique stations
    for evid, model_data in raw_trace_data.items():
        if evid not in evid_to_analyst:
            continue
        
        for model, traces in model_data.items():
            # Count unique stations in traces
            unique_station_count = count_unique_stations(traces)
            
            # Process each trace
            for trace_data in traces:
                if len(trace_data) < 3:
                    continue
                
                trace_index = trace_data[0]
                probs = trace_data[1]
                
                # Get SNR (saturate if necessary)
                snr = None
                if len(trace_data) > 3 and isinstance(trace_data[3], (int, float)):
                    snr = min(float(trace_data[3]), SNR_SATURATE)
                else:
                    continue  # Skip if no SNR
                
                # Extract probabilities for each class
                eq_prob = probs[0]
                ex_prob = probs[1]
                su_prob = probs[3]  # Skipping noise
                
                # Find class with highest probability
                class_probs = [eq_prob, ex_prob, su_prob]
                highest_prob = max(class_probs)
                highest_prob_index = class_probs.index(highest_prob)
                highest_prob_class = event_types[highest_prob_index]
                
                # Get channel information
                if isinstance(trace_data[2], list):
                    channels = trace_data[2]
                    # Extract NET.STA
                    netsta = extract_netsta(channels)
                    if not netsta:
                        continue  # Skip if couldn't extract NET.STA
                else:
                    continue  # Skip if no channel info
                
                # Store data only for highest probability class
                prob_snr_data[model]["highest"]["prob"].append(highest_prob)
                prob_snr_data[model]["highest"]["snr"].append(snr)
                prob_snr_data[model]["highest"]["class"].append(highest_prob_class)
                
                # Store data for the highest probability class subplot
                prob_snr_data[model][highest_prob_class]["prob"].append(highest_prob)
                prob_snr_data[model][highest_prob_class]["snr"].append(snr)
    
    # Generate scatter plots for each model
    for model in sorted(prob_snr_data.keys()):
        # Skip if no highest probability data
        if "highest" not in prob_snr_data[model] or not prob_snr_data[model]["highest"]["prob"]:
            print(f"Skipping probability vs. SNR scatter plot for {model} - no data available")
            continue
        
        # Count total traces
        total_trace_count = len(prob_snr_data[model]["highest"]["prob"])
        
        # Create figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        plt.suptitle(f'Probability vs. SNR Scatter Plots for {model} (n={total_trace_count})')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot highest probability vs SNR (first subplot)
        ax = axes[0]
        for event_type in event_types:
            # Extract data points where this event type had the highest probability
            indices = [i for i, cls in enumerate(prob_snr_data[model]["highest"]["class"]) if cls == event_type]
            snr_values = [prob_snr_data[model]["highest"]["snr"][i] for i in indices]
            prob_values = [prob_snr_data[model]["highest"]["prob"][i] for i in indices]
            
            ax.scatter(snr_values, prob_values, alpha=0.6, color=COLORS[event_type], 
                     label=f"{event_type.upper()} (n={len(indices)})")
        
        ax.set_title(f'Highest Probability vs. SNR (n={total_trace_count})')
        ax.set_xlabel('SNR')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot individual event type probabilities vs SNR (subplots 2-4)
        for i, event_type in enumerate(event_types):
            ax = axes[i+1]
            
            if event_type in prob_snr_data[model] and prob_snr_data[model][event_type]["snr"]:
                snr_values = prob_snr_data[model][event_type]["snr"]
                prob_values = prob_snr_data[model][event_type]["prob"]
                
                ax.scatter(snr_values, prob_values, alpha=0.6, color=COLORS[event_type], 
                         label=f"{event_type.upper()} (n={len(snr_values)})")
                
                ax.set_title(f'{event_type.upper()} Probability vs. SNR (n={len(snr_values)})')
            else:
                ax.text(0.5, 0.5, f"No data for {event_type.upper()}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{event_type.upper()} Probability vs. SNR (n=0)')
            
            ax.set_xlabel('SNR')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            if len(snr_values) > 0:
                ax.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for title
        
        output_path = os.path.join(OUTPUT_DIR, f'probability_vs_snr_scatter_fixed_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved updated probability vs. SNR scatter plot for {model} to {output_path}")

def generate_channel_performance_plots2():
    """
    Generate performance metric plots broken down by model, with bars for each channel type.
    
    Creates a grid of plots showing performance metrics for each model.
    Each plot contains 6 bars representing channel types for each of 4 metrics.
    Uses the updated data structure to calculate precision correctly.
    """
    print("\nGenerating model performance plots...")
    
    # Channel types and their display names (keeping original order)
    channel_types = [
        ('strong_motion', 'Strong Motion'),
        ('4_channel', '4 Channel'),
        ('short_period_vertical', 'Short Period Vertical'),
        ('short_period_3c', 'Short Period 3C'),
        ('broadband', 'Broadband'),
        ('all', 'All Channels')
    ]
    
    # Consistent color palette for channel types
    channel_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b'   # Brown
    ]
    channel_color_map = dict(zip([ct[0] for ct in channel_types], channel_colors))
    
    # Calculate metrics for each channel type and model
    channel_metrics = {}
    
    # Iterate through channel types
    for channel_type, _ in channel_types:
        channel_metrics[channel_type] = {}
        
        # Get all models that have data for this channel type
        models = []
        for model in sorted(trace_channel_results[channel_type].keys()):
            # Check if there's any data for this model
            if (any(trace_channel_results[channel_type][model]["correct"].values()) or 
                any(trace_channel_results[channel_type][model]["incorrect"].values())):
                models.append(model)
        
        # Calculate metrics for each model
        for model in models:
            # Initialize metrics dictionary for this model
            channel_metrics[channel_type][model] = {
                'precision': {},
                'recall': {},
                'f1': {},
                'accuracy': 0,
                'macro_precision': 0,
                'macro_recall': 0,
                'macro_f1': 0
            }
            
            # Calculate per-class metrics
            for event_type in event_types:
                # Get counts for precision calculation (true positives and false positives)
                true_positives = len(trace_predictions[channel_type][model]["true_positive"].get(event_type, []))
                false_positives = len(trace_predictions[channel_type][model]["false_positive"].get(event_type, []))
                
                # Get counts for recall calculation (true positives and false negatives)
                false_negatives = len(trace_predictions[channel_type][model]["false_negative"].get(event_type, []))
                
                # Calculate total predictions of this class
                total_actual = true_positives + false_negatives
                
                # Calculate precision
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate recall
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                # Calculate F1 score - directly from precision and recall
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store metrics
                channel_metrics[channel_type][model]['precision'][event_type] = precision
                channel_metrics[channel_type][model]['recall'][event_type] = recall
                channel_metrics[channel_type][model]['f1'][event_type] = f1
            
            # Calculate overall accuracy
            total_correct = sum(len(data) for data in trace_channel_results[channel_type][model]["correct"].values())
            total_incorrect = sum(len(data) for data in trace_channel_results[channel_type][model]["incorrect"].values())
            total = total_correct + total_incorrect
            
            accuracy = total_correct / total if total > 0 else 0
            channel_metrics[channel_type][model]['accuracy'] = accuracy
            
            # Calculate macro-averaged metrics
            channel_metrics[channel_type][model]['macro_precision'] = (
                sum(channel_metrics[channel_type][model]['precision'].values()) / 
                len(channel_metrics[channel_type][model]['precision']) 
                if channel_metrics[channel_type][model]['precision'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_recall'] = (
                sum(channel_metrics[channel_type][model]['recall'].values()) / 
                len(channel_metrics[channel_type][model]['recall']) 
                if channel_metrics[channel_type][model]['recall'] else 0
            )
            
            channel_metrics[channel_type][model]['macro_f1'] = (
                sum(channel_metrics[channel_type][model]['f1'].values()) / 
                len(channel_metrics[channel_type][model]['f1']) 
                if channel_metrics[channel_type][model]['f1'] else 0
            )
    
    # Get list of models (sorted to ensure consistent order)
    models = sorted(set(model for channel_type in channel_metrics.values() 
                        for model in channel_type.keys()))
    
    # Setup figure with (models)x1 grid
    fig, axes = plt.subplots(len(models), 1, figsize=(20, 5*len(models)), squeeze=False)
    
    # Metrics to plot
    metric_types = [
        ('macro_precision', 'Precision'),
        ('macro_recall', 'Recall'),
        ('macro_f1', 'F1'),
        ('accuracy', 'Accuracy')
    ]
    
    # Prepare metrics for plotting
    for model_idx, model in enumerate(models):
        ax = axes[model_idx, 0]
        
        # Bar positioning
        n_channel_types = len(channel_types)
        bar_width = 0.15
        
        # Set up a single combined legend
        legend_handles = []
        legend_labels = []
                
        # Plot each channel type across metrics
        for channel_idx, (channel_type, channel_display) in enumerate(channel_types):
            # Calculate trace count for this channel type and model
            trace_count = calculate_channel_trace_count(trace_channel_results, channel_type, model)
            
            values = []
            
            # Collect metric values for this channel type and model
            for metric_key, metric_label in metric_types:
                try:
                    value = channel_metrics[channel_type][model][metric_key]
                    values.append(value)
                except KeyError:
                    values.append(0)
            
            # Calculate base x position
            base_x = np.arange(len(metric_types))
            x = base_x + channel_idx * (bar_width * 1.0)
            
            # Plot bars for this channel type
            bars = ax.bar(x, values, bar_width, 
                          color=channel_color_map[channel_type], 
                          edgecolor='black', 
                          linewidth=1,
                          label=f"{channel_display} (n={trace_count})")
            
            # Add to legend only if this is the first metric (to avoid duplicates)
            if channel_idx == 0:
                # Create a single handle for the legend
                legend_patch = plt.Rectangle((0,0),1,1, 
                                          color=channel_color_map[channel_type],
                                          edgecolor='black',
                                          linewidth=1)
                legend_handles.append(legend_patch)
                legend_labels.append(f"{channel_display} (n={trace_count})")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only label bars above 5%
                    ax.text(bar.get_x() + bar.get_width()/2., height+0.01,
                            f'{height:.2f}', 
                            ha='center', va='bottom', 
                            fontsize=10)
        
        # Customize the plot
        ax.set_title(f"Model: {model}")
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Set x-ticks to metric names
        x_tick_positions = np.arange(len(metric_types)) + (len(channel_types) * bar_width * 0.6) / 2
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([metric[1] for metric in metric_types], rotation=0)
        
        # Add single legend with combined handles
        ax.legend(legend_handles, legend_labels, 
                 loc='upper right', bbox_to_anchor=(1.15, 1),
                 title='Channel Types')

    # Adjust layout
    plt.suptitle('Performance Metrics by Model and Channel Type', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'channel_performance2_fixed.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved fixed model performance plot to {output_path}")

def generate_channel_performance_plots2_weighted():
    """
    Generate performance metric plots broken down by model, with bars for each channel type.
    
    Creates a grid of plots showing performance metrics for each model.
    Each plot contains 6 bars representing channel types for each of 4 metrics.
    Uses weighted averages instead of macro-averages for precision, recall, and F1.
    """
    print("\nGenerating model performance plots with weighted metrics...")
    
    # Channel types and their display names (keeping original order)
    channel_types = [
        ('strong_motion', 'Strong Motion'),
        ('4_channel', '4 Channel'),
        ('short_period_vertical', 'Short Period Vertical'),
        ('short_period_3c', 'Short Period 3C'),
        ('broadband', 'Broadband'),
        ('all', 'All Channels')
    ]
    
    # Consistent color palette for channel types
    channel_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b'   # Brown
    ]
    channel_color_map = dict(zip([ct[0] for ct in channel_types], channel_colors))
    
    # Calculate metrics for each channel type and model
    channel_metrics = {}
    
    # Iterate through channel types
    for channel_type, _ in channel_types:
        channel_metrics[channel_type] = {}
        
        # Get all models that have data for this channel type
        models = []
        for model in sorted(trace_channel_results[channel_type].keys()):
            # Check if there's any data for this model
            if (any(trace_channel_results[channel_type][model]["correct"].values()) or 
                any(trace_channel_results[channel_type][model]["incorrect"].values())):
                models.append(model)
        
        # Calculate metrics for each model
        for model in models:
            # Initialize metrics dictionary for this model
            channel_metrics[channel_type][model] = {
                'precision': {},
                'recall': {},
                'f1': {},
                'support': {},
                'accuracy': 0,
                'weighted_precision': 0,
                'weighted_recall': 0,
                'weighted_f1': 0
            }
            
            # Calculate per-class metrics
            for event_type in event_types:
                # Get counts for precision calculation (true positives and false positives)
                true_positives = len(trace_predictions[channel_type][model]["true_positive"].get(event_type, []))
                false_positives = len(trace_predictions[channel_type][model]["false_positive"].get(event_type, []))
                
                # Get counts for recall calculation (true positives and false negatives)
                false_negatives = len(trace_predictions[channel_type][model]["false_negative"].get(event_type, []))
                
                # Calculate total predictions of this class (support)
                total_actual = true_positives + false_negatives
                channel_metrics[channel_type][model]['support'][event_type] = total_actual
                
                # Calculate precision
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate recall
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                # Calculate F1 score - directly from precision and recall
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store metrics
                channel_metrics[channel_type][model]['precision'][event_type] = precision
                channel_metrics[channel_type][model]['recall'][event_type] = recall
                channel_metrics[channel_type][model]['f1'][event_type] = f1
            
            # Calculate overall accuracy
            total_correct = sum(len(data) for data in trace_channel_results[channel_type][model]["correct"].values())
            total_incorrect = sum(len(data) for data in trace_channel_results[channel_type][model]["incorrect"].values())
            total = total_correct + total_incorrect
            
            accuracy = total_correct / total if total > 0 else 0
            channel_metrics[channel_type][model]['accuracy'] = accuracy
            
            # Calculate weighted averages (weighted by support)
            total_support = sum(channel_metrics[channel_type][model]['support'].values())
            
            if total_support > 0:
                # Weighted precision
                weighted_precision = sum(
                    channel_metrics[channel_type][model]['precision'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['precision']
                )
                
                # Weighted recall
                weighted_recall = sum(
                    channel_metrics[channel_type][model]['recall'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['recall']
                )
                
                # Weighted F1
                weighted_f1 = sum(
                    channel_metrics[channel_type][model]['f1'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['f1']
                )
            else:
                weighted_precision = 0
                weighted_recall = 0
                weighted_f1 = 0
            
            # Store weighted metrics
            channel_metrics[channel_type][model]['weighted_precision'] = weighted_precision
            channel_metrics[channel_type][model]['weighted_recall'] = weighted_recall
            channel_metrics[channel_type][model]['weighted_f1'] = weighted_f1
    
    # Get list of models (sorted to ensure consistent order)
    models = sorted(set(model for channel_type in channel_metrics.values() 
                         for model in channel_type.keys()))
    
    # Setup figure with (models)x1 grid
    fig, axes = plt.subplots(len(models), 1, figsize=(20, 5*len(models)), squeeze=False)
    
    # Metrics to plot
    metric_types = [
        ('weighted_precision', 'Weighted Precision'),
        ('weighted_recall', 'Weighted Recall'),
        ('weighted_f1', 'Weighted F1'),
        ('accuracy', 'Accuracy')
    ]
    
    # Prepare metrics for plotting
    for model_idx, model in enumerate(models):
        ax = axes[model_idx, 0]
        
        # Bar positioning
        n_channel_types = len(channel_types)
        bar_width = 0.15
        
        # Set up a single combined legend
        legend_handles = []
        legend_labels = []
                
        # Plot each channel type across metrics
        for channel_idx, (channel_type, channel_display) in enumerate(channel_types):
            # Calculate trace count for this channel type and model
            trace_count = calculate_channel_trace_count(trace_channel_results, channel_type, model)
            
            values = []
            
            # Collect metric values for this channel type and model
            for metric_key, metric_label in metric_types:
                try:
                    value = channel_metrics[channel_type][model][metric_key]
                    values.append(value)
                except KeyError:
                    values.append(0)
            
            # Calculate base x position
            base_x = np.arange(len(metric_types))
            x = base_x + channel_idx * (bar_width * 1.0)
            
            # Plot bars for this channel type
            bars = ax.bar(x, values, bar_width, 
                          color=channel_color_map[channel_type], 
                          edgecolor='black', 
                          linewidth=1,
                          label=f"{channel_display} (n={trace_count})")
            
            # Add to legend only if this is the first metric (to avoid duplicates)
            if channel_idx == 0:
                # Create a single handle for the legend
                legend_patch = plt.Rectangle((0,0),1,1, 
                                          color=channel_color_map[channel_type],
                                          edgecolor='black',
                                          linewidth=1)
                legend_handles.append(legend_patch)
                legend_labels.append(f"{channel_display} (n={trace_count})")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only label bars above 5%
                    ax.text(bar.get_x() + bar.get_width()/2., height+0.01,
                            f'{height:.2f}', 
                            ha='center', va='bottom', 
                            fontsize=10)
        
        # Customize the plot
        ax.set_title(f"Model: {model}")
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Set x-ticks to metric names
        x_tick_positions = np.arange(len(metric_types)) + (len(channel_types) * bar_width * 0.6) / 2
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([metric[1] for metric in metric_types], rotation=0)
        
        # Add single legend with combined handles
        ax.legend(legend_handles, legend_labels, 
                 loc='upper right', bbox_to_anchor=(1.15, 1),
                 title='Channel Types')

    # Adjust layout
    plt.suptitle('Performance Metrics by Model and Channel Type (Weighted)', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'channel_performance2_weighted.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved weighted model performance plot to {output_path}")

def generate_channel_performance_plots_weighted():
    """
    Generate performance metric plots broken down by channel type.
    
    Creates a 2x3 grid of plots showing performance metrics for each channel type.
    Uses weighted averages instead of macro-averages for precision, recall, and F1.
    """
    print("\nGenerating channel performance plots with weighted metrics...")
    
    # Setup figure with 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey=True)
    
    # Channel types and their display names
    channel_types = [
        ('strong_motion', 'Strong Motion\n(HNE/HNN/HNZ or ENE/ENN/ENZ)'),
        ('4_channel', '4 Channel\n((ENE+ENN or HNE+HNN) + EHZ)'),
        ('short_period_3c', 'Short Period 3 Component\n(EHE/EHN/EHZ)'),
        ('short_period_vertical', 'Short Period Vertical\n(EHZ Only)'),
        ('broadband', 'Broadband\n(BHE/BHN/BHZ or HHE/HHN/HHZ)'),
        ('all', 'All Channels\nCombined')
    ]
    
    # Find global y-max for consistent scaling
    y_max = 0
    
    # Calculate metrics for each channel type and find max value
    channel_metrics = {}
    for channel_type, _ in channel_types:
        channel_metrics[channel_type] = {}
        
        # Get all models that have data for this channel type
        models = []
        for model in sorted(trace_channel_results[channel_type].keys()):
            # Check if there's any data for this model
            if (any(trace_channel_results[channel_type][model]["correct"].values()) or 
                any(trace_channel_results[channel_type][model]["incorrect"].values())):
                models.append(model)
        
        if not models:
            continue  # Skip if no models have data for this channel type
        
        # Calculate metrics for each model
        for model in models:
            # Initialize metrics dictionary for this model
            channel_metrics[channel_type][model] = {
                'precision': {},
                'recall': {},
                'f1': {},
                'support': {},
                'accuracy': 0,
                'weighted_precision': 0,
                'weighted_recall': 0,
                'weighted_f1': 0
            }
            
            # Calculate per-class metrics
            for event_type in event_types:
                # Get counts for precision calculation (true positives and false positives)
                true_positives = len(trace_predictions[channel_type][model]["true_positive"].get(event_type, []))
                false_positives = len(trace_predictions[channel_type][model]["false_positive"].get(event_type, []))
                
                # Get counts for recall calculation (true positives and false negatives)
                false_negatives = len(trace_predictions[channel_type][model]["false_negative"].get(event_type, []))
                
                # Calculate total predictions of this class (support)
                total_actual = true_positives + false_negatives
                channel_metrics[channel_type][model]['support'][event_type] = total_actual
                
                # Calculate precision
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate recall
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store metrics
                channel_metrics[channel_type][model]['precision'][event_type] = precision
                channel_metrics[channel_type][model]['recall'][event_type] = recall
                channel_metrics[channel_type][model]['f1'][event_type] = f1
            
            # Calculate overall accuracy
            total_correct = sum(len(data) for data in trace_channel_results[channel_type][model]["correct"].values())
            total_incorrect = sum(len(data) for data in trace_channel_results[channel_type][model]["incorrect"].values())
            total = total_correct + total_incorrect
            
            accuracy = total_correct / total if total > 0 else 0
            channel_metrics[channel_type][model]['accuracy'] = accuracy
            
            # Calculate weighted averages (weighted by support)
            total_support = sum(channel_metrics[channel_type][model]['support'].values())
            
            if total_support > 0:
                # Weighted precision
                weighted_precision = sum(
                    channel_metrics[channel_type][model]['precision'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['precision']
                )
                
                # Weighted recall
                weighted_recall = sum(
                    channel_metrics[channel_type][model]['recall'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['recall']
                )
                
                # Weighted F1
                weighted_f1 = sum(
                    channel_metrics[channel_type][model]['f1'][et] * 
                    channel_metrics[channel_type][model]['support'][et] / total_support
                    for et in event_types if et in channel_metrics[channel_type][model]['f1']
                )
            else:
                weighted_precision = 0
                weighted_recall = 0
                weighted_f1 = 0
            
            # Store weighted metrics
            channel_metrics[channel_type][model]['weighted_precision'] = weighted_precision
            channel_metrics[channel_type][model]['weighted_recall'] = weighted_recall
            channel_metrics[channel_type][model]['weighted_f1'] = weighted_f1
            
            # Update global y-max
            y_max = max(y_max, 
                        weighted_precision,
                        weighted_recall,
                        weighted_f1,
                        accuracy)
    
    # Add a small buffer to y_max
    y_max = min(1.0, y_max * 1.1)

    # Now plot each channel type in its grid position
    for i, (channel_type, display_name) in enumerate(channel_types):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        models = list(sorted(channel_metrics.get(channel_type, {}).keys()))
        
        if not models:
            ax.text(0.5, 0.5, f"No data for {display_name}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name)
            continue
        
        # Calculate total trace count for this channel type
        total_trace_count = 0
        for model in models:
            # Get count of unique stations
            model_trace_count = calculate_channel_trace_count(trace_channel_results, channel_type, model)
            total_trace_count += model_trace_count
        
        # Set up bar positions
        x = np.arange(len(models))
        bar_width = 0.2
        
        # Plot bars for each metric
        precision_bars = ax.bar(x - 1.5*bar_width, 
                                [channel_metrics[channel_type][model]['weighted_precision'] for model in models],
                                bar_width, label='Weighted Precision', color='skyblue')
        
        recall_bars = ax.bar(x - 0.5*bar_width, 
                            [channel_metrics[channel_type][model]['weighted_recall'] for model in models],
                            bar_width, label='Weighted Recall', color='lightgreen')
        
        f1_bars = ax.bar(x + 0.5*bar_width, 
                        [channel_metrics[channel_type][model]['weighted_f1'] for model in models],
                        bar_width, label='Weighted F1', color='salmon')
        
        accuracy_bars = ax.bar(x + 1.5*bar_width, 
                              [channel_metrics[channel_type][model]['accuracy'] for model in models],
                              bar_width, label='Accuracy', color='mediumpurple')
        
        # Add text labels on top of the bars
        bar_label_fontsize = 7
        
        # Function to add formatted labels to bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only add label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=bar_label_fontsize)
        
        # Add labels to all bar sets
        add_labels(precision_bars)
        add_labels(recall_bars)
        add_labels(f1_bars)
        add_labels(accuracy_bars)
        
        # Customize the plot
        ax.set_title(f"{display_name}\nn={total_trace_count} stations")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        if col == 0:
            ax.set_ylabel('Score')
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set main title and adjust layout
    plt.suptitle('Performance Metrics by Channel Type (Weighted)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'channel_performance_weighted.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved weighted channel performance plot to {output_path}")

def generate_macro_metrics_plot(metrics):
    """
    Generate macro metrics plot showing precision, recall, F1 and accuracy.
    Use correct trace-weighted calculations.
    
    Args:
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating macro metrics plot...")
    
    models = sorted(metrics.keys())
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Width and positions for grouped bars
    bar_width = 0.15
    x = np.arange(len(models))
    
    # Calculate trace-level macro metrics using trace_predictions
    trace_macro_metrics = {}
    for model in models:
        if model not in trace_predictions:
            continue
            
        # Calculate trace-weighted metrics
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0
        total_events = 0
        
        for event_type in event_types:
            # Get counts
            true_positives = len(trace_predictions[model]["true_positive"].get(event_type, []))
            false_positives = len(trace_predictions[model]["false_positive"].get(event_type, []))
            false_negatives = len(trace_predictions[model]["false_negative"].get(event_type, []))
            
            # Calculate event count for this class
            event_count = true_positives + false_negatives
            total_events += event_count
            
            # Skip if no events
            if event_count == 0:
                continue
            
            # Calculate precision
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            # Calculate recall
            recall = true_positives / event_count
            
            # Calculate F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Weight by event count
            weighted_precision += precision * event_count
            weighted_recall += recall * event_count
            weighted_f1 += f1 * event_count
        
        # Calculate weighted averages
        if total_events > 0:
            weighted_precision /= total_events
            weighted_recall /= total_events
            weighted_f1 /= total_events
        
        # Store
        trace_macro_metrics[model] = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    
    # Plot event-wise macro metrics
    event_precision_bars = plt.bar(x - 2.5*bar_width, [metrics[model]['macro_precision'] for model in models], 
            bar_width, label='Precision (Event)', color='skyblue')
    event_recall_bars = plt.bar(x - 1.5*bar_width, [metrics[model]['macro_recall'] for model in models], 
            bar_width, label='Recall (Event)', color='lightgreen')
    event_f1_bars = plt.bar(x - 0.5*bar_width, [metrics[model]['macro_f1'] for model in models], 
            bar_width, label='F1 (Event)', color='salmon')
    
    # Plot trace-wise macro metrics with hatching
    trace_precision_bars = plt.bar(x + 0.5*bar_width, 
                               [trace_macro_metrics[model]['precision'] if model in trace_macro_metrics else 0 for model in models], 
                               bar_width, label='Precision (Trace)', color='skyblue', hatch='////')
    trace_recall_bars = plt.bar(x + 1.5*bar_width, 
                            [trace_macro_metrics[model]['recall'] if model in trace_macro_metrics else 0 for model in models], 
                            bar_width, label='Recall (Trace)', color='lightgreen', hatch='////')
    trace_f1_bars = plt.bar(x + 2.5*bar_width, 
                         [trace_macro_metrics[model]['f1'] if model in trace_macro_metrics else 0 for model in models], 
                         bar_width, label='F1 (Trace)', color='salmon', hatch='////')
    
    # Add formatted labels to the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add labels to all bar sets
    add_labels(event_precision_bars)
    add_labels(event_recall_bars)
    add_labels(event_f1_bars)
    add_labels(trace_precision_bars)
    add_labels(trace_recall_bars)
    add_labels(trace_f1_bars)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Macro-averaged Performance Metrics by Model')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend(ncol=2)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    # Remove vertical grid lines at x-tick positions
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, grid_alpha=0)
    
    # Add extra bottom margin to make room for labels
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.2)  # Additional adjustment for x-tick labels
    
    output_path = os.path.join(OUTPUT_DIR, 'macro_metrics.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved macro metrics plot to {output_path}")

# Update the main function to include this new visualization
def main():
    """Main function to orchestrate the analysis process."""
    # Ensure output directory exists
    ensure_output_dir()
    
    # Load distance data
    distances = load_distances()
    
    # Process model outputs and get raw event data
    raw_event_data, raw_trace_data = process_model_outputs()
    
    # Find best parameters
    best_params_by_event, best_overall_params = find_best_parameters()
    
    # Build histograms using only the best parameter set for each model
    build_event_histograms(raw_event_data, best_overall_params)
    build_trace_histograms(raw_trace_data, raw_event_data)
    
    # Calculate metrics
    metrics = calculate_metrics(best_overall_params)
    
    # Output results

    print("chatgpt ")
    print_and_write_results_chatgpt(best_params_by_event, metrics, raw_event_data)
    #print("claude ")
    #print_and_write_results_claude(best_params_by_event, metrics, raw_event_data)
    #print("old")
    #print_and_write_results(best_params_by_event, metrics, raw_event_data)

    # Generate performance visualizations
    generate_confusion_matrices(metrics)
    generate_performance_plots(metrics)
    
    # Generate original visualizations
    generate_magnitude_histograms_v1(best_overall_params)
    generate_magnitude_histograms_v2(best_overall_params)
    generate_probdist_histograms_event(best_overall_params)
    generate_probdist_histograms_trace(best_overall_params)
    generate_max_probability_histograms(best_overall_params)
    # Removed: generate_scatter_plots_event()
    # Removed: generate_scatter_plots_trace()
    generate_heatmaps_event()
    generate_heatmaps_trace()
    
    # Generate fixed performance plots (original macro-averaging)
    generate_channel_performance_plots()
    generate_channel_performance_plots2()
    
    # Generate performance plots with weighted metrics (better for imbalanced classes)
    generate_channel_performance_plots_weighted()
    generate_channel_performance_plots2_weighted()
    
    # Generate classification report heatmap
    generate_classification_report(best_overall_params, metrics)
    
    # Generate updated probability vs. SNR visualizations using actual SNR values
    # Removed: generate_probability_vs_snr_scatter_fixed(raw_trace_data, raw_event_data)
    generate_probability_vs_snr_heatmap_fixed(raw_trace_data, raw_event_data)
    
    # Generate distance vs probability visualizations
    generate_distance_vs_probability_plots(raw_trace_data, raw_event_data, distances)
    
    # Generate new dataset histograms
    generate_dataset_histograms(raw_trace_data, raw_event_data, distances)
    
    # Generate combined visualizations
    # Removed the original implementation because it's being replaced:
    # generate_small_multiples_grid(raw_trace_data, raw_event_data, distances)
    
    # Generate new small multiples visualization as heatmap
    generate_small_multiples_heatmap(raw_trace_data, raw_event_data, distances)
    
    generate_heatmap_with_bubbles(raw_trace_data, raw_event_data, distances)
    generate_contour_overlay(raw_trace_data, raw_event_data, distances)
    generate_binned_statistics(raw_trace_data, raw_event_data, distances)

    # Generate SNR histograms by channel type
    generate_snr_histograms_by_channel(raw_trace_data, raw_event_data)


    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()


