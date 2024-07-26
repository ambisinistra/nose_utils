import os
import warnings

import numpy as np
import pandas as pd

from datetime import timedelta

D_columns = ["D" + str(i) for i in range(1, 65)]

def process_file(file_path, is_train=True):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract patient information
    patient_id = lines[0].strip().split(':')[1].strip()
    if is_train:
        result = 1 if lines[1].strip().split(':')[1].strip() == 'POSITIVE' else 0

    # Extract sensor data
    if is_train:
        header = lines[3].strip().split('\t')
        data_lines = [line.strip().split('\t') for line in lines[4:]]
    else:
        header = lines[2].strip().split('\t')
        data_lines = [line.strip().split('\t') for line in lines[3:]]

    # Create DataFrame for sensor data
    df = pd.DataFrame(data_lines, columns=header)

    # Add patient information to the DataFrame
    df['Patient_Id'] = patient_id
    if is_train:
        df['Result'] = result

    # Rename columns for consistency
    df.rename(columns={'Min:Sec': 'Timestamp'}, inplace=True)

    # Reorder columns
    if is_train:
        columns = ['Patient_Id', 'Result', 'Timestamp'] + header[1:]
    else:
        columns = ['Patient_Id', 'Timestamp'] + header[1:]
    df = df[columns]

    return df

def process_directory(dir_name, is_train=True):
    all_data = list()

    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_name, filename)
            df = process_file(file_path)
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df.rename(inplace=True, columns=dict(zip([" D" + str(i) for i in range(1, 65)], D_columns)))

    for col in D_columns:
        combined_df[col] = combined_df[col].astype(float)

    return combined_df

def convert_to_timedelta(time_str):
    minute, second = time_str.split(':')
    second, millisecond = second.split('.')
    return timedelta(hours=0, minutes=int(minute), seconds=int(second), milliseconds=int(millisecond))

def subtract_first_timestamp(group):
    if group.index[0] == group.Timestamp.idxmin():
        return group.Timestamp - group.Timestamp.min()
    else:
        zero_index = group.Timestamp.idxmin()
        group.loc[group.index >= zero_index, "Timestamp"] += pd.Timedelta(hours=1)
        return group.Timestamp - group.Timestamp.min()

def time_processing(df, time_column="Timestamp",
                    pat_column="Patient_Id", debug=False):

    if debug:
        df["time_str"] = df["Timestamp"]

    df[time_column] = df[time_column].apply(convert_to_timedelta)
    df[time_column] = df.groupby(pat_column).apply(subtract_first_timestamp).reset_index(level=0, drop=True)
    df[time_column] = pd.to_timedelta(df[time_column])

    return df

def resample_sensors(df, sample_time="1s", last_idx=840, return_np=False,
                     pat_col="Patient_Id", time_col="Timestamp", warn=True):
    
    coef = int(sample_time[:-1])
    last_idx = int(last_idx / coef)
    if warn:
        warnings.warn("right indexing for non seconds not implemented")
        warnings.warn("Warning, this function drop nonsensor columns")
    resampled_df = list()
    patient_ids = df[pat_col].unique().tolist()
    for pat_id in patient_ids:
        cur_patient = df[df[pat_col] == pat_id]
        cur_patient = cur_patient.set_index(time_col)[D_columns].resample(sample_time).mean().ffill().iloc[:last_idx]
        resampled_df.append(cur_patient)
    if return_np:
        return np.stack(resampled_df)
    else:
        return pd.concat(resampled_df, keys=patient_ids).reset_index(level=0).rename(columns={'level_0': 'Patient_Id'})

def normalize_patient(df):
    deviation = df.std()
    deviation[deviation.isnull()] = 1
    return (df - df.mean()) / deviation

def normalize_sensors(df, pat_col="Patient_Id", time_col="Timestamp"):
    return df.groupby(pat_col).apply(normalize_patient).reset_index().set_index(time_col) 

def unpack_dict(nested_dict):
    flat_dict = {}
    for outer_key, outer_value in nested_dict.items():
        for inner_key, inner_dict in outer_value.items():
            for stat_key, stat_value in inner_dict.items():
                new_key = f"{inner_key} {outer_key} {stat_key}"
                flat_dict[new_key] = stat_value
    return flat_dict

def extract_feature_timeline(timeline, features=None):
    output = dict()
    for feature in features:
        if feature == "min":
            output["min"] = timeline.min()
        if feature == "max":
            output["max"] = timeline.max()
        if feature == "mean":
            output["mean"] = timeline.mean()
    return output

def extract_feature_windows(df, windows=None):
    output = dict()
    features = ["min", "max", "mean"]
    for key in windows.keys():
        window_name = str(key) + " " + str(windows[key])
        time_window = (df.index >= timedelta(seconds=key)) & (df.index < timedelta(seconds=windows[key]))
        output[window_name] = df[time_window].apply(extract_feature_timeline, features=features)
    return unpack_dict(output)

def extract_feature_sensors(df, pat_col="Patient_Id", sensors=D_columns, windows=None):
    output = df.groupby(pat_col)[sensors].apply(extract_feature_windows, windows=windows)
    return output.apply(pd.Series)