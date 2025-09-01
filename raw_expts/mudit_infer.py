import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, Reshape, Conv2D, BatchNormalization, Conv1D, 
                                     LSTM, Dense, Activation, concatenate, UpSampling1D, 
                                     MaxPool2D, Dropout, Permute, Flatten, TimeDistributed)

print('Starting')

# Replace CuDNNLSTM with LSTM since CuDNNLSTM is deprecated
# from tensorflow.keras.layers import CuDNNLSTM  # This is no longer needed
from tensorflow.keras.models import Model, Sequential
#################

def calculate_rpa(predicted, ground_truth):
    # Example RPA calculation (you may need to adjust this based on your requirements)
    diff = np.abs(predicted - ground_truth)
    rpa = np.mean(diff < 50)  # Percentage of frames with pitch error less than 50 cents
    return rpa

def calculate_vrr(predicted, ground_truth):
    # Example VRR calculation (you may need to adjust this based on your requirements)
    voiced_pred = predicted > 0
    voiced_gt = ground_truth > 0
    vrr = np.mean(voiced_pred == voiced_gt)
    return vrr
########    ############################
# Set file path to the model weights
filepath = "/speech/nishanth/raw_exps/models/awgn_40ms_res_time_singlescale_cnn_lstm_model_context__ptdb/weights-improvement-12-0.81.keras"
print(f"Loading model from: {filepath}")

# Load the model
model = load_model(filepath)
print(model.summary())

folders = ['clean','20','10','5','0']

dataset = "/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/test"
save_path = "/speech/nishanth/raw_exps/model_pred"

for folder in folders:
    os.makedirs(f"{save_path}/{folder}", exist_ok=True)
    path = f"{dataset}/{folder}/concat_frames.npy"
    print(f"Processing data from: {path}")

    X_train = np.load(path)
    seq_len = X_train.shape[1]

    X_test = X_train
    out_dim = 300
    X_predicted = model.predict(X_test)
    final_f0 = np.empty(len(X_predicted))

    for j in range(len(X_predicted)):
        frame = X_predicted[j,]
        center = int(np.argmax(frame))

        frame = frame.reshape(out_dim,)
        start = max(0, center - 4)
        end = min(len(frame), center + 5)

        X_check = frame[start:end]
        cents_mapping = (np.linspace(0, 5980, 300) + 1997.37)
        cents_mapping[0]= 0
        product_sum = np.sum(X_check * cents_mapping[start:end])
        weight_sum = np.sum(X_check)
        cents = product_sum / weight_sum

        final_f0[j] = cents

    # predict = np.power(2, final_f0 / 1200) * 10


    concat_cents = np.load(f"/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/test/{folder}/concat_cents.npy")
    print(concat_cents.shape, final_f0.shape)
    rpa = calculate_rpa(final_f0, concat_cents)
    vrr = calculate_vrr(final_f0, concat_cents)
    print(f"RPA : {rpa}\tVRR : {vrr}")

    result = np.stack((concat_cents, final_f0), axis=0)

    print(f"Saving results for folder: {folder}")
    np.savetxt(f"{save_path}/{folder}_result.txt", np.round(result))

    # Clear memory for the next loop
    del X_test, final_f0
