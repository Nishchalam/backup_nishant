# import numpy as np
# import tensorflow as tf
# import os
# from tensorflow.keras import layers, models, callbacks
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate, Dropout, MaxPool1D, LSTM
# from tensorflow.keras.models import Model

# print('Starting')

# # Define directories for input and output files
# input_dir = '/speech/nishanth/raw_exps/full_final_data_crepe_16khz/train/raw/'
# output_dir = '/speech/nishanth/raw_exps/full_final_data_crepe_16khz/train/labels_one_hot/'

# # Get the list of input and output filenames
# X_train_file_names = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
# Y_train_file_names = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])

# # Make sure the lists are aligned (input-output pairs correspond to each other)
# file_paths = []
# for i in range(len(X_train_file_names)):
#     file_paths.append({
#         'input': os.path.join(input_dir, X_train_file_names[i]),
#         'output': os.path.join(output_dir, Y_train_file_names[i])
#     })

# # Split the data into training and validation sets
# split_ratio = 0.75
# split_index = int(len(file_paths) * split_ratio)

# train_file_paths = file_paths[:split_index]
# val_file_paths = file_paths[split_index:]

# # Parameters
# context = 0
# dropout_factor = 0.25
# out_dim = 300

# # Custom Data Generator
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, file_paths, batch_size=64, seq_len=512, n_dim=3, shuffle=True):
#         self.file_paths = file_paths
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.n_dim = n_dim
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.floor(len(self.file_paths) / self.batch_size))

#     def __getitem__(self, index):
#         batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
#         X, Y = self.__data_generation(batch_files)
#         return X, Y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.file_paths)

#     def __data_generation(self, batch_files):
#         X = np.empty((self.batch_size, self.seq_len, self.n_dim))
#         Y = np.empty((self.batch_size, out_dim))

#         for i, file_pair in enumerate(batch_files):
#             input_data = np.load(file_pair['input'])
#             output_data = np.load(file_pair['output'])

#             # Extract relevant context
#             input_data = input_data[:, 2-context:2+context+1]

#             X[i,] = input_data
#             Y[i,] = output_data

#         return X, Y

# # Define the model architecture
# def MultiBranchConv1D(input, filters1, kernel_size1, strides1, strides2):
#     x1 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides1, padding='same', activation='relu')(input)
#     x1 = Dropout(0.25)(x1)
    
#     x2 = Conv1D(filters=filters1, kernel_size=kernel_size1+6, strides=strides1, padding='same', activation='relu')(input)
#     x2 = Dropout(0.25)(x2)
    
#     x3 = Conv1D(filters=filters1, kernel_size=kernel_size1+12, strides=strides1, padding='same', activation='relu')(input)
#     x3 = Dropout(0.25)(x3)
    
#     y1 = concatenate([x1, x2, x3], axis=-1)

#     x4 = Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides2, padding='same', activation='relu')(y1)
#     x4 = Dropout(0.25)(x4)
    
#     x5 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides2, padding='same', activation='relu')(y1)
#     x5 = Dropout(0.25)(x5)
    
#     x6 = Conv1D(filters=filters1, kernel_size=kernel_size1+4, strides=strides2, padding='same', activation='relu')(y1)
#     x6 = Dropout(0.25)(x6)
    
#     x = concatenate([x4, x5, x6], axis=-1)
#     return x

# def TempPyramid(input_f):
#     conv1 = MultiBranchConv1D(input_f, 64, 3, 2, 2)
#     return conv1

# seq_input = Input(shape=(512, 3), name='full_scale')  # Adjust shape according to your data
# concat = TempPyramid(seq_input)

# out = MaxPool1D()(concat)
# out = MaxPool1D()(out)
# out = LSTM(512)(out)
# out = Dense(300, activation='softmax')(out)

# model = Model(inputs=seq_input, outputs=out)

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20)

# # Update the file path with the correct .keras extension
# filepath = '/models/awgn_40ms_res_time_singlescale_cnn_lstm_model_context_' + str(context) + '_ptdb/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras'
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [earlystopping, checkpoint]

# # Initialize data generators for training and validation
# training_generator = DataGenerator(train_file_paths, batch_size=64, seq_len=512, n_dim=3, shuffle=True)
# validation_generator = DataGenerator(val_file_paths, batch_size=64, seq_len=512, n_dim=3, shuffle=False)

# # Train the model
# model.fit(training_generator, validation_data=validation_generator, epochs=200, verbose=1, shuffle=True, callbacks=callbacks_list)

# print("Saved model to disk")
# print('Training completed')
# print('Done')







import numpy as np

import tensorflow as tf
print('Starting')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Conv1D, LSTM, Dense, Activation, concatenate, UpSampling1D
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense, TimeDistributed

# Replace LSTM with LSTM
# from tensorflow.keras.layers import CuDNNLSTM  # This is no longer needed
from tensorflow.keras.models import Model, Sequential

# Now you can use LSTM instead of CuDNNLSTM



np.random.seed(1)

input_array = np.load('/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/train/concat_frames.npy')

randomize = np.arange(len(input_array))
np.random.shuffle(randomize)
X_train = input_array[randomize]
del input_array

output_array = np.load('/speech/nishanth/raw_exps/5_fold_rev_final_data_8khz/train/concat_labels.npy')
Y_train = output_array[randomize]
del output_array

# context=0
# X_train = X_train[:,:,2-context:2+context+1]
print('data loaded')
print(X_train.shape)
print(Y_train.shape)

dropout_factor = 0.25
out_dim = 300
n_dim = X_train.shape[2]
# Build the model ...

seq_len = X_train.shape[1]


#########################################################################################################################
def MultiBranchConv1D(input, filters1, kernel_size1, strides1, strides2):
    x1 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides1, padding='same', activation='relu')(input)
    x1 = Dropout(0.25)(x1)
    
    x2 = Conv1D(filters=filters1, kernel_size=kernel_size1+6, strides=strides1, padding='same', activation='relu')(input)
    x2 = Dropout(0.25)(x2)
    
    x3 = Conv1D(filters=filters1, kernel_size=kernel_size1+12, strides=strides1, padding='same', activation='relu')(input)
    x3 = Dropout(0.25)(x3)
    
    y1 = concatenate([x1, x2, x3], axis=-1)

    x4 = Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides2, padding='same', activation='relu')(y1)
    x4 = Dropout(0.25)(x4)
    
    x5 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides2, padding='same', activation='relu')(y1)
    x5 = Dropout(0.25)(x5)
    
    x6 = Conv1D(filters=filters1, kernel_size=kernel_size1+4, strides=strides2, padding='same', activation='relu')(y1)
    x6 = Dropout(0.25)(x6)
    
    x = concatenate([x4, x5, x6], axis=-1)
    return x

#########################################################################################################################
# Define a temporal pyramid network
def TempPyramid(input_f):

    #### Full scale sequences
    conv1 = MultiBranchConv1D(input_f, 64, 3, 2, 2)

    #### Half scale sequences
    #conv2 = MultiBranchConv1D(input_2, 64, 3, 2, 1)

    #### Recurrent layers
    #x = concatenate([conv1, conv2], axis=-1)
    return conv1

#########################################################################################################################


# Build the model ...

#### Full scale sequences
seq_input = Input(shape = (seq_len,n_dim), name = 'full_scale')
#### Half scale sequences
#seq_input_2 = Input(shape=(int(seq_len/2), n_dim), name='half_scale')
concat = TempPyramid(seq_input)
#########################################################################################################################
out = layers.MaxPool1D()(concat)
out = layers.MaxPool1D()(out)

#Add LSTM layer
#out = LSTM(512, return_sequences=True)(out)
out =LSTM (512)(out)
#############################################################

#out = layers.Flatten()(out)
#out = Dense(512, activation = 'relu')(out)
out = Dense(300, activation = 'softmax')(out)


model = Model(inputs=seq_input, outputs=out)


#########################################################################################################################

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
#loss = tensorflow.keras.losses.categorical_crossentropy()

#'binary_crossentropy' sigmoid
# sparse_categorical_crossentropy softmax
model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20)
#checkpoint
filepath='models/awgn_40ms_res_time_singlescale_cnn_lstm_model_context_'+'_ptdb/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [earlystopping,checkpoint]


#callbacks_list = [earlystopping]

print(model.summary())
#########################################################################################################################
X_train = X_train.reshape(len(X_train),seq_len,n_dim)
#X_train_2 = X_train[:,0::2,:]



########

########



#########################################################################################################################
model.fit(X_train, Y_train, validation_split=0.25, epochs=200, batch_size=64, verbose=1, shuffle=True, callbacks=callbacks_list)

print("Saved model to disk")
print('Training completed')

#-------------------------------------------------------------------------------------------------------------------------

print('Done')
