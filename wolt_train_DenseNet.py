import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from WindowGenerator import *

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("orders_autumn_2020.txt")

# Combine all events into days
# and datetime conversion from string datatype to useful signal data
date_time = pd.to_datetime(df['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
df['TIMESTAMP'] = date_time
groupby = df.groupby(pd.Grouper(key = 'TIMESTAMP', freq = 'D'))
df = groupby.mean()
df['COUNTS'] = groupby.size()

timestamp = df.index.map(datetime.datetime.timestamp)
year = (365.2425)*24*60*60
df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

# This model will not incorporate location data so we can remove these features
df = df.drop(columns = ['USER_LONG','USER_LAT','VENUE_LONG','VENUE_LAT'])

#Plot some interesting features
#Plot the data
plt.figure(figsize = (12, 8))
plt.suptitle("FEATURE VISUALIZATION")
plot_cols = ['COUNTS', 'PRECIPITATION','ITEM_COUNT']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)

# Plotting feature correlation to identify any strong correlations in data.
# No badly correlating features found.
plt.figure(figsize = (12, 8))
plt.suptitle("FEATURE CORRELATION")
cor = df.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()

# Splitting data into training, validation, and testing sets.
# NOTE: This testing set is not directly used in this model and it is extracted later.
# The training and validation data have 20% overlap, because the daily training data is limited.
column_dict = {name : i for i, name in enumerate(df.columns)}
n = len(df)
train_df  = df[0:int(n*0.7)]
val_df = df[int(n*0.5):]
test_df = df[int(n*0.5):]

num_features = df.shape[1]
df.fillna(df.mean(),inplace = True)

#Normalize the training data
training_mean = train_df.mean()
training_std = train_df.std()

train_df = (train_df - training_mean) / training_std
val_df = (val_df - training_mean) / training_std
test_df = (test_df - training_mean) / training_std

#Visualize the distribution of features
df_std = (df - training_mean) / training_std
df_std = df_std.select_dtypes([np.number])
df_std = df_std.melt(var_name = "Features", value_name = "Normalized")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x = "Features", y = "Normalized", data = df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

# Initialize window for training using WindowGenerator class.
IN_STEPS = 23
OUT_STEPS = 7
WINDOW = IN_STEPS + OUT_STEPS
multi_window = WindowGenerator(input_width = IN_STEPS,
                               label_width = OUT_STEPS,
                               shift = OUT_STEPS,
                               train_df = train_df,
                               val_df = val_df,
                               test_df = test_df)

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])


# Train the data and plot the validation results using WindowGenerator.
history = compile_and_fit(multi_dense_model, multi_window)
IPython.display.clear_output()
multi_window.plot(multi_dense_model, plot_col = 'COUNTS')

# Make the forecast by sampling the latest test data.
data = np.array(test_df[-IN_STEPS:],dtype=np.float32)
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data = data,
        targets = None,
        sequence_length = IN_STEPS,
        sequence_stride = 1,
        shuffle = True,
        batch_size = 32)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
predictions = multi_dense_model.predict(ds, verbose = 0, callbacks=[early_stopping])
j = df.columns.get_loc('COUNTS')
count_predictions = tf.reshape(predictions[0,:,j],[1,OUT_STEPS]).numpy()*training_std[j]+training_mean[j]

#Plot the predicted values
plt.figure(figsize = (12, 8))
plt.suptitle('PREDICTION FOR NEXT 7 DAYS')
plt.plot((test_df['COUNTS'][-IN_STEPS:]).apply(lambda x: x*training_std[j] + training_mean[j]), label='Inputs', marker='.', zorder= -10)
prediction_indices = pd.date_range(start= test_df.index[-1] + pd.DateOffset(1), periods = OUT_STEPS, freq = 'D')
plt.scatter(prediction_indices, count_predictions,
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)
plt.xticks(rotation=90)
plt.legend()
plt.xlabel('Date')

#Plot the training and validation loss for each epoch.
plt.figure(figsize = (12, 8))
plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()




