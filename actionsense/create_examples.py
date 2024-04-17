import h5py
import numpy as np
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict
import os, glob
script_dir = os.path.dirname(os.path.realpath(__file__))

from helpers import *
from utils.print_utils import *
from utils.dict_utils import *
from utils.time_utils import *

# config
# Define segmentation parameters.
resampled_Fs = 50 # define a resampling rate for all sensors to interpolate
segment_duration_s = 2
segment_length = int(round(resampled_Fs*segment_duration_s))

# Define filtering parameters.
filter_cutoff_emg_Hz = 5
filter_cutoff_tactile_Hz = 2
filter_cutoff_gaze_Hz = 5
num_tactile_rows_aggregated = 4
num_tactile_cols_aggregated = 4

def process(output_dir, output_file, input_dir):
  #######################################
  ############ CONFIGURATION ############
  #######################################

  # Define where outputs will be saved.
  output_filepath = os.path.join(output_dir, output_file)

  # Specify the input data.
  # data_root_dir = "data/experiments"
  data_folders_bySubject = OrderedDict([
    ('unknown id subject', input_dir),
    # ('S00', os.path.join(data_root_dir, '2022-06-07_experiment_S00')),
    # ('S02', os.path.join(data_root_dir, '2022-06-13_experiment_S02')),
    # ('S03', os.path.join(data_root_dir, '2022-06-14_experiment_S03')),
    # ('S04', os.path.join(data_root_dir, '2022-06-14_experiment_S04')),
    # ('S05', os.path.join(data_root_dir, '2022-06-14_experiment_S05')),
  ])

  # Define the modalities to use.
  # Each entry is (device_name, stream_name, extraction_function)
  #  where extraction_function can select a subset of the stream columns.
  device_streams_for_features = [
    ('eye-tracking-gaze', 'position', lambda data: data),
    ('myo-left', 'emg', lambda data: data),
    ('myo-right', 'emg', lambda data: data),
    # ('tactile-glove-left', 'tactile_data', lambda data: data),
    # ('tactile-glove-right', 'tactile_data', lambda data: data),
    ('xsens-joints', 'rotation_xzy_deg', lambda data: data[:,0:22,:]), # exclude fingers by using the first 22 joints
  ]

  # Make the output folder if needed.
  if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    print('\n')
    print('Saving outputs to')
    print(output_filepath)
    print('\n')
    
  ################################################
  ############ INTERPOLATE AND FILTER ############
  ################################################

  # Will filter each column of the data.
  def lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data.T).T
    return y

  # Load the original data.
  data_bySubject = {}
  for (subject_id, data_folder) in data_folders_bySubject.items():
    print()
    print('Loading data for subject %s' % subject_id)
    data_bySubject[subject_id] = []
    hdf_filepaths = glob.glob(os.path.join(data_folder, '**/*.hdf5'), recursive=True)
    for hdf_filepath in hdf_filepaths:
      if 'archived' in hdf_filepath:
        continue
      data_bySubject[subject_id].append({})
      with h5py.File(hdf_filepath, 'r') as hdf_file:  # It's better to use context manager to ensure file closure
        print(hdf_filepath)
        # Add the activity label information.
        have_all_streams = True
        try:
          device_name = 'experiment-activities'
          stream_name = 'activities'
          data_bySubject[subject_id][-1].setdefault(device_name, {})
          data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
          for key in ['time_s', 'data']:
            data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
          num_activity_entries = len(data_bySubject[subject_id][-1][device_name][stream_name]['time_s'])
          if num_activity_entries == 0:
            have_all_streams = False
          elif data_bySubject[subject_id][-1][device_name][stream_name]['time_s'][0] == 0:
            have_all_streams = False
        except KeyError:
          have_all_streams = False

        # Load data for each of the streams that will be used as features.
        for (device_name, stream_name, _) in device_streams_for_features:
          data_bySubject[subject_id][-1].setdefault(device_name, {})
          data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
          for key in ['time_s', 'data']:
            try:
              data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
            except KeyError:
              have_all_streams = False

        # Add the eye-tracking video world gaze frame timestamps
        try:
          data_bySubject[subject_id][-1].setdefault('eye-tracking-video-worldGaze', {})
          data_bySubject[subject_id][-1]['eye-tracking-video-worldGaze'].setdefault('frame_timestamp', {})
          data_bySubject[subject_id][-1]['eye-tracking-video-worldGaze']['frame_timestamp']['time_s'] = \
            hdf_file['eye-tracking-video-worldGaze']['frame_timestamp']['time_s'][:]
        except KeyError as e:
          print('KeyError:', e)
          have_all_streams = False

        if not have_all_streams:
          data_bySubject[subject_id].pop()
          print('  Ignoring HDF5 file:', hdf_filepath)
      
  # Filter data.
  print()
  for (subject_id, file_datas) in data_bySubject.items():
    print('Filtering data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
      print(' Data file index', data_file_index)
      # Filter EMG data.
      for myo_key in ['myo-left', 'myo-right']:
        if myo_key in file_data:
          t = file_data[myo_key]['emg']['time_s']
          Fs = (t.size - 1) / (t[-1] - t[0])
          print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (myo_key, Fs, filter_cutoff_emg_Hz))
          data_stream = file_data[myo_key]['emg']['data'][:, :]
          y = np.abs(data_stream)
          y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
          # plt.plot(t-t[0], data_stream[:,0])
          # plt.plot(t-t[0], y[:,0])
          # plt.show()
          file_data[myo_key]['emg']['data'] = y
      # Filter tactile data.
      for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
        if tactile_key in file_data:
          t = file_data[tactile_key]['tactile_data']['time_s']
          Fs = (t.size - 1) / (t[-1] - t[0])
          print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (tactile_key, Fs, filter_cutoff_tactile_Hz))
          data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
          y = data_stream
          y = lowpass_filter(y, filter_cutoff_tactile_Hz, Fs)
          # Eliminate ringing at beginning or end.
          y[0:int(Fs*30),:,:] = np.mean(y, axis=0)
          y[y.shape[0]-int(Fs*30):y.shape[0]+1,:,:] = np.mean(y, axis=0)
          # plt.plot(t-t[0], data_stream[:,0,0])
          # plt.plot(t-t[0], y[:,0,0])
          # plt.xlim(10,t[-1]-t[0])
          # plt.ylim(550,570)
          # plt.show()
          file_data[tactile_key]['tactile_data']['data'] = y
      # Filter eye-gaze data.
      if 'eye-tracking-gaze' in file_data:
        t = file_data['eye-tracking-gaze']['position']['time_s']
        Fs = (t.size - 1) / (t[-1] - t[0])
        
        data_stream = file_data['eye-tracking-gaze']['position']['data'][:, :]
        y = data_stream
        
        # Apply a ZOH to remove clipped values.
        #  The gaze position is already normalized to video coordinates,
        #   so anything outside [0,1] is outside the video.
        print(' Holding clipped values in %s' % ('eye-tracking-gaze'))
        clip_low = 0.05
        clip_high = 0.95
        y = np.clip(y, clip_low, clip_high)
        y[y == clip_low] = np.nan
        y[y == clip_high] = np.nan
        y = pd.DataFrame(y).interpolate(method='zero').to_numpy()
        # Replace any remaining NaNs with a dummy value,
        #  in case the first or last timestep was clipped (interpolate() does not extrapolate).
        y[np.isnan(y)] = 0.5
        # plt.plot(t-t[0], data_stream[:,0], '*-')
        # plt.plot(t-t[0], y[:,0], '*-')
        # plt.ylim(-2,2)
        
        # Filter to smooth.
        print('   Filtering %s with Fs %0.1f Hz to cutoff %f' % ('eye-tracking-gaze', Fs, filter_cutoff_gaze_Hz))
        y = lowpass_filter(y, filter_cutoff_gaze_Hz, Fs)
        # plt.plot(t-t[0], y[:,0])
        # plt.ylim(-2,2)
        # plt.show()
        file_data['eye-tracking-gaze']['position']['data'] = y
      data_bySubject[subject_id][data_file_index] = file_data

  # Normalize data.
  print()
  for (subject_id, file_datas) in data_bySubject.items():
    print('Normalizing data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
      # Normalize EMG data.
      for myo_key in ['myo-left', 'myo-right']:
        if myo_key in file_data:
          data_stream = file_data[myo_key]['emg']['data'][:, :]
          y = data_stream
          print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (myo_key, np.amin(y), np.amax(y)))
          # Normalize them jointly.
          y = y / ((np.amax(y) - np.amin(y))/2)
          # Jointly shift the baseline to -1 instead of 0.
          y = y - np.amin(y) - 1
          file_data[myo_key]['emg']['data'] = y
          print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
          # plt.plot(y.reshape(y.shape[0], -1))
          # plt.show()
      # Normalize tactile data.
      # NOTE: Will clip here, but will normalize later after aggregating.
      for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
        if tactile_key in file_data:
          data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
          y = data_stream
          min_val = np.amin(y)
          max_val = np.amax(y)
          print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (tactile_key, min_val, max_val))
          # Clip the values based on the distribution of values across all channels.
          mean_val = np.mean(y)
          std_dev = np.std(y)
          clip_low = mean_val - 2*std_dev # shouldn't be much below the mean, since the mean should be rest basically
          clip_high = mean_val + 3*std_dev
          print('  Clipping to [%0.1f, %0.1f]' % (clip_low, clip_high))
          # heatmap(np.mean(y, axis=0), 'Pre clipping')
          y = np.clip(y, clip_low, clip_high)
          # heatmap(np.mean(y, axis=0), 'Post clipping')
          # input()
          # Store the result.
          file_data[tactile_key]['tactile_data']['data'] = y
      # Normalize Xsens joints.
      if 'xsens-joints' in file_data:
        data_stream = file_data['xsens-joints']['rotation_xzy_deg']['data'][:, :]
        y = data_stream
        min_val = -180
        max_val = 180
        print(' Normalizing %s with forced min/max [%0.1f, %0.1f]' % ('xsens-joints', min_val, max_val))
        # Normalize all at once since using fixed bounds anyway.
        # Preserve relative bends, such as left arm being bent more than the right.
        y = y / ((max_val - min_val)/2)
        # for i in range(20):
        #   plt.plot(y[:,i])
        #   plt.ylim(-1,1)
        #   plt.show()
        file_data['xsens-joints']['rotation_xzy_deg']['data'] = y
        print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
        # plt.plot(y.reshape(y.shape[0], -1))
        # plt.show()
      # Normalize eye-tracking gaze.
      if 'eye-tracking-gaze' in file_data:
        data_stream = file_data['eye-tracking-gaze']['position']['data'][:]
        t = file_data['eye-tracking-gaze']['position']['time_s'][:]
        y = data_stream
        # The gaze position is already normalized to video coordinates,
        #  so anything outside [0,1] is outside the video.
        clip_low = 0.05
        clip_high = 0.95
        print(' Clipping %s to [%0.1f, %0.1f]' % ('eye-tracking-gaze', clip_low, clip_high))
        y = np.clip(y, clip_low, clip_high)
        # Put in range [-1, 1] for extra resolution.
        y = (y - np.mean([clip_low, clip_high]))/((clip_high-clip_low)/2)
        # plt.plot(t-t[0], y)
        # plt.show()
        file_data['eye-tracking-gaze']['position']['data'] = y
        print('   Now has range [%0.1f, %0.1f]' % (np.amin(y), np.amax(y)))
        # plt.plot(y.reshape(y.shape[0], -1))
        # plt.show()
        
      data_bySubject[subject_id][data_file_index] = file_data

  # Aggregate data (and normalize if needed).
  print()
  for (subject_id, file_datas) in data_bySubject.items():
    print('Aggregating data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
      # Aggregate EMG data.
      for myo_key in ['myo-left', 'myo-right']:
        if myo_key in file_data:
          pass
      # Aggregate tactile data.
      for tactile_key in ['tactile-glove-left', 'tactile-glove-right']:
        if tactile_key in file_data:
          data_stream = file_data[tactile_key]['tactile_data']['data'][:, :]
          y = data_stream
          # Make a smaller grid of values, averaging the channels they contain.
          num_rows = y.shape[1]
          num_cols = y.shape[2]
          row_stride = int(num_rows / num_tactile_rows_aggregated + 0.5)
          col_stride = int(num_rows / num_tactile_cols_aggregated + 0.5)
          data_aggregated = np.zeros(shape=(y.shape[0], num_tactile_rows_aggregated, num_tactile_cols_aggregated))
          for r, row_offset in enumerate(range(0, num_rows, row_stride)):
            for c, col_offset in enumerate(range(0, num_cols, col_stride)):
              mask = np.zeros(shape=(num_rows, num_cols))
              mask[row_offset:(row_offset+row_stride), col_offset:(col_offset+col_stride)] = 1
              data_aggregated[:,r,c] = np.sum(y*mask, axis=(1,2))/np.sum(mask)
          y = data_aggregated
          # # De-mean each channel individually.
          # y = y - np.mean(y, axis=0)
          # Normalize all channels jointly.
          y = y / ((np.amax(y) - np.amin(y))/2)
          # Shift baseline to -1 jointly.
          y = y - np.amin(y) - 1
          # Store the result.
          file_data[tactile_key]['tactile_data']['data'] = y
          print('  Tactile now has shape %s and now has range [%0.1f, %0.1f]' % (y.shape, np.amin(y), np.amax(y)))
          # plt.plot(y.reshape(y.shape[0], -1))
          # plt.show()
      # Aggregate Xsens joints.
      if 'xsens-joints' in file_data:
        pass
      # Aggregate eye-tracking gaze.
      if 'eye-tracking-gaze' in file_data:
        pass
      
      data_bySubject[subject_id][data_file_index] = file_data
      
  # Resample data.
  print()
  for (subject_id, file_datas) in data_bySubject.items():
    print('Resampling data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
      for (device_name, stream_name, _) in device_streams_for_features:
        data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
        time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
        target_time_s = np.linspace(time_s[0], time_s[-1],
                                    num=int(round(1+resampled_Fs*(time_s[-1] - time_s[0]))),
                                    endpoint=True)
        fn_interpolate = interpolate.interp1d(
            time_s, # x values
            data,   # y values
            axis=0,              # axis of the data along which to interpolate
            kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
            fill_value='extrapolate' # how to handle x values outside the original range
        )
        data_resampled = fn_interpolate(target_time_s)
        if np.any(np.isnan(data_resampled)):
          print('\n'*5)
          print('='*50)
          print('='*50)
          print('FOUND NAN')
          print(subject_id, device_name, stream_name)
          timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1,np.ndim(data_resampled))))
          print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
          print_var(data_resampled)
          # input('Press enter to continue ')
          print('\n'*5)
          time.sleep(10)
          data_resampled[np.isnan(data_resampled)] = 0
        file_data[device_name][stream_name]['time_s'] = target_time_s
        file_data[device_name][stream_name]['data'] = data_resampled
      data_bySubject[subject_id][data_file_index] = file_data

  #########################################
  ############ CREATE FEATURES ############
  #########################################

  def get_feature_matrix(experiment_data, start_time_s, end_time_s):
    feature_matrix = np.empty(shape=(segment_length, 0))
    for (device_name, stream_name, extraction_fn) in device_streams_for_features:
      # print(' Adding data from [%s][%s]' % (device_name, stream_name))
      data = np.squeeze(np.array(experiment_data[device_name][stream_name]['data']))
      time_s = np.squeeze(np.array(experiment_data[device_name][stream_name]['time_s']))
      time_indexes = np.where((time_s >= start_time_s) & (time_s <= end_time_s))[0]
      # Expand if needed until the desired segment length is reached.
      time_indexes = list(time_indexes)
      while len(time_indexes) < segment_length:
        print(' Increasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), segment_length, device_name, stream_name, start_time_s))
        if time_indexes[0] > 0:
          time_indexes = [time_indexes[0]-1] + time_indexes
        elif time_indexes[-1] < len(time_s)-1:
          time_indexes.append(time_indexes[-1]+1)
        else:
          raise AssertionError
      while len(time_indexes) > segment_length:
        print(' Decreasing segment length from %d to %d for %s %s for segment starting at %f' % (len(time_indexes), segment_length, device_name, stream_name, start_time_s))
        time_indexes.pop()
      time_indexes = np.array(time_indexes)
      
      # Extract the data.
      time_s = time_s[time_indexes]
      data = data[time_indexes,:]
      data = extraction_fn(data)
      # print('  Got data of shape', data.shape)
      # Add it to the feature matrix.
      data = np.reshape(data, (segment_length, -1))
      if np.any(np.isnan(data)):
        print('\n'*5)
        print('='*50)
        print('='*50)
        print('FOUND NAN')
        print(device_name, stream_name, start_time_s)
        timesteps_have_nan = np.any(np.isnan(data), axis=tuple(np.arange(1,np.ndim(data))))
        print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
        print_var(data)
        # input('Press enter to continue ')
        print('\n'*5)
        time.sleep(10)
        data[np.isnan(data)] = 0
      feature_matrix = np.concatenate((feature_matrix, data), axis=1)
    return feature_matrix

  def create_segments_from_frames(first_frame_time, last_frame_time, segment_length_s=segment_duration_s):
    current_time = first_frame_time
    segment_times = []
    while current_time + segment_length_s <= last_frame_time:
      segment_times.append((current_time, current_time + segment_length_s))
      current_time += segment_length_s
    if current_time < last_frame_time:
      segment_times.append((current_time, last_frame_time))
    return segment_times

  example_matrices = []
  example_labels = []
  example_segment_indices = []

  for (subject_id, file_datas) in data_bySubject.items():
    for (data_file_index, file_data) in enumerate(file_datas):
      # get activity labels
      activity_datas = [[x.decode('utf-8') for x in datas] for datas in file_data['experiment-activities']['activities']['data']]
      activity_times_s = file_data['experiment-activities']['activities']['time_s']
      activities_labels = []
      activities_start_times_s = []
      activities_end_times_s = []
      for (row_index, time_s) in enumerate(activity_times_s):
        label    = activity_datas[row_index][0]
        is_start = activity_datas[row_index][1] == 'Start'
        is_stop  = activity_datas[row_index][1] == 'Stop'
        if is_start:
          activities_labels.append(label)
          activities_start_times_s.append(time_s)
        if is_stop:
          activities_end_times_s.append(time_s)

      frame_timestamps = file_data['eye-tracking-video-worldGaze']['frame_timestamp']['time_s']
      segment_times = create_segments_from_frames(frame_timestamps[0].item(), frame_timestamps[-1].item())
      # Now process each segment to get the feature matrices
      for segment_index, (start_time, end_time) in enumerate(segment_times):
        # if there is an error (usually caused by the entire segment being outside the range), 
        # then fill in None for the matrix, and later, replace None with np.zeros of the correct shape
        try:
          feature_matrices = get_feature_matrix(file_data, start_time, end_time)
          default_example_matrix = np.zeros_like(feature_matrices)
        except:
          print("error in get_features_matrix, skipping it, segment index:", segment_index)
          feature_matrices = None
        example_label = ""
        for i in range(len(activities_labels)):
          if activities_start_times_s[i] <= start_time <= activities_end_times_s[i]:
            example_label = activities_labels[i]

        example_matrices.append(feature_matrices)
        example_labels.append(example_label)
        example_segment_indices.append(segment_index)

  for i in range(len(example_matrices)):
    if example_matrices[i] is None:
      example_matrices[i] = default_example_matrix

  example_matrices = np.array(example_matrices)
  example_segment_indices = np.array(example_segment_indices)

  # my own normalization stuff
  # kind of dumb becuase I concatenated, then unconcatenated, but its fine
  print(example_matrices.shape)
  assert example_matrices.shape[2] == 84
  # normalize to between [-1, 1]
  def normalize(data):
    normalized_data = 2 * ((data - np.min(data)) / (np.max(data) - np.min(data))) - 1
    return normalized_data
  eye_data = example_matrices[:,:,0:2]
  emg_data = example_matrices[:,:,2:18]
  body_data = example_matrices[:,:,18:84]
  eye_data = normalize(eye_data)
  emg_data = normalize(emg_data)
  body_data = normalize(body_data)
  example_matrices = np.concatenate([eye_data, emg_data, body_data], axis=-1)
  print(example_matrices.shape)

  # saving to HDF5
  if output_filepath is not None:
    with h5py.File(output_filepath, 'w') as hdf_file:
      hdf_file.create_dataset('example_matrices', data=example_matrices)
      hdf_file.create_dataset('example_labels', data=example_labels)
      hdf_file.create_dataset('example_segment_indices', data=example_segment_indices)

      print()
      print('Saved processed data to', output_filepath)
      print()

output_dir = "data/actionsense_processed"
input_parent_dir = "actionsense/experiments"
# subdirs = ["S00"]
subdirs = os.listdir(input_parent_dir)
for subdir in subdirs:
  if subdir == ".DS_Store":
    continue
  # might error if subdir is DS_Store
  print(subdir)
  process(
    output_dir=output_dir,
    output_file=f"{subdir}.hdf5",
    input_dir=f"{input_parent_dir}/{subdir}"
    )