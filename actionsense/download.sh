#!/bin/bash

# array of URLs to files
urls=(
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_streamLog_actionNet-wearables_S02.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-12_experiment_S06/2022-07-12_15-07-50_actionNet-wearables_S06/2022-07-12_15-08-08_streamLog_actionNet-wearables_S06.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S07/2022-07-13_11-01-18_actionNet-wearables_S07/2022-07-13_11-02-03_streamLog_actionNet-wearables_S07.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-13_experiment_S08/2022-07-13_14-15-03_actionNet-wearables_S08/2022-07-13_14-15-26_streamLog_actionNet-wearables_S08.hdf5"
"https://data.csail.mit.edu/ActionNet/wearable_data/2022-07-14_experiment_S09/2022-07-14_09-58-40_actionNet-wearables_S09/2022-07-14_09-59-00_streamLog_actionNet-wearables_S09.hdf5"
)

# array of target directories (make sure this lines up with urls)
dirs=(
"experiments/S00"
"experiments/S02_1"
"experiments/S02_2"
"experiments/S02_3"
"experiments/S03"
"experiments/S04"
"experiments/S05"
"experiments/S06"
"experiments/S07"
"experiments/S08"
"experiments/S09"
)

for i in "${!urls[@]}"; do 
  mkdir -p "${dirs[$i]}"
  wget -P "${dirs[$i]}" "${urls[$i]}"
done