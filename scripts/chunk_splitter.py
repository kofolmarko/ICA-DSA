import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from helpers.progress_bar import printProgressBar

# Paths to data directories
parent_dir = '../simulator_data'
user_data_dir = os.path.join(parent_dir, 'user_data')
timestamps_dir = os.path.join(parent_dir, 'timestamps')

# Import user ids
with open('helpers/user_ids.txt', 'r') as f:
    lines = f.readlines()
    user_ids = [line.strip() for line in lines]

# Calculate the desired time step between frames in seconds
target_fps = 50
target_time_step = 1 / target_fps

# @ MAIN LOOP @ #
# Iterate over each directory
def iterate_for_scenario(scenario):
    # Print progress bar
    printProgressBar(0, len(user_ids), prefix = f'Splitting for scenario {scenario}:', suffix = 'Complete', length = 50)

    for idx, user_id in enumerate(user_ids):
        # Construct the filename for the csv and txt files
        generic_file_name = f'user_{user_id}_s{scenario}'
        csv_file = os.path.join(user_data_dir, generic_file_name) + '.csv'
        txt_file = os.path.join(timestamps_dir, generic_file_name) + '.txt'

        # Check if the .csv and .txt files exist in the current directory and import .csv data
        if os.path.isfile(csv_file) and os.path.isfile(txt_file):
            df_iterator = pd.read_csv(
                csv_file,
                usecols=[
                    'TIMESTAMP',
                    'AUTO_DRIVE',
                    'LEAD_TIME',
                    'HUD_5019',
                    'STEERING_WHEEL_ANGLE',
                    'SPEED',
                    'SPEED_LIMIT',
                    'ACCELERATION',
                    'ACCELERATION_Y'
                ],
                chunksize=1000000,
                low_memory=False,
                delimiter=';'
            )

            # Read the starting timestamp to sync with video
            with open(txt_file, 'r') as file:
                timestamp = int(file.read())

            # Declare resampled df
            resampled_df = pd.DataFrame()

            # Resample the data inside all chunks to match the video framerate
            for chunk in df_iterator:
                # Get the timestamps from the current chunk of data
                timestamps = chunk['TIMESTAMP'].to_numpy()

                # Convert to miliseconds, because microseconds are too much to handle
                timestamps = timestamps / 1000

                # Create an interpolation function for each column
                interp_auto_drive = interp1d(timestamps, chunk['AUTO_DRIVE'].to_numpy(), kind='linear')
                interp_lead_time = interp1d(timestamps, chunk['LEAD_TIME'].to_numpy(), kind='linear')
                interp_hud_5019 = interp1d(timestamps, chunk['HUD_5019'].to_numpy(), kind='linear')
                interp_steering_wheel_angle = interp1d(timestamps, chunk['STEERING_WHEEL_ANGLE'].to_numpy(), kind='linear')
                interp_speed = interp1d(timestamps, chunk['SPEED'].to_numpy(), kind='linear')
                interp_speed_limit = interp1d(timestamps, chunk['SPEED_LIMIT'].to_numpy(), kind='linear')
                interp_acceleration = interp1d(timestamps, chunk['ACCELERATION'].to_numpy(), kind='linear')
                interp_acceleration_y = interp1d(timestamps, chunk['ACCELERATION_Y'].to_numpy(), kind='linear')

                # Generate a new set of timestamps spaced by the target time step
                new_timestamps = np.arange(timestamps[0], timestamps[-1], target_time_step)

                # Interpolate the data for each column using the new timestamps
                resampled_auto_drive = interp_auto_drive(new_timestamps)
                resampled_lead_time = interp_lead_time(new_timestamps)
                resampled_hud_5019 = interp_hud_5019(new_timestamps)
                resampled_steering_wheel_angle = interp_steering_wheel_angle(new_timestamps)
                resampled_speed = interp_speed(new_timestamps)
                resampled_speed_limit = interp_speed_limit(new_timestamps)
                resampled_acceleration = interp_acceleration(new_timestamps)
                resampled_acceleration_y = interp_acceleration_y(new_timestamps)

                # ! There are more frames than data samples, most likely because of the resampling ! #
                # ? Right now this loop does nothing, but that's fine for out semi-automatic purposes ? #
                # Cut where video starts
                for i in range(len(new_timestamps)):
                    if new_timestamps[i] >= timestamp:
                        new_timestamps = new_timestamps[i:]
                        resampled_auto_drive = resampled_auto_drive[i:]
                        resampled_lead_time = resampled_lead_time[i:]
                        resampled_hud_5019 = resampled_hud_5019[i:]
                        resampled_steering_wheel_angle = resampled_steering_wheel_angle[i:]
                        resampled_speed = resampled_speed[i:]
                        resampled_speed_limit = resampled_speed_limit[i:]
                        resampled_acceleration = resampled_acceleration[i:]
                        resampled_acceleration_y = resampled_acceleration_y[i:]
                        break

                # Combine the resampled data into a single DataFrame
                resampled_df = pd.DataFrame({
                    'FRAME': range(len(new_timestamps)),
                    'TIMESTAMP': new_timestamps,
                    'AUTO_DRIVE': resampled_auto_drive,
                    'LEAD_TIME': resampled_lead_time,
                    'HUD_5019': resampled_hud_5019,
                    'STEERING_WHEEL_ANGLE': resampled_steering_wheel_angle,
                    'SPEED': resampled_speed,
                    'SPEED_LIMIT': resampled_speed_limit,
                    'ACCELERATION': resampled_acceleration,
                    'ACCELERATION_Y': resampled_acceleration_y,
                })

            # Initialize variables
            chunk_num = 0
            temp_df = pd.DataFrame()
            next_frames = 100

            # Create directory for chunks
            post_analysis_dir = '../post_analysis'
            parent_chunks_dir = 'chunks'
            child_chunks_dir = f'chunks_user_{user_id}_s{scenario}'

            if not os.path.exists(post_analysis_dir):
                os.mkdir(post_analysis_dir)

            if not os.path.exists(os.path.join(post_analysis_dir, parent_chunks_dir)):
                os.mkdir(os.path.join(post_analysis_dir, parent_chunks_dir))

            if not os.path.exists(os.path.join(post_analysis_dir, parent_chunks_dir, child_chunks_dir)):
                os.mkdir(os.path.join(post_analysis_dir, parent_chunks_dir, child_chunks_dir))

            # Iterate over rows and split into chunks
            # Only saving data, where TAKE OVER request or AUTO DRIVE request is present

            # Start saving chunks
            for i, row in resampled_df.iterrows():
                if row['LEAD_TIME'] > 0 or (row['HUD_5019'] > 0 and row['AUTO_DRIVE'] < 4):
                    temp_df = pd.concat([temp_df, row.to_frame().transpose()])
                elif not temp_df.empty:
                    # Prevent going over the last row if there are less left than next_frames
                    index = resampled_df.index.get_loc(i)
                    if (index + next_frames) > len(resampled_df):
                        next_rows = resampled_df.loc[i:len(resampled_df)]
                    else:
                        next_rows = resampled_df.loc[i:index + next_frames]
                    
                    # Declare current chunk path
                    current_chunk = f'chunk_{chunk_num}.csv'
                    current_chunk_path = os.path.join(post_analysis_dir, parent_chunks_dir, child_chunks_dir, current_chunk)

                    # Add next_frames to the chunk and save it
                    temp_df = pd.concat([temp_df, next_rows])
                    temp_df.to_csv(current_chunk_path, index=False)
                    chunk_num += 1
                    temp_df = pd.DataFrame()

            # Save any remaining data in temp_df
            if not temp_df.empty:
                temp_df.to_csv('temp/remaining_data.csv', index=False)

        # Print progress bar
        printProgressBar(idx + 1, len(user_ids), prefix = f'Splitting for scenario {scenario}:', suffix = 'Complete', length = 50)

# @ CALL MAIN LOOP @ #
# Each scenario once
iterate_for_scenario(1)
iterate_for_scenario(3)