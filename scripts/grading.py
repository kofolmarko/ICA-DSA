import os
import pandas as pd

# @ GLOBAL @ #
# Construct dataframe structure
data = {
    'USER': [], # user ID
    'HUD': [], # user has HUD
    'REQUEST_TYPE': [], # type of transition, either from AUTOMATIC to MANUAL or vice versa
    'ADL_UNDER_THRESHOLD': [], # switched to auto mode in under 5 seconds
    'ADL_FAIL': [], # unintended ADL activation
    'ADL_ROAD': [], # % of road watchtime after ADL activation (5s)
    'ADL_DISTRACTION': [], # % of distraction watchtime after ADL activation (5s)
    'ADL_OTHER_OR_UNDEFINED': [], # % of undefined looking at frames (either nothing relevant or eyetracker lost)
    'TOR_RT': [], # reaction time between TOR and TO
    'TOR_RESPONSE': [], # did the user respond to visual or auditory warning or did he not respond at all
    'TOR_MIRRORS': [], # checked at least one mirror
    'TOR_ROAD': [], # % of road watchtime between TOR and TO
    'TOR_DASHBOARD': [],
    'TOR_HUD': [],
    'TOR_OTHER_OR_UNDEFINED': [], # % of undefined looking at frames (either nothing relevant or eyetracker lost)
    'TOR_SPEEDING': [], # % of time speeding after TO (5s)
    'TOR_ACC': [], # % of time acceleration exceeds a threshold (5s)
    'TOR_DCC': [], # % of time decceleration exceeds a threshold (5s)
    'TOR_ACC_Y': [], # % of time acceleration_y exceeds a threshold (5s)
}

FPS = 50

# @ FUNCTIONS @ #
def fill_empty_tor():
    data['TOR_RT'].append('-')
    data['TOR_RESPONSE'].append('-')
    data['TOR_MIRRORS'].append('-')
    data['TOR_ROAD'].append('-')
    data['TOR_DASHBOARD'].append('-')
    data['TOR_HUD'].append('-')
    data['TOR_OTHER_OR_UNDEFINED'].append('-')
    data['TOR_SPEEDING'].append('-')
    data['TOR_ACC'].append('-')
    data['TOR_DCC'].append('-')
    data['TOR_ACC_Y'].append('-')

def fill_empty_adl():
    data['ADL_UNDER_THRESHOLD'].append('-')
    data['ADL_ROAD'].append('-')
    data['ADL_DISTRACTION'].append('-')
    data['ADL_OTHER_OR_UNDEFINED'].append('-')

def check_adl_switch(chunk: pd.DataFrame):
    threshold = 5 * FPS # 5 seconds

    mask = (chunk['DRIVING_MODE'] == 'MANUAL')
    frames_before_adl = len(chunk[mask])

    if frames_before_adl < threshold:
        data['ADL_UNDER_THRESHOLD'].append(True)
    else:
        data['ADL_UNDER_THRESHOLD'].append(False)

def adl_looking(chunk: pd.DataFrame):
    mask = (chunk['DRIVING_MODE'] == 'AUTOMATIC')

    all_frames = len(chunk[mask].head(5 * FPS))
    road_frames = 0
    disctraction_frames = 0
    other_frames = 0

    for idx, row in chunk[mask].head(5 * FPS).iterrows():
        objects = str(row['SEEN_OBJECTS']).split(',')

        if 'road' in objects:
            road_frames += 1
        elif 'distraction' in objects:
            disctraction_frames += 1
        else:
            other_frames += 1

    data['ADL_ROAD'].append(road_frames / all_frames)
    data['ADL_DISTRACTION'].append(disctraction_frames / all_frames)
    data['ADL_OTHER_OR_UNDEFINED'].append(other_frames / all_frames)

def tor_reaction(chunk: pd.DataFrame):
    mask = (chunk['DRIVING_MODE'] == 'AUTOMATIC')

    rt = len(chunk[mask]) / FPS
    data['TOR_RT'].append(rt)

    visual_threshold = 10
    auditory_threshold = 15

    if rt < visual_threshold:
        data['TOR_RESPONSE'].append('VISUAL')
    elif rt < auditory_threshold:
        data['TOR_RESPONSE'].append('AUDITORY')
    else:
        data['TOR_RESPONSE'].append('NO_RESPONSE')

def tor_looking(chunk: pd.DataFrame):
    mask = (chunk['DRIVING_MODE'] == 'AUTOMATIC')

    all_frames = len(chunk[mask])
    mirror_frames = 0
    road_frames = 0
    dashboard_frames = 0
    hud_frames = 0
    other_frames = 0

    for idx, row in chunk[mask].iterrows():
        objects = str(row['SEEN_OBJECTS']).split(',')

        if 'road' in objects:
            road_frames += 1
        elif 'mirror' in objects or 'rearview_mirror' in objects or 'right_mirror' in objects or 'left_mirror' in objects:
            mirror_frames += 1
        elif 'dashboard' in objects:
            dashboard_frames += 1
        elif 'header_display' in objects:
            hud_frames += 1
        else:
            other_frames += 1

    data['TOR_MIRRORS'].append(mirror_frames / all_frames)
    data['TOR_ROAD'].append(road_frames / all_frames)
    data['TOR_DASHBOARD'].append(dashboard_frames / all_frames)
    data['TOR_HUD'].append(hud_frames / all_frames)
    data['TOR_OTHER_OR_UNDEFINED'].append(other_frames / all_frames)

def after_to(chunk: pd.DataFrame):
    mask = (chunk['DRIVING_MODE'] == 'MANUAL')
    chunk = chunk[mask].head(5 * FPS)

    all_frames = len(chunk)

    data['TOR_SPEEDING'].append(chunk[(chunk['SPEED'] > (chunk['SPEED_LIMIT'])*1.10)]['FRAME'].count() / all_frames)
    data['TOR_ACC'].append(chunk[(chunk['ACCELERATION'] > 1.0)]['ACCELERATION'].count() / all_frames)
    data['TOR_DCC'].append(chunk[(chunk['ACCELERATION'] < -2.0)]['ACCELERATION'].count() / all_frames)
    data['TOR_ACC_Y'].append(chunk[(chunk['ACCELERATION_Y'].abs() > 0.5)]['ACCELERATION_Y'].count() / all_frames)

# @ MAIN LOOP @ #

post_video_anaylsis_dir = '../post_analysis/video_analysis'

for user in sorted(os.listdir(post_video_anaylsis_dir)):
    # Skip if not .csv
    if user.split('.', 1)[-1] != 'csv':
        continue

    scenario = user.split('_s', 1)
    user_data_path = os.path.join(post_video_anaylsis_dir, user)
    df = pd.read_csv(user_data_path, sep=';')

    # ! Do something with files that have more than 8 chunks (0-7)
    num_of_chunks = df['CHUNK'].tail(1).values[0]
    if num_of_chunks != 7:
        print(user, str(num_of_chunks))

    # Separate dataframe by chunks
    chunks = df.groupby('CHUNK')

    for chunk_num, chunk in chunks:
        # Add user to df
        data['USER'].append(scenario[0])

        # Determine if user has HUD
        if scenario[-1] == '1.csv':
            data['HUD'].append(True)
        elif scenario[-1] == '3.csv':
            data['HUD'].append(False)

        # ! FIX later
        data['ADL_FAIL'].append(num_of_chunks != 7)

        driving_mode = chunk.head(1)['DRIVING_MODE'].values # First row of a chunk
        if driving_mode == 'MANUAL':
            data['REQUEST_TYPE'].append('AUTO_DRIVE')
            fill_empty_tor() # ADL request is present, fill all TOR columns with empty
            check_adl_switch(chunk) # Check if user switched to AUTOMATIC in under 5s
            adl_looking(chunk) # Calculate % of looked at objects after ADL
        elif driving_mode == 'AUTOMATIC':
            data['REQUEST_TYPE'].append('TAKE_OVER')
            fill_empty_adl() # TOR is present, fill all ADL colimns with empty
            tor_looking(chunk) # Calculate % of looked at objects before TO
            tor_reaction(chunk) # Determine reaction time and type
            after_to(chunk) # Situational awareness after TO

grading_data_path = '../post_analysis/grading'
if not os.path.exists(grading_data_path):
    os.mkdir(grading_data_path)

# Iterate through the dictionary and get the lengths of each list
for key, value in data.items():
    print("Length of list in key '{}': {}".format(key, len(value)))

res = pd.DataFrame.from_dict(data)
res.to_csv(f'{grading_data_path}/grading_data.csv', sep=';')