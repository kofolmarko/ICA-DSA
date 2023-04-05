import os
import sys
import numpy as np
import cv2
import torch
import pandas as pd

# @ cv2 labeling video @ #
# Define functions that will help us later
def center_point(coords):
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (int(center_x), int(center_y))

def rectangles_intersect(rect1_tl, rect1_br, rect2_tl, rect2_br):
    x1_tl, y1_tl = rect1_tl
    x1_br, y1_br = rect1_br
    x2_tl, y2_tl = rect2_tl
    x2_br, y2_br = rect2_br
    w1 = x1_br - x1_tl
    h1 = y1_br - y1_tl
    w2 = x2_br - x2_tl
    h2 = y2_br - y2_tl
    if x1_tl > x2_tl + w2 or x2_tl > x1_tl + w1:
        return False
    if y1_tl > y2_tl + h2 or y2_tl > y1_tl + h1:
        return False
    return True

# Read the labels file and assign a random color to each label
label_names = []
label_colors = []

with open('helpers/labels.txt', 'r') as file:
    lines = file.read().splitlines()
    for line in lines:
        label_names.append(line.split(' ')[0])
        color = np.random.randint(0, 255, size=(3,)).tolist()
        color = [int(c) for c in color]
        label_colors.append(color)

# Load in the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolo/best.pt')

# Enable/disable checking for multiple view_markers by pressing 'C' key
check_for_multiple_view_markers = False

# Path to videos directory
parent_dir = '../simulator_data'
videos_dir = os.path.join(parent_dir, 'videos')

# Create the results directory
video_analysis_data_dir = '../post_analysis/video_analysis'
if not os.path.exists(video_analysis_data_dir):
    os.mkdir(video_analysis_data_dir)

# @ MAIN LOOP @ #
# Iterate over each directory
for video in os.listdir(videos_dir):
    # Skip file if it's not .mp4
    video_split = video.split('.', 1)
    suffix = video_split[-1]
    if suffix != 'mp4':
        continue

    # Construct the filename for the mp4 file, chunks directory and results directory
    user = video_split[0]
    mp4_file = os.path.join(videos_dir, video)
    chunks_dir = os.path.join('../post_analysis', 'chunks', f'chunks_{user}')

    # Check if the csv and txt files exist in the current directory
    if os.path.isfile(mp4_file):
        # Load in the video
        cap = cv2.VideoCapture(mp4_file)

        # Dictionary for creating a dataframe for the final .csv file
        data_export = {
            'CHUNK': [],
            'FRAME': [],
            'DRIVING_MODE': [],
            'SEEN_OBJECTS': [], # All of the looked at objects in one frame separated by a coma
            'STEERING_WHEEL_ANGLE': [],
            'ACCELERATION': [],
            'ACCELERATION_Y': [],
            'SPEED': [],
            'SPEED_LIMIT': [],
            'BRAKE_PEDAL': [],
            'INDICATORS': [],
        }

        # Get a list of all chunks inside the chunks directory
        chunks = [c for c in sorted(os.listdir(chunks_dir))]
        for i in range(len(chunks)):
            chunks[i] = os.path.join(chunks_dir, chunks[i])

        for index, chunk in enumerate(chunks):
            df = pd.read_csv(chunk)

            # Takes one frame per loop
            for fidx, f in enumerate(df['FRAME']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                _, frame = cap.read()
                _, view_markers_frame = cap.read()

                # Get frame dimensions
                height = frame.shape[0]
                width = frame.shape[1]

                # Quits if frame does not exist
                if frame is None:
                    break

                # Reset labels in frame
                labels_in_frame = []
                view_marker_index = -1

                # Get results for one frame using the trained model and extract labels/objects and coordinates
                results = model(frame)
                labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

                # Goes through each label/object found in one frame of the video
                for i in range(len(labels)):
                    # Gets confidence of every label
                    confidence = cord_thres[i][4]
                    if confidence < 0.5:
                        continue

                    # Save coordinates to variables and map them to pixels
                    x1 = int(cord_thres[i][0] * frame.shape[1])
                    y1 = int(cord_thres[i][1] * frame.shape[0])
                    x2 = int(cord_thres[i][2] * frame.shape[1])
                    y2 = int(cord_thres[i][3] * frame.shape[0])

                    # Extract needer properties for a label
                    name = label_names[int(labels[i])]
                    color = label_colors[int(labels[i])]
                    center = center_point([(x1, y1), (x2, y2)])

                    # Adding a new label to the array
                    labels_in_frame.append({
                        'index': i,
                        'name': label_names[int(labels[i])],
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                        'center': center,
                        'color': label_colors[int(labels[i])]
                    })

                    # Save view marker on each frame to be able to check for intersections
                    if label_names[int(labels[i])] == 'view_marker':
                        is_correct_marker = False
                        if view_marker_index == -1:
                            cv2.putText(view_markers_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            cv2.rectangle(view_markers_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        elif check_for_multiple_view_markers:
                            # If there are multiple view_markers in one frame, the script will ask you about the correct one
                            cv2.putText(view_markers_frame, 'Is this the view marker? (Y/N)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            cv2.rectangle(view_markers_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.imshow('frame', view_markers_frame)
                            while True:
                                key = cv2.waitKey(0) & 0xFF
                                if key == ord('y'):
                                    is_correct_marker = True
                                    break
                                elif key == ord('n'):
                                    break
                                elif key == ord('c') or key == ord('q'):
                                    check_for_multiple_view_markers = not check_for_multiple_view_markers
                                    break
                        if is_correct_marker or view_marker_index == -1:
                            view_marker_index = i

                # Retrieving the view_marker from labels_in_frame
                view_marker = None
                if view_marker_index != -1:
                    view_marker = labels_in_frame[view_marker_index]

                # Looking at variables
                looking_at_object = ''
                looked_at_objects = ''

                # Mirrors
                mirrors = []
                for label in labels_in_frame:
                    if label['name'] == 'mirror':
                        mirrors.append(label)
                if len(mirrors) > 0:
                    # Sort the mirrors by their x-coordinate
                    mirrors_sorted = sorted(mirrors, key=lambda m: m['center'][0])
                    left_mirror = mirrors_sorted[0]
                    right_mirror = mirrors_sorted[-1]
                    # Find the mirror with an x-coordinate between the left and right mirrors (rear-view mirror)
                    rearview_mirror = None
                    for mirror in mirrors_sorted:
                        if mirror != left_mirror and mirror != right_mirror:
                            if left_mirror['center'][0] < mirror['center'][0] < right_mirror['center'][0]:
                                rearview_mirror = mirror
                                break

                    for label in labels_in_frame:
                        if label == left_mirror:
                            label['name'] = 'left_mirror'
                        if label == right_mirror:
                            label['name'] = 'right_mirror'
                        if label == rearview_mirror:
                            label['name'] = 'rearview_mirror'

                # Going over all labels found in one frame
                for label in labels_in_frame:
                    # Skip duplicate view_markers
                    if label['name'] == 'view_marker' and label != view_marker:
                        continue

                    # Change the name of 'marker' to 'road'
                    if label['name'] == 'marker':
                        label['name'] = 'road'

                    # Checking for intersections with the view_marker in one frame
                    if label['name'] != 'view_marker' and view_marker is not None:
                        intersects = rectangles_intersect(
                            (view_marker['x1'], view_marker['y1']),
                            (view_marker['x2'], view_marker['y2']),
                            (label['x1'], label['y1']),
                            (label['x2'], label['y2'])
                        )
                        if intersects:
                            # Get the name of the object the driver is looking at and add it to the string
                            # of all looked at objects in this frame, separated with a coma
                            looking_at_object = str(label['name'])
                            looked_at_objects += f'{looking_at_object},'

                    # Displaying the labels
                    cv2.rectangle(frame, (label['x1'], label['y1']), (label['x2'], label['y2']), label['color'], 2)
                    cv2.putText(frame, label['name'], (label['x1'], label['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label['color'], 2)
                    cv2.circle(frame, label['center'], 3, label['color'], 1)

                # Display current frame and other dataframe info
                cv2.rectangle(frame, (0, height), (300, height - 110), (0, 0, 0), -1)
                cv2.putText(frame, 'Looking at: ' + looking_at_object, (10, height - 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(frame, 'Frame: ' + str(int(f)), (10, height - 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                
                is_auto_drive = df.query(f'FRAME=={f}')['AUTO_DRIVE'].values[0]
                if is_auto_drive == 4:
                    auto_drive_text_color = (100, 255, 50)
                    auto_drive_text = 'AUTOMATIC'
                else:
                    auto_drive_text = 'MANUAL'
                    auto_drive_text_color = (50, 100, 255)
                cv2.putText(frame, 'Driving mode: ', (10, height - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.putText(frame, auto_drive_text, (130, height - 30), cv2.FONT_HERSHEY_PLAIN, 1, auto_drive_text_color, 1)

                is_lead_time = float(df.query(f'FRAME=={f}')['LEAD_TIME'].values[0]) # type: ignore 
                if is_lead_time > 0:
                    cv2.putText(frame, 'Take over request', (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (50, 100, 255), 1)
                
                is_hud_5019 = float(df.query(f'FRAME=={f}')['HUD_5019'].values[0]) # type: ignore 
                if is_hud_5019 > 0:
                    cv2.putText(frame, 'Automatic drive request', (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (100, 255, 50), 1)

                # Display which user we are watching
                cv2.putText(frame, user, (10, height - 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 100, 255), 1)

                # Show single frame
                cv2.imshow('frame', frame)

                # Save frame data into data export
                data_export['CHUNK'].append(index)
                data_export['FRAME'].append(int(f))
                data_export['DRIVING_MODE'].append(auto_drive_text)
                data_export['SEEN_OBJECTS'].append(looked_at_objects)
                data_export['STEERING_WHEEL_ANGLE'].append(df['STEERING_WHEEL_ANGLE'][fidx])
                data_export['ACCELERATION'].append(df['ACCELERATION'][fidx])
                data_export['ACCELERATION_Y'].append(df['ACCELERATION_Y'][fidx])
                data_export['SPEED'].append(df['SPEED'][fidx])
                data_export['SPEED_LIMIT'].append(df['SPEED_LIMIT'][fidx])
                data_export['BRAKE_PEDAL'].append(df['BRAKE_PEDAL'][fidx])
                data_export['INDICATORS'].append(df['INDICATORS'][fidx])

                # Quit if 'q' is pressed, change to waitKey(0) to freeze each frame
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    check_for_multiple_view_markers = not check_for_multiple_view_markers
                if key == ord('p'):
                    continue
                elif key == ord('q'):
                    break
                elif key == 27:
                    sys.exit(0)

        # Save data export into a .csv file
        df_export = pd.DataFrame(data_export)
        df_export.to_csv(f'{os.path.join(video_analysis_data_dir, user)}.csv', index=False, sep=';')

        # Close the video
        cap.release()
        cv2.destroyAllWindows()