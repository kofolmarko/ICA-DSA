# Inferential continuous assessment of driver's situational awareness (ICA-DSA)
This repository contains a collection of Python scripts used to construct a semi-automated system for performing the ICA-DSA method. All of the data processed with this method was collected by performing a user study with an eyetracking system and a driving simulator. The user study was conducted in the **University of Ljubljana, Faculty of Electrical Engineering**.

# Project architecture
## Scripts
The core of this project lays within the ***scripts*** directory, where all of the Python automation is written. Inside it there is a ***helpers*** directory with a **progress_bar.py** script which displays a progress bar in the terminal when executing other scripts. There are also two .txt files: **labels.txt** with all of the labels used when training the dataset and **user_ids.txt** containing integer ids of the valid users, used for targeted data import.

The scripts included inside the directory are the following:
- **extract_test_data.py**

    Extracts the images used to build a dataset.

- **chunk_splitter.py**

    Splits the data files into relevant chunks.

- **video_analysis.py**

    Goes over the chunks of relevant data and produces results based on a frame-by-frame anaylsis.

- **grading.py**

    Further processing of data to extract needed information.

- **display_results.py**

    Displays the results in different forms such as pie charts or bar graphs.
## Yolo
The ***yolo*** folder contains a machine learning model created using PyTorch. Besides the **best.pt** file there is also a **custom_data.yaml** file, used in performing the machine learning process. Training a custom YOLOv5 model was done on an official Google Colab page provided on the YOLOv5 Github repository.
>YOLOv5 repository https://github.com/ultralytics/yolov5

>Modified official training example https://colab.research.google.com/drive/1dqDmPicbrlGFgMcxae6tOty8i0v6P0GJ?usp=sharing

>Labeling tool repository https://github.com/developer0hye/Yolo_Label
## Other
There are several folders excluded from this repository such as ***simulator_data*** and ***post_analysis***, containing data to be processed and already processed data. 

# Reqirements
## Libraries
- cv2
- matplotlib
- numpy
- os
- pandas
- random
- scipy
- sys
- torch

# Reporoducing the results
This is the order of running scripts, for displaying the results, considering you have already traing a custom YOLOv5 model and have all of the data available inside the ***simulator_data*** directory.
## Run the scripts in the following order:
- **chunk_splitter.py**
- **video_analysis.py**
- **grading.py**
- **display_results.py**
