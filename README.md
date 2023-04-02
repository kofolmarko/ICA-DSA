# DrivingSimulatorAnalysis
An in-depth analysis of a driving simulator with real-time eyetracker and vehicle data.

# About this repository
In this repository there are scripts for handling driving-simulator-generated data and videos. The scripts can be found in the **scripts** folder. For the execution of these scripts I'm using private data stored inside the **simulator_data** data folder. The data is stored within .csv and .txt files and the videos are in .mp4 format.

# Reqirements
pip3 install stuff

# Reporoducing the results
To achieve the final output of extracted data that is needed for my analysis, I execute the following steps:
- Run **chunk_splitter.py** to create a new folder of .csv files that only include the parts of data we need. Those are the parts where the driver switches from automatic to manual driving mode or vice versa. Inside this script the data is also interpolated to match the framerate of simulator videos for further analysis.