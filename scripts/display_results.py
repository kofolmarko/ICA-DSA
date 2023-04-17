import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# @ GLOBAL @ #
# Import grading results
grading_path = '../post_analysis/grading/grading_data.csv'
df = pd.read_csv(grading_path, sep=';')

# Define all of the labels (seen objects) and colors for the pie charts 
labels = ['Road', 'Distraction', 'Mirrors', 'Dashboard', 'HUD', 'Other/Undefined']
colors = ['#FFB299', '#7AD7FF', '#A3FFA3', '#FFD499', '#C9C9F7', '#FFEC80']

# @ FUNCTIONS @ #
def single_user_charts(user_df: pd.DataFrame):
    # Split the drive by sections
    user_adl = user_df[user_df['REQUEST_TYPE'] == 'AUTO_DRIVE']
    user_tor = user_df[user_df['REQUEST_TYPE'] == 'TAKE_OVER']

    # Create subplot layout
    fig, axs = plt.subplots(3, 4)

    # First set of pie charts (ADL Transition)
    for i in range(len(user_adl)):
        values_adl = [
            user_adl['ADL_ROAD'].values[i],
            user_adl['ADL_DISTRACTION'].values[i],
            0, # We must keep the same structure both sets of pie charts, so that
            0, # we can have a common legend and colors.
            0, #
            user_adl['ADL_OTHER_OR_UNDEFINED'].values[i]
        ]
        wedges1, texts1, autolabels1 = axs[0, i].pie(values_adl, labels=None, startangle=90, autopct='%1.1f%%', colors=colors)
        axs[0, i].set_title(f'ADL Transition {i + 1}')

        # Hide percentage labels under a threshold
        for text in autolabels1:
            if float(text.get_text().strip('%')) <= 10:
                text.set_text('')
                
    # Second set of pie charts (TOR Transition)
    for i in range(len(user_tor)):
        values_tor = [
            user_tor['TOR_ROAD'].values[i],
            0,
            user_tor['TOR_MIRRORS'].values[i],
            user_tor['TOR_DASHBOARD'].values[i],
            user_tor['TOR_HUD'].values[i],
            user_tor['TOR_OTHER_OR_UNDEFINED'].values[i]
        ]
        wedges2, texts2, autolabels1 = axs[1, i].pie(values_tor, labels=None, startangle=90, autopct='%1.1f%%', colors=colors)
        axs[1, i].set_title(f'TOR Transition {i + 1}')

        # Set labels for segments with percentage > 10%
        for text in autolabels1:
            if float(text.get_text().strip('%')) <= 10:
                text.set_text('')

    # Create a single legend for all pie charts
    axs[0, 0].legend(wedges1, labels, loc="best", title="Seen objects", bbox_to_anchor=(0, 1.2))

    # Mean values
    values_adl = [
        pd.to_numeric(user_adl['ADL_ROAD'], errors='coerce').mean(),
        pd.to_numeric(user_adl['ADL_DISTRACTION'], errors='coerce').mean(),
        0,
        0,
        0,
        pd.to_numeric(user_adl['ADL_OTHER_OR_UNDEFINED'], errors='coerce').mean()
    ]
    values_tor = [
        pd.to_numeric(user_tor['TOR_ROAD'], errors='coerce').mean(),
        0,
        pd.to_numeric(user_tor['TOR_MIRRORS'], errors='coerce').mean(),
        pd.to_numeric(user_tor['TOR_DASHBOARD'], errors='coerce').mean(),
        pd.to_numeric(user_tor['TOR_HUD'], errors='coerce').mean(),
        pd.to_numeric(user_tor['TOR_OTHER_OR_UNDEFINED'], errors='coerce').mean()
    ]

    # Draw horizontal bar graphs, show in percentage
    axs[2,0].barh(labels, values_adl, height=1, color=colors)
    axs[2,0].set_title('Mean percentage of seen objects before ADL')
    axs[2,0].set_xlim([0, 1])
    axs[2,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[2,0].xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f'{x*100:.0f}%'))
    axs[2,0].invert_yaxis()

    axs[2,2].barh(labels, values_tor, height=1, color=colors)
    axs[2,2].set_title('Mean percentage of seen objects after TOR(5s)')
    axs[2,2].set_xlim([0, 1])
    axs[2,2].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[2,2].xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f'{x*100:.0f}%'))
    axs[2,2].invert_yaxis()

    axs[2,1].remove()
    axs[2,3].remove()

    # Create a new figure for other graphs

    adl_under_threshold = np.count_nonzero(user_adl['ADL_UNDER_THRESHOLD'].values == 'True')
    tor_rt = user_tor['TOR_RT'].values
    tor_response = user_tor['TOR_RESPONSE'].values
    tor_speeding = user_tor['TOR_SPEEDING'].values
    tor_acc = user_tor['TOR_ACC'].values
    tor_dcc = user_tor['TOR_DCC'].values
    tor_accy = user_tor['TOR_ACC_Y'].values

    # Create an empty figure
    fig = plt.figure(facecolor='white')

    # Set the text properties
    text = f'ADL under threshold {adl_under_threshold}/4 times.\n'
    text += f'TOR mean reaction times: {tor_rt}\n'
    text += f'TOR response types: {tor_response}\n'
    text += f'After TO speeding: {tor_speeding}\n'
    text += f'After TO acceleration over threshold: {tor_acc}\n'
    text += f'After TO decceleration over threshold: {tor_dcc}\n'
    text += f'After TO lateral acceleration over threshold: {tor_accy}\n'

    # Add the text to the figure
    plt.text(0.5, 0.5, text, color='black', fontsize=12, ha='center', va='center')

    # Remove the axis
    plt.axis('off')

    # Display the figure
    plt.show()

# @ MAIN @ #
# Create a mask for extracting one drive from grades
user_id = 151
user = f'user_{user_id}'
hud = False

mask = (df['USER'] == user) & (df['HUD'] == hud)
user_df = df[mask]
single_user_charts(user_df)