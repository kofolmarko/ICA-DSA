import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import math

# @ GLOBAL @ #
# Import grading results
grading_path = '../post_analysis/grading/grading_data.csv'
df = pd.read_csv(grading_path, sep=';')

# Define all of the labels (seen objects) and colors for the pie charts 
labels = ['Road', 'Distraction', 'Mirrors', 'Dashboard', 'HUD', 'Other/Undefined']
colors = ['#FFB299', '#7AD7FF', '#A3FFA3', '#FFD499', '#C9C9F7', '#FFEC80']

# @ FUNCTIONS @ #
def evaluate_so(adl_df, tor_df, hud):
    adl_scores = []
    tor_scores = []

    for idx, row in adl_df.iterrows():
        row_score = 1.5
        if row['ADL_UNDER_THRESHOLD'] != 'TRUE':
            row_score -= 0.5
        row_score += float(row['ADL_ROAD'])
        row_score -= float(row['ADL_DISTRACTION'])
        row_score = row_score / 3
        adl_scores.append(round(row_score, 3))

    for idx, row in tor_df.iterrows():
        row_score = 6
        row_score -= float(row['TOR_RT']) / 15
        row_score += float(row['TOR_MIRRORS'])
        row_score += float(row['TOR_ROAD'])
        row_score += float(row['TOR_DASHBOARD']) * 0.5
        row_score -= float(row['TOR_OTHER_OR_UNDEFINED']) * 0.9
        row_score -= float(row['TOR_SPEEDING'])
        row_score -= float(row['TOR_ACC'])
        row_score -= float(row['TOR_DCC'])
        row_score -= float(row['TOR_ACC_Y'])
        if hud:
            row_score += float(row['TOR_HUD'])
            row_score = row_score / 8.5
        else:
            row_score = row_score / 8
        tor_scores.append(round(row_score, 3))

    return adl_scores, tor_scores

def single_user_charts(user_df: pd.DataFrame, adl_scores, tor_scores, display):
    # Split the drive by sections
    user_adl = user_df[user_df['REQUEST_TYPE'] == 'AUTO_DRIVE']
    user_tor = user_df[user_df['REQUEST_TYPE'] == 'TAKE_OVER']

    # Create subplot layout
    fig, axs = plt.subplots(3, len(adl_scores))

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
        
        adl_title_text = f'ADR {i + 1}'
        if user_adl['ADL_UNDER_THRESHOLD'].values[i] != 'True':
            adl_title_text += '\nTime to transition\nexceeded threshold'
        axs[0, i].set_title(adl_title_text)
        axs[0, i].annotate(f"SA score: {round(adl_scores[i] * 100, 1)}%", xy=(0.5, -0.05), xycoords='axes fraction', ha='center', va='bottom')

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
        
        tor_response_text = user_tor['TOR_RESPONSE'].values[i]
        tor_title_text = f'TOR {i + 1}'
        axs[1, i].set_title(tor_title_text)
        axs[1, i].annotate(f"SA score: {round(tor_scores[i] * 100, 1)}%", xy=(0.5, -0.05), xycoords='axes fraction', ha='center', va='bottom')


        # Set labels for segments with percentage > 10%
        for text in autolabels1:
            if float(text.get_text().strip('%')) <= 10:
                text.set_text('')

    # Create a single legend for all pie charts
    legend_title = 'Seen objects'
    if hud:
        legend_title += ' (with HUD)'
    else: 
        legend_title += ' (no HUD)'
    axs[0, 0].legend(wedges1, labels, loc="best", title=legend_title, bbox_to_anchor=(0, 1.2))

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
    axs[2,0].set_title('Percentage of seen objects before AD')
    axs[2,0].set_xlim([0, 1])
    axs[2,0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[2,0].xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f'{x*100:.0f}%'))
    axs[2,0].invert_yaxis()

    axs[2,2].barh(labels, values_tor, height=1, color=colors)
    axs[2,2].set_title('Percentage of seen objects after TOR (5 s)')
    axs[2,2].set_xlim([0, 1])
    axs[2,2].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs[2,2].xaxis.set_major_formatter(FuncFormatter(lambda x, loc: f'{x*100:.0f}%'))
    axs[2,2].invert_yaxis()

    axs[2,1].remove()
    #axs[2,3].remove()
    adl_so = round(sum(adl_scores) / len(adl_scores), 3)
    tor_so = round(sum(tor_scores) / len(tor_scores), 3)
    overall_so = round((adl_so + tor_so) / 2, 3)
    so_text = f'General ADR SA score: {round(adl_so * 100, 1)}%\n\nGeneral TOR SA score: {round(tor_so * 100, 1)}%\n\nGeneral driving SA score: {round(overall_so * 100, 1)}%'
    axs[2, 3].text(0, 0.5, so_text, color='black', fontsize=12, ha='left', va='center')
    axs[2, 3].axis('off')


    # # Create a new figure for other graphs

    # adl_under_threshold = np.count_nonzero(user_adl['ADL_UNDER_THRESHOLD'].values == 'True')
    # tor_rt = user_tor['TOR_RT'].values
    # tor_response = user_tor['TOR_RESPONSE'].values
    # tor_speeding = user_tor['TOR_SPEEDING'].values
    # tor_acc = user_tor['TOR_ACC'].values
    # tor_dcc = user_tor['TOR_DCC'].values
    # tor_accy = user_tor['TOR_ACC_Y'].values

    # # Create an empty figure
    # fig = plt.figure(facecolor='white')

    # # Set the text properties
    # text = f'ADL under threshold {adl_under_threshold}/4 times.\n'
    # text += f'TOR mean reaction times: {tor_rt}\n'
    # text += f'TOR response types: {tor_response}\n'
    # text += f'After TO speeding: {tor_speeding}\n'
    # text += f'After TO acceleration over threshold: {tor_acc}\n'
    # text += f'After TO decceleration over threshold: {tor_dcc}\n'
    # text += f'After TO lateral acceleration over threshold: {tor_accy}\n'

    # # Add the text to the figure
    # plt.text(0.5, 0.5, text, color='black', fontsize=12, ha='center', va='center')

    # # Remove the axis
    # plt.axis('off')

    # Display the figure
    if display:
        plt.show()

# @ MAIN @ #
# Create a mask for extracting one drive from grades
user_id = 160
user = f'user_{user_id}'
hud = False

mask = (df['USER'] == user) & (df['HUD'] == hud)
user_df = df[mask]
user_adl_data = user_df[(user_df['REQUEST_TYPE'] == 'AUTO_DRIVE')]
user_tor_data = user_df[(user_df['REQUEST_TYPE'] == 'TAKE_OVER')]
adl_scores, tor_scores = evaluate_so(user_adl_data, user_tor_data, hud)
single_user_charts(user_df, adl_scores, tor_scores, True)

overall_time_looked_hud = pd.to_numeric(df[(df['REQUEST_TYPE'] == 'TAKE_OVER')]['TOR_HUD']).mean()
print(str(round(overall_time_looked_hud * 100, 1)) + '%')

with open('helpers/user_ids.txt', 'r') as f:
    lines = f.readlines()
    user_ids = [line.strip() for line in lines]
    overall_adl_scores = []
    overall_tor_scores = []
    overall_adl_scores_hud = []
    overall_tor_scores_hud = []
    for id in user_ids:
        user = f'user_{id}'
        mask = (df['USER'] == user)
        user_df = df[mask]
        # Skip user 164 because of bad data
        if (id == 164):
            continue
        hud = True
        mask = (df['USER'] == user) & (df['HUD'] == hud)
        user_df = df[mask]
        user_adl_data = user_df[(user_df['REQUEST_TYPE'] == 'AUTO_DRIVE')]
        user_tor_data = user_df[(user_df['REQUEST_TYPE'] == 'TAKE_OVER')]
        adl_scores, tor_scores = evaluate_so(user_adl_data, user_tor_data, hud)
        overall_adl_scores_hud.append(round(sum(adl_scores) / len(adl_scores), 3)) 
        overall_tor_scores_hud.append(round(sum(tor_scores) / len(tor_scores), 3))
        hud = False
        user = f'user_{id}'
        mask = (df['USER'] == user) & (df['HUD'] == hud)
        user_df = df[mask]
        user_adl_data = user_df[(user_df['REQUEST_TYPE'] == 'AUTO_DRIVE')]
        user_tor_data = user_df[(user_df['REQUEST_TYPE'] == 'TAKE_OVER')]
        adl_scores, tor_scores = evaluate_so(user_adl_data, user_tor_data, hud)
        overall_adl_scores.append(round(sum(adl_scores) / len(adl_scores), 3)) 
        overall_tor_scores.append(round(sum(tor_scores) / len(tor_scores), 3))

   # Estimate mean and standard deviation for overall_adl_scores and overall_tor_scores
    mu1 = np.mean(overall_adl_scores)
    sigma1 = np.std(overall_adl_scores)

    mu2 = np.mean(overall_adl_scores_hud)
    sigma2 = np.std(overall_adl_scores_hud)

    # Estimate mean and standard deviation for overall_adl_scores_hud and overall_tor_scores_hud
    mu3 = np.mean(overall_tor_scores)
    sigma3 = np.std(overall_tor_scores)

    mu4 = np.mean(overall_tor_scores_hud)
    sigma4 = np.std(overall_tor_scores_hud)

    # Generate data for the Gaussian distributions
    x = np.linspace(min(min(mu1 - 3 * sigma1, mu2 - 3 * sigma2), min(mu3 - 3 * sigma3, mu4 - 3 * sigma4)),
                    max(max(mu1 + 3 * sigma1, mu2 + 3 * sigma2), max(mu3 + 3 * sigma3, mu4 + 3 * sigma4)), 100)

    y1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    y2 = (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    y3 = (1 / (sigma3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu3) / sigma3) ** 2)
    y4 = (1 / (sigma4 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu4) / sigma4) ** 2)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot overall_adl_scores and overall_tor_scores on the first subplot
    axs[0].plot(x, y1, label='No HUD')
    axs[0].plot(x, y2, label='HUD')
    axs[0].set_ylabel('Probability Density')
    axs[0].set_title('Gaussian Distribution for ADR SA score')
    axs[0].legend()

    # Plot overall_adl_scores_hud and overall_tor_scores_hud on the second subplot
    axs[1].plot(x, y3, label='No HUD')
    axs[1].plot(x, y4, label='HUD')
    axs[1].set_xlabel('SA score')
    axs[1].set_ylabel('Probability Density')
    axs[1].set_title('Gaussian Distribution for TOR SA score')
    axs[1].legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.show()