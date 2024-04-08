import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_excel('./result/robustness_test_1_kor.xlsx')
# df = pd.read_excel('./result/robustness_test_1_eng.xlsx')
# df = pd.read_excel('./result/robustness_test_1_2.xlsx')
df['answer'] = df['answer'].replace(' ', np.nan)
answer_list_1 = ['7']

aggregated_data = df.groupby(['model', 'user_prompt_content_id', 'user_prompt_style_id']).agg(
    total_responses=pd.NamedAgg(column='answer', aggfunc='count'),
    correct_responses=pd.NamedAgg(column='answer', aggfunc=lambda x: x.apply(lambda y: y in answer_list_1).sum())
).reset_index()


pivot_table = pd.pivot_table(aggregated_data, values='correct_responses', index=['model', 'user_prompt_content_id'], columns='user_prompt_style_id', aggfunc='sum')

# Split the data by model and create a bar graph for each model
models = pivot_table.index.get_level_values(0).unique()

def compare_models_content_style():
    # Create a figure and axes for each model
    fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(15, 7 * len(models)), constrained_layout=True)
    y_min, y_max = float('inf'), float('-inf')

    for i, model in enumerate(models):
        # Filter data for the current model
        model_data = pivot_table.xs(model, level='model')
        
        # Create bar graph for the current model
        model_data.plot(kind='bar', ax=axes[i], title=f'Correct Responses for {model}')
        axes[i].set_ylabel('Number of Correct Responses')
        axes[i].set_xlabel('User Prompt Content ID')
        axes[i].tick_params(axis='x', rotation=0)
        # Update y_min and y_max
        current_y_min, current_y_max = axes[i].get_ylim()
        y_min = min(y_min, current_y_min)
        y_max = max(y_max, current_y_max)

    # Set the same y-axis limits for all subplots
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.show()

def compare_models_content():
    pivot_content_id = pd.pivot_table(aggregated_data, values='correct_responses', index=['model', 'user_prompt_content_id'], aggfunc='sum')

    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(20, 5 * len(models)))
    y_min, y_max = float('inf'), float('-inf')

    for i, model in enumerate(models):
        # Filter data for the current model - by content ID
        model_data_content_id = pivot_content_id.xs(model, level='model')
        model_data_content_id.plot(kind='bar', ax=axes[i], title=f'Correct Responses by Content ID for {model}')
        axes[i].set_ylabel('Number of Correct Responses')
        axes[i].set_xlabel('User Prompt Content ID')
        axes[i].tick_params(axis='x', rotation=0)
        # Update y_min and y_max
        current_y_min, current_y_max = axes[i].get_ylim()
        y_min = min(y_min, current_y_min)
        y_max = max(y_max, current_y_max)

    # Set the same y-axis limits for all subplots
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

def compare_models_style():
    pivot_style_id = pd.pivot_table(aggregated_data, values='correct_responses', index=['model', 'user_prompt_style_id'], aggfunc='sum')
    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(20, 5*len(models)))
    y_min, y_max = float('inf'), float('-inf')

    for i, model in enumerate(models):
        # Filter data for the current model - by style ID
        model_data_style_id = pivot_style_id.xs(model, level='model')
        model_data_style_id.plot(kind='bar', ax=axes[i], title=f'Correct Responses by Style ID for {model}')
        axes[i].set_ylabel('Number of Correct Responses')
        axes[i].set_xlabel('User Prompt Style ID')
        axes[i].tick_params(axis='x', rotation=0)
        # Update y_min and y_max
        current_y_min, current_y_max = axes[i].get_ylim()
        y_min = min(y_min, current_y_min)
        y_max = max(y_max, current_y_max)

    # Set the same y-axis limits for all subplots
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def compare_models_content_style_heatmap():
    # Create a figure and axes for each model
    fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(15, 7 * len(models)), constrained_layout=True)

    for i, model in enumerate(models):
        # Filter data for the current model
        model_data = pivot_table.xs(model, level='model')

        # Plotting the heatmap
        sns.heatmap(model_data, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[i])
        axes[i].set_title(f'Correct Answer Heatmap for Model: {model}')
        axes[i].set_xlabel('User Prompt Style ID')
        axes[i].set_ylabel('User Prompt Content ID')

    plt.show()