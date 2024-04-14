import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to rank data within each model in a pivot table
def rank_within_model(pivot_data, axis=1, ascending=False, method='min'):
    """
    Rank data within each model in a pivot table.

    Args:
        pivot_data (pd.DataFrame): The pivot table to rank.
        axis (int): The axis to perform ranking (0 for rows, 1 for columns).
        ascending (bool): Whether to rank in ascending order.
        method (str): The method of ranking (e.g., 'average', 'min', 'max', etc.).

    Returns:
        pd.DataFrame: Ranked data.
    """
    rank_data = pd.DataFrame(index=pivot_data.index, columns=pivot_data.columns)

    # Determine if DataFrame has MultiIndex and 'model' is one of the levels
    is_multiindex = isinstance(pivot_data.index, pd.MultiIndex)
    model_level = 0 if is_multiindex else None

    # Iterate through each model to rank data within the model
    for model in pivot_data.index.get_level_values(model_level).unique():
        model_data = pivot_data.xs(model, level=model_level, drop_level=False)

        # Perform ranking
        ranked = model_data.rank(axis=axis, ascending=ascending, method=method).astype(int)

        # Properly handle MultiIndex to avoid misalignment
        if is_multiindex:
            rank_data.loc[model_data.index] = ranked.values
        else:
            rank_data.loc[model] = ranked

    return rank_data

# Function to process and aggregate data
def process_and_aggregate_data(df, used_data, answer_key):
    df['used_data'] = used_data
    df['answer'] = df['answer'].replace(' ', np.nan).astype(str)
    df['type'] = df['answer'].apply(lambda x: type(x))
    return df.groupby(['used_data', 'model', 'var_list_index']).agg(
        total_responses=pd.NamedAgg(column='answer', aggfunc='count'),
        correct_responses=pd.NamedAgg(column='answer', aggfunc=lambda x: x.apply(lambda y: y in answer_key).sum())
    ).reset_index()

# Function to compare models by variable list index
def compare_models_var(pivot_table):
    rank_table = rank_within_model(pivot_table)
    n_models = pivot_table.index.get_level_values('model').nunique()
    y_min, y_max = float('inf'), float('-inf')

    # Use one row and n_models columns for side-by-side plotting
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(20, 10))

    # Make sure axes is an array, even if there's only one model
    if n_models == 1:
        axes = [axes]

    # Define a list of colors for the bars, ensuring enough colors for all bars
    color_cycle = plt.cm.tab20.colors
    colors = color_cycle * (len(pivot_table.columns) // len(color_cycle) + 1)

    for i, (model, model_data) in enumerate(pivot_table.groupby(level='model')):
        ax = axes[i]

        # Extract bar positions and heights
        bar_positions = np.arange(len(model_data.columns))
        bar_heights = model_data.values.flatten()

        # Plot the bars
        bars = ax.bar(bar_positions, bar_heights, color=colors[:len(model_data.columns)])

        # Retrieve the rank data for the current model
        model_rank_data = rank_table.xs(model, level='model').values.flatten()

        # Add text for the rank above each bar
        for bar, rank in zip(bars, model_rank_data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(rank),
                    ha='center', va='bottom', color='black', fontsize=8)

        # Set the title and labels
        ax.set_title(f'Correct Responses by Var List Index for {model}')
        ax.set_ylabel('Number of Correct Responses')
        ax.set_xlabel('Var List Index')

        # Set the x-axis tick labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(model_data.columns)
        current_y_min, current_y_max = axes[i].get_ylim()
        y_min = min(y_min, current_y_min)
        y_max = max(y_max, current_y_max)
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


# Load data
df_2_kor = pd.read_excel('./result/robustness_test_2_kor.xlsx')
df_2_kor_adv = pd.read_excel('./result/robustness_test_2_kor_adv.xlsx')
df_3_kor = pd.read_excel('./result/robustness_test_3_kor.xlsx')

# Replace empty strings with NaN
df_2_kor['answer'] = df_2_kor['answer'].replace(' ', np.nan)
df_2_kor_adv['answer'] = df_2_kor_adv['answer'].replace(' ', np.nan)
df_3_kor['answer'] = df_3_kor['answer'].replace(' ', np.nan)

# Define answer list
answer_list = ['7']
# answer_list = ['파란색', '파랑', '파랑색', '청색', '푸른색']

aggregated_data_2_kor = process_and_aggregate_data(df_2_kor, 'robustness_test_2_kor', answer_list)
aggregated_data_2_kor_adv = process_and_aggregate_data(df_2_kor_adv, 'robustness_test_2_kor_adv', answer_list)
aggregated_data_3_kor = process_and_aggregate_data(df_3_kor, 'robustness_test_3_kor', answer_list)

pivot_table_2_kor = pd.pivot_table(aggregated_data_2_kor, values='correct_responses', index=['used_data', 'model'], columns=['var_list_index'], aggfunc='sum')
pivot_table_2_kor_adv = pd.pivot_table(aggregated_data_2_kor_adv, values='correct_responses', index=['used_data', 'model'], columns=['var_list_index'], aggfunc='sum')
pivot_table_3_kor = pd.pivot_table(aggregated_data_3_kor, values='correct_responses', index=['used_data', 'model'], columns=['var_list_index'], aggfunc='sum')

# models = pivot_table_concat_1.index.get_level_values(1).unique()
models = pivot_table_2_kor_adv.index.get_level_values(1).unique()

# print(pivot_table_2_kor_adv)
compare_models_var(pivot_table_2_kor_adv)
