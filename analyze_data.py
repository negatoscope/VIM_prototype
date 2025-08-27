import pandas as pd
from scipy.stats import zscore, pearsonr

# --- Configuration ---
CSV_PATH = "VIM Experiment Data - Data.csv"
INTENSITY_PARAMS = ['brightness', 'contrast', 'saturation']
SPECIFICITY_PARAMS = ['blurriness', 'detailedness', 'precision']
# --- End Configuration ---


def analyze_vim_data(filepath):
    """
    Reads, cleans, and analyzes the VIM and VVIQ data according to the pre-registered plan.
    """
    print("--- Starting Full Data Analysis ---")

    # 1. Load Data
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Found {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {filepath}")
        return

    # --- Section A: VIM Analysis ---

    # 2. Preprocessing and Cleaning VIM Data
    df_vim = df[df['trial_id'].str.startswith('main_', na=False)].copy()
    
    # Report number of subjects found in the file
    num_subjects = df_vim['sessionID'].nunique()
    print(f"\nFound data for {num_subjects} unique participants.")

    df_vim['selected_level'] = pd.to_numeric(df_vim['selected_level'], errors='coerce')
    df_clean = df_vim.dropna(subset=['selected_level'])
    df_clean = df_clean[df_clean['parameter'] != 'attention_check']
    
    # 3. Reverse-Score Blurriness
    is_blurriness = (df_clean['parameter'] == 'blurriness')
    df_clean.loc[is_blurriness, 'selected_level'] = 22 - df_clean.loc[is_blurriness, 'selected_level']

    # 4. Within-Participant Z-Scoring
    df_clean['z_score'] = df_clean.groupby('sessionID')['selected_level'].transform(lambda x: zscore(x, nan_policy='omit'))

    # 5. Calculate Composite Scores
    df_wide = df_clean.pivot_table(index=['sessionID', 'trial_id', 'condition'],
                                   columns='parameter',
                                   values='z_score').reset_index()
    df_wide['Intensity'] = df_wide[INTENSITY_PARAMS].mean(axis=1)
    df_wide['Specificity'] = df_wide[SPECIFICITY_PARAMS].mean(axis=1)
    df_wide['TotalVividness'] = df_wide[['Intensity', 'Specificity']].mean(axis=1)
    
    # 6. Generate VIM Summary Tables
    # (Table 1: Raw Scores)
    raw_summary = df_clean.groupby(['condition', 'parameter'])['selected_level'].mean().unstack()
    param_order = INTENSITY_PARAMS + SPECIFICITY_PARAMS
    raw_summary = raw_summary[param_order]
    print("\n--- VIM Task Results ---")
    print("Table 1: Mean Raw Scores by Condition (Scale 1-21):")
    print("(Note: Blurriness is reverse-scored, where 21 = sharpest)\n")
    print(raw_summary.to_string(float_format="%.2f"))

    # (Table 2: Z-Scores)
    final_scores_z = pd.melt(df_wide, id_vars=['sessionID', 'condition'], 
                             value_vars=param_order + ['Intensity', 'Specificity', 'TotalVividness'],
                             var_name='measure', value_name='score')
    summary_table_z = final_scores_z.groupby(['condition', 'measure'])['score'].mean().unstack()
    ordered_columns_z = param_order + ['Intensity', 'Specificity', 'TotalVividness']
    summary_table_z = summary_table_z[ordered_columns_z]
    print("\n\nTable 2: Mean Z-Scores by Condition (Standardized):")
    print("(Positive values are above participant's average; Negative are below)\n")
    print(summary_table_z.to_string(float_format="%.3f"))


    # --- Section B: VVIQ Analysis & Correlation ---
    
    # 7. Load and Score VVIQ Data
    df_vviq = df[df['trial_id'] == 'VVIQ'].copy()
    if not df_vviq.empty:
        # The 32 scores are in columns 'vviq_1' through 'vviq_32'. We sum them up.
        vviq_cols = [f'vviq_{i}' for i in range(1, 33)]
        # Make sure the columns exist and are numeric before summing
        for col in vviq_cols:
            if col not in df_vviq.columns:
                df_vviq[col] = pd.NA # Add missing column if it doesn't exist
        df_vviq[vviq_cols] = df_vviq[vviq_cols].apply(pd.to_numeric, errors='coerce')
        
        df_vviq['vviq_total_score'] = df_vviq[vviq_cols].sum(axis=1)
        
        # Keep only the sessionID and the total score for merging
        vviq_scores = df_vviq[['sessionID', 'vviq_total_score']].dropna()
        print(f"\n--- VVIQ Analysis ---")
        print(f"Scored VVIQ data for {len(vviq_scores)} participants.")

        # 8. Display VVIQ Descriptive Statistics
        vviq_mean = vviq_scores['vviq_total_score'].mean()
        vviq_std = vviq_scores['vviq_total_score'].std()
        vviq_min = vviq_scores['vviq_total_score'].min()
        vviq_max = vviq_scores['vviq_total_score'].max()

        print("\nTable 3: VVIQ-2 Total Score Summary:")
        print(f"  - Mean: {vviq_mean:.2f}")
        print(f"  - Std. Dev.: {vviq_std:.2f}")
        print(f"  - Range: {vviq_min:.0f} - {vviq_max:.0f}\n")
        
        # 8. Correlate VVIQ with VIM scores
        
        # First, calculate the mean score for each measure for each participant.
        participant_means_by_condition = final_scores_z.groupby(['sessionID', 'condition', 'measure'])['score'].mean().unstack()
        
        # Merge the VVIQ scores with the mean VIM scores
        merged_data = pd.merge(participant_means_by_condition.reset_index(), vviq_scores, on='sessionID')

        print("\nTable 4: Pearson Correlations with Total VVIQ Score (by Condition):\n")
        
        # Loop through each condition to run and print the correlations separately
        conditions_to_analyze = ['perceptual_recall', 'episodic_recall', 'scene_imagination']
        for condition in conditions_to_analyze:
            print(f"--- Condition: {condition} ---")
            condition_data = merged_data[merged_data['condition'] == condition]
            
            # Check if we have enough data to run a correlation (at least 2 data points)
            if len(condition_data) < 2:
                print("  Not enough data to calculate correlation for this condition.")
                continue

            for measure in ordered_columns_z:
                if measure in condition_data.columns and not condition_data[measure].isnull().all():
                    r, p_value = pearsonr(condition_data[measure], condition_data['vviq_total_score'])
                    print(f"  {measure:>15}: r = {r:+.3f}, p = {p_value:.3f}")
            print() # Add a blank line for readability

    else:
        print("\n--- VVIQ Analysis ---")
        print("No VVIQ data found in the file.")


# --- Run the analysis ---
if __name__ == "__main__":
    analyze_vim_data(CSV_PATH)