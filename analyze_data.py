import pandas as pd
from scipy.stats import zscore, pearsonr, ttest_rel
from itertools import combinations

# --- Configuration ---
# This URL is constructed to directly export YOUR SPECIFIC Google Sheet as a CSV file.
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1DmUF3NcYdlHPqYfa4kIMSLx5XTkdNDA_Qsz_fsGOawI/export?format=csv"

INTENSITY_PARAMS = ['brightness', 'contrast', 'saturation']
SPECIFICITY_PARAMS = ['blurriness', 'detailedness', 'precision']
# --- End Configuration ---


def analyze_vim_data(url):
    """
    Reads, cleans, and analyzes the VIM and VVIQ data from a Google Sheet.
    """
    print("--- Starting Full Data Analysis ---")

    # 1. Load Data from Google Sheet
    try:
        # Using engine='python' is crucial for handling complex, user-generated text
        # that can cause parsing errors like the ones you've seen.
        df = pd.read_csv(url, engine='python')
        print(f"Successfully loaded data from Google Sheet. Found {len(df)} rows.")
    except Exception as e:
        print(f"ERROR: Could not load data from the Google Sheet URL.")
        print(f"Please ensure the link is correct and that the sheet is public or has link sharing enabled.")
        print(f"Error details: {e}")
        return

    # --- Section A: VIM Analysis ---

    # 2. Preprocessing and Cleaning VIM Data
    # Ensure 'trial_id' is treated as a string to prevent errors with .str accessor
    df['trial_id'] = df['trial_id'].astype(str)
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

    # --- Generate Subject-Level Raw Score Table ---
    print("\n--- Individual Participant Raw Score Summary ---")
    print("Table 0: Mean Raw Scores by Participant (Scale 1-21):")
    print("(Note: Blurriness is reverse-scored)\n")

    # Use pivot_table to get sessionID as rows, parameters as columns, and mean raw score as values
    subject_raw_summary = df_clean.pivot_table(
        index='sessionID',
        columns='parameter',
        values='selected_level',
        aggfunc='mean'
    )

    # Define the desired order of columns to match the request
    param_order = INTENSITY_PARAMS + SPECIFICITY_PARAMS
    # Ensure all expected columns are present, fill missing with NaN if necessary
    for col in param_order:
        if col not in subject_raw_summary:
            subject_raw_summary[col] = pd.NA
    subject_raw_summary = subject_raw_summary[param_order] # Reorder columns

    # Print the resulting table
    print(subject_raw_summary.to_string(float_format="%.2f"))
    print("\n" + "="*50 + "\n") # Add a separator for clarity before the main analysis begins


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
    raw_summary = raw_summary[param_order]
    print("--- VIM Task Results ---")
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

    # --- Section B: VIM Pairwise Comparisons (Corrected for Proper N) ---
    print("\n\n--- VIM Pairwise Comparisons (Paired T-Tests) ---")
    print("Comparing conditions for each measure using within-participant z-scores.")
    print("Note: N represents the number of participants included in each specific comparison.\n")

    # --- Step 1: Aggregate the data FIRST ---
    # Calculate the mean score for each participant for each measure and condition.
    participant_means = final_scores_z.groupby(['sessionID', 'condition', 'measure'])['score'].mean().reset_index()

    # Get unique measures to loop through
    all_measures = ordered_columns_z

    # --- Step 2: Loop through each measure and perform the t-tests ---
    for measure in all_measures:
        print(f"--- Measure: {measure} ---")

        # Filter the aggregated data for the current measure
        measure_data = participant_means[participant_means['measure'] == measure]

        # Pivot the table so that each row is a participant and each column is a condition's score.
        wide_measure_data = measure_data.pivot(index='sessionID', columns='condition', values='score')

        # Get all unique pairs of conditions to compare
        condition_pairs = combinations(wide_measure_data.columns, 2)

        for cond1, cond2 in condition_pairs:
            # Select the aggregated scores for the two conditions
            cond1_scores = wide_measure_data[cond1]
            cond2_scores = wide_measure_data[cond2]

            # The t-test function with nan_policy='omit' handles participants who might be missing data
            t_stat, p_value = ttest_rel(cond1_scores, cond2_scores, nan_policy='omit')

            # To get the correct 'n', we count how many participants have a valid score in BOTH columns.
            valid_n = len(wide_measure_data[[cond1, cond2]].dropna())

            # Check if we have enough data to run a meaningful test
            if valid_n < 2:
                print(f"  {cond1:<18} vs. {cond2:<18}: Not enough complete participant data to test.")
                continue

            # A final check in case the remaining valid pairs had zero variance
            if pd.isna(t_stat):
                print(f"  {cond1:<18} vs. {cond2:<18}: Result is NaN, likely because the variance of differences is zero.")
            else:
                print(f"  {cond1:<18} vs. {cond2:<18}: t = {t_stat:+.3f}, p = {p_value:.3f}  (n={valid_n})")
        print() 


    # --- Section C: VVIQ Analysis & Correlation ---
    
    # Load and Score VVIQ Data
    df['trial_id'] = df['trial_id'].astype(str)
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

        # Display VVIQ Descriptive Statistics
        vviq_mean = vviq_scores['vviq_total_score'].mean()
        vviq_std = vviq_scores['vviq_total_score'].std()
        vviq_min = vviq_scores['vviq_total_score'].min()
        vviq_max = vviq_scores['vviq_total_score'].max()

        print("\nTable 3: VVIQ-2 Total Score Summary:")
        print(f"  - Mean: {vviq_mean:.2f}")
        print(f"  - Std. Dev.: {vviq_std:.2f}")
        print(f"  - Range: {vviq_min:.0f} - {vviq_max:.0f}\n")
        
        # Correlate VVIQ with VIM scores
        
        # We can reuse the 'participant_means' DataFrame from the t-test section
        participant_means_by_condition = participant_means.pivot_table(
            index='sessionID', 
            columns=['condition', 'measure'], 
            values='score'
        ).reset_index()
        # Flatten the multi-level column headers
        participant_means_by_condition.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] else col[0] for col in participant_means_by_condition.columns.values]


        # Merge the VVIQ scores with the mean VIM scores
        merged_data = pd.merge(participant_means_by_condition, vviq_scores, on='sessionID')

        print("\nTable 4: Pearson Correlations with Total VVIQ Score (by Condition):\n")
        
        # Loop through each condition to run and print the correlations separately
        conditions_to_analyze = ['perceptual_recall', 'episodic_recall', 'scene_imagination']
        for condition in conditions_to_analyze:
            print(f"--- Condition: {condition} ---")
            condition_data = merged_data
            
            # Check if we have enough data to run a correlation (at least 2 data points)
            if len(condition_data) < 2:
                print("  Not enough data to calculate correlation for this condition.")
                continue

            for measure in ordered_columns_z:
                # Construct the column name based on how pandas pivots
                measure_col_name = f'{condition}_{measure}'
                if measure_col_name in condition_data.columns and not condition_data[measure_col_name].isnull().all():
                    # Calculate correlation between the specific measure in this condition and the VVIQ score
                    r, p_value = pearsonr(condition_data[measure_col_name], condition_data['vviq_total_score'])
                    print(f"  {measure:>15}: r = {r:+.3f}, p = {p_value:.3f}")
            print()

    else:
        print("\n--- VVIQ Analysis ---")
        print("No VVIQ data found in the file.")


# --- Run the analysis ---
if __name__ == "__main__":
    analyze_vim_data(GOOGLE_SHEET_URL)