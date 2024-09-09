import pandas as pd
import numpy as np
from scipy.stats import zscore

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns['% of Total Values'] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns



def outliers_table(df):
    # Initialize empty lists to store results
    outliers_info = []
    
    # Iterate through numerical columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate Z-scores
        z_scores = np.abs(zscore(df[col].dropna()))
        
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Z-score method
        z_score_outliers = (z_scores > 3).sum()
        
        # IQR method
        iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        
        # Append results
        outliers_info.append({
            'Column': col,
            'Z-Score Outliers': z_score_outliers,
            'IQR Outliers': iqr_outliers
        })
    
    # Convert results to DataFrame
    outliers_table_df = pd.DataFrame(outliers_info)
    
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "Outlier information for numerical columns is displayed below.")
    
    # Return the dataframe with outlier information
    return outliers_table_df


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1e+6  # 1 MB = 1e+6 Bytes
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


def convert_ms_to_seconds(ms):
    return ms / 1000



def fix_outlier(df, column, percentile=0.95):
    threshold = df[column].quantile(percentile)
    median = df[column].median()
    df[column] = np.where(df[column] > threshold, median, df[column])
    return df[column]


def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    df = df.copy()  # Avoid modifying the original DataFrame
    z_scores = zscore(df[column_to_process].dropna())
    df['z_score'] = np.nan
    df.loc[df[column_to_process].notna(), 'z_score'] = z_scores

    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(df['z_score']) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column, 'z_score'], errors='ignore')

    return df
