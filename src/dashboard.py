import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))
from utils import missing_values_table, fix_outlier, convert_bytes_to_megabytes, convert_ms_to_seconds
from load_data import load_data_from_postgres

# Query to load data from PostgreSQL
query = "SELECT * FROM xdr_data"
df_postgres = load_data_from_postgres(query)



user_overview_columns = [
    'IMSI', 'MSISDN/Number', 'IMEI', 'Handset Manufacturer', 'Handset Type',
    'Dur. (ms)', 'Start', 'End', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)',
    'Total DL (Bytes)', 'Total UL (Bytes)',
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)',
    'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
]
df_user_overview = df_postgres[user_overview_columns].copy()

# Title for the app
st.title("User Overview Analysis Dashboard")

# Navigation Sidebar
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Select Analysis", ['User Overview Analysis'])

# Main Page: User Overview Analysis
if analysis_type == 'User Overview Analysis':
    st.title("User Overview Analysis")

    ### Segment 1: Columns Extracted ###
    st.subheader("Columns Extracted for Analysis")

    columns = ['IMSI', 'MSISDN/Number', 'Dur. (ms)', 'Activity Duration DL (s)', 'Activity Duration UL (s)', 
               'Handset Type', 'Handset Manufacturer'] 
    descriptions = {
        'IMSI': 'Unique identifier for the user, crucial for tracking activity.',
        'MSISDN/Number': 'Phone number used for identifying individual users in the network.',
        'Dur. (ms)': 'Duration of the session in seconds, important for understanding engagement.',
        'Activity Duration DL (s)': 'Duration of data download sessions in seconds.',
        'Activity Duration UL (s)': 'Duration of data upload sessions in seconds.',
        'Handset Type': 'Type of handset used by the user.',
        'Handset Manufacturer': 'The manufacturer of the handset device.'
    }

    for col in columns:
        with st.expander(f"Column: {col}"):
            st.write(descriptions.get(col))

    ### Segment 2: Data Preprocessing ###
    st.subheader("Data Preprocessing")

    # Data Cleaning
    st.markdown("### Null Values Handling")
    st.write("Before fixing null values in percentage:")
    st.dataframe(missing_values_table(df_user_overview))

    # Fixing null values: Example - Filling missing numeric values with median
    st.write("Dropping missing MSISDN/Number:")
    df_user_overview.dropna(subset=['MSISDN/Number'], inplace=True)
    st.dataframe(missing_values_table(df_user_overview))

    st.write("Replacing the missing Avg RTT DL(ms) and Avg RTT UL(ms) by their mean values:")
    # Calculate mean values
    mean_rtt_dl = df_user_overview['Avg RTT DL (ms)'].mean()
    mean_rtt_ul = df_user_overview['Avg RTT UL (ms)'].mean()

    # Fill missing values with mean
    df_user_overview['Avg RTT DL (ms)'].fillna(mean_rtt_dl, inplace=True)
    df_user_overview['Avg RTT UL (ms)'].fillna(mean_rtt_ul, inplace=True)

    df_user_overview['Dur. (ms)'] = df_user_overview['Dur. (ms)'].fillna(df_user_overview['Dur. (ms)'].median())
    df_user_overview['IMSI'] = df_user_overview['IMSI'].fillna('Unknown')  # Example: Filling categorical missing values with 'Unknown'
    st.write(df_user_overview.isnull().sum())

    # Outlier Fixing

    # Outlier detection and fixing function
    def calculate_bounds(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    ### Data Preprocessing: Outlier Fixing ###

    st.markdown("## Outlier Fixing")
    st.write("We are detecting and fixing outliers by capping them based on the 5th and 95th percentiles for relevant columns. Here's how we handle them:")

    # List of relevant columns to apply the outlier fixing
    user_overview_columns_to_apply_quartiles = [
        'Dur. (ms)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)',
        'Total DL (Bytes)', 'Total UL (Bytes)',
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)',
        'Email DL (Bytes)', 'Email UL (Bytes)',
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)',
        'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
    ]

    # Apply the external fix_outlier function to each specified column
    df_no_outliers = df_user_overview.copy()
    progress_bar = st.progress(0)  # Add progress bar for user experience

    for idx, column in enumerate(user_overview_columns_to_apply_quartiles):
        if column in df_no_outliers.columns:
            # Calculate bounds
            lower_bound, upper_bound = calculate_bounds(df_no_outliers, column)

            # Display bounds using metrics
            with st.expander(f"Outlier detection for {column}"):
                col1, col2 = st.columns(2)
                col1.metric("Lower bound (IQR)", f"{lower_bound:.2f}")
                col2.metric("Upper bound (IQR)", f"{upper_bound:.2f}")

            # Detect and display the number of outliers
            outliers = df_no_outliers[(df_no_outliers[column] < lower_bound) | (df_no_outliers[column] > upper_bound)]
            st.write(f"Outliers detected in {column}: **{len(outliers)}**")

            # Apply the external outlier fixing function
            df_no_outliers = fix_outlier(df_no_outliers, column)  
            progress_bar.progress((idx + 1) / len(user_overview_columns_to_apply_quartiles))  # Update progress

    # Display summary after processing
    st.success("Outlier fixing completed!")
    st.write("After fixing outliers, the dataset has the following shape:")
    st.write(f"**Rows: {df_no_outliers.shape[0]}, Columns: {df_no_outliers.shape[1]}**")

    # Optional: Display data or statistics
    if st.checkbox("Show cleaned dataset preview"):
        st.write(df_no_outliers.head())

    # Data Formatting

    byte_columns = [
        'Total DL (Bytes)', 'Total UL (Bytes)',
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'YouTube DL (Bytes)', 'YouTube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)',
        'Email DL (Bytes)', 'Email UL (Bytes)',
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]

    millisecond_columns = [
        'Dur. (ms)',
        'Activity Duration DL (ms)',
        'Activity Duration UL (ms)',
        'Avg RTT DL (ms)',
        'Avg RTT UL (ms)'
    ]

    ### Data Formatting and Renaming ###
    st.markdown("### Data Formatting and Renaming Columns")

    # Show sample table before conversion and renaming
    st.subheader("Sample Data Before Conversion and Renaming")
    st.write(df_no_outliers.head(5))

    # Apply conversion for byte columns
    for column in byte_columns:
        if column in df_no_outliers.columns:
            df_no_outliers[column] = df_no_outliers[column].apply(convert_bytes_to_megabytes)

    # Apply conversion for millisecond columns
    for column in millisecond_columns:
        if column in df_no_outliers.columns:
            df_no_outliers[column] = df_no_outliers[column].apply(convert_ms_to_seconds)

    # Rename columns from Bytes to Megabytes and (ms) to (s)
    df_no_outliers.rename(columns=lambda x: x.replace('Bytes', 'Megabytes') if 'Bytes' in x else x, inplace=True)
    df_no_outliers.rename(columns=lambda x: x.replace('(ms)', '(s)') if '(ms)' in x else x, inplace=True)

    # Show sample table after conversion and renaming
    st.subheader("Sample Data After Conversion and Renaming")
    st.write(df_no_outliers.head(5))  # Display the updated DataFrame

    # Display the renamed column headers
    st.write("Updated Column Headers:")
    st.write(list(df_no_outliers.columns))

    # Additional formatting: Converting 'Dur. (s)' to integer type
    st.subheader("Additional Formatting")
    if 'Dur. (s)' in df_no_outliers.columns:
        df_no_outliers['Dur. (s)'] = df_no_outliers['Dur. (s)'].astype(int)
        st.write("Formatted 'Dur. (s)' as integer type.")

    # Display updated data types
    st.write("Updated data types:")
    st.write(df_no_outliers.dtypes)

    ### Segment 3: Exploratory Data Analysis (EDA) ###
    st.subheader("Exploratory Data Analysis (EDA)")

    # Head of the dataset
    st.write("**Dataset Head**")
    st.dataframe(df_user_overview.head())

    # Data Info
    st.write("**Data Information**")
    buffer = io.StringIO()
    df_user_overview.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Histograms and Boxplots for Key Columns
    st.write("**Session Duration Histogram**")
    plt.figure(figsize=(14, 8))
    sns.histplot(df_user_overview['Dur. (ms)'], kde=True)
    plt.title('Distribution of Session Duration')
    plt.xlabel('Session Duration (s)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    for column in ['Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Megabytes)']:
        st.write(f"**Distribution of {column}**")
        plt.figure(figsize=(14, 8))
        sns.histplot(df_user_overview[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Boxplots for Key Columns
    st.write("**Session Duration Boxplot**")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_user_overview['Dur. (ms)'])
    plt.title('Boxplot of Session Duration')
    plt.xlabel('Session Duration (s)')
    st.pyplot(plt)

    for column in ['Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Megabytes)']:
        st.write(f"**Boxplot of {column}**")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_user_overview[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        st.pyplot(plt)

    ### Segment 4: Findings ###
    st.subheader("Findings")

    # Top 10 Handsets
    st.write("**Top 10 Handsets**")
    top_10_handsets = df_user_overview['Handset Type'].value_counts().head(10)
    st.bar_chart(top_10_handsets)

    # Top 3 Handset Manufacturers
    st.write("**Top 3 Handset Manufacturers**")
    top_3_manufacturers = df_user_overview['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_3_manufacturers)

    # Top 5 Handsets for Each Manufacturer
    st.write("**Top 5 Handsets for Each of the Top 3 Manufacturers**")
    for manufacturer in top_3_manufacturers.index:
        st.write(f"Top 5 handsets for {manufacturer}")
        top_handsets_per_manufacturer = df_user_overview[df_user_overview['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.bar_chart(top_handsets_per_manufacturer)

    ### Segment 5: Aggregated Data ###
    st.subheader("Aggregated Data")

    # Aggregated statistics by manufacturer
    aggregated_data = df_user_overview.groupby('Handset Manufacturer').agg({
        'Dur. (ms)': 'sum', 
        'Activity Duration DL (ms)': 'sum'
    }).reset_index()
    st.table(aggregated_data)

    ### Segment 6: Visualization ###
    st.subheader("Visualization")

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    corr = df_user_overview.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    ### Additional Visualizations ###
    st.subheader("Additional Visualizations")

    # Heatmap of Average Data Volume by Application
    application_columns = [
        'Social Media DL (Megabytes)', 'Social Media UL (Megabytes)',
        'Youtube DL (Megabytes)', 'Youtube UL (Megabytes)',
        'Netflix DL (Megabytes)', 'Netflix UL (Megabytes)',
        'Google DL (Megabytes)', 'Google UL (Megabytes)',
        'Email DL (Megabytes)', 'Email UL (Megabytes)',
        'Gaming DL (Megabytes)', 'Gaming UL (Megabytes)',
        'Other DL (Megabytes)', 'Other UL (Megabytes)'
    ]

    user_aggregated_data = df_user_overview.groupby('IMSI').agg({
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Megabytes)': 'sum',
        **{col: 'sum' for col in application_columns}
    }).rename(columns={'IMSI': 'Number of xDR Sessions'})

    # Add total data columns for each application
    for app in ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']:
        user_aggregated_data[f'{app} Total Data (Megabytes)'] = user_aggregated_data[f'{app} DL (Megabytes)'] + user_aggregated_data[f'{app} UL (Megabytes)']

    # Plot Average Total Data Volume by Application
    st.write("**Heatmap of Average Total Data Volume by Application**")
    heatmap_data = user_aggregated_data.filter(like='Total Data (Megabytes)').mean()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data.values.reshape(1, -1), annot=True, cmap='coolwarm', xticklabels=heatmap_data.index, yticklabels=['Average'])
    plt.title('Heatmap of Average Total Data Volume by Application')
    plt.xlabel('Application')
    plt.ylabel('Average')
    st.pyplot(plt)

    # Distribution of Session Durations and Total Data
    st.write("**Distribution of Session Durations and Total Data**")
    plt.figure(figsize=(12, 6))
    sns.histplot(user_aggregated_data['Dur. (s)'], bins=30, kde=True)
    plt.title('Distribution of Session Durations')
    plt.xlabel('Session Duration (s)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.histplot(user_aggregated_data['Total DL (Bytes)'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Total Download Data')
    plt.xlabel('Total Download Data (MB)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    sns.histplot(user_aggregated_data['Total UL (Megabytes)'], bins=30, kde=True, color='orange')
    plt.title('Distribution of Total Upload Data')
    plt.xlabel('Total Upload Data (MB)')
    plt.ylabel('Frequency')
    st.pyplot(plt)