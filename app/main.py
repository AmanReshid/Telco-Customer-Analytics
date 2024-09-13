import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from load_data import load_data_using_sqlalchemy
from utils import missing_values_table, convert_bytes_to_megabytes, outliers_table, fix_outlier, convert_ms_to_seconds

# Define your data loading function
def load_data():
    # This should be your actual data loading logic
    query = "SELECT * FROM xdr_data;"
    df = load_data_using_sqlalchemy(query)
    return df

# Load the data
df = load_data()

# Streamlit Sidebar for user interaction
st.sidebar.title('Dashboard Navigation')
options = st.sidebar.radio('Select an option:', ['User Overview Analysis', 'User Engagement Analysis', 'User Experience Analysis'])

if options == 'User Overview Analysis':
    st.subheader('User Overview Analysis')


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
    st.dataframe(missing_values_table(df))

    # Fixing null values: Example - Filling missing numeric values with median
    st.write("Dropping missing MSISDN/Number:")
    df.dropna(subset=['MSISDN/Number'], inplace=True)
    st.dataframe(missing_values_table(df))

    st.write("Replacing the missing Avg RTT DL(ms) and Avg RTT UL(ms) by their mean values:")
    # Calculate mean values
    mean_rtt_dl = df['Avg RTT DL (ms)'].mean()
    mean_rtt_ul = df['Avg RTT UL (ms)'].mean()

    # Fill missing values with mean
    df['Avg RTT DL (ms)'].fillna(mean_rtt_dl, inplace=True)
    df['Avg RTT UL (ms)'].fillna(mean_rtt_ul, inplace=True)

    df['Dur. (ms)'] = df['Dur. (ms)'].fillna(df['Dur. (ms)'].median())
    df['IMSI'] = df['IMSI'].fillna('Unknown')  # Example: Filling categorical missing values with 'Unknown'
    st.write(df.isnull().sum())

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
    df_no_outliers = df.copy()
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
    st.dataframe(df.head())

    # Data Info
    st.write("**Data Information**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Histograms and Boxplots for Key Columns
    st.write("**Session Duration Histogram**")
    plt.figure(figsize=(14, 8))
    sns.histplot(df['Dur. (ms)'], kde=True)
    plt.title('Distribution of Session Duration')
    plt.xlabel('Session Duration (s)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    for column in ['Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Megabytes)']:
        st.write(f"**Distribution of {column}**")
        plt.figure(figsize=(14, 8))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Boxplots for Key Columns
    st.write("**Session Duration Boxplot**")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Dur. (ms)'])
    plt.title('Boxplot of Session Duration')
    plt.xlabel('Session Duration (s)')
    st.pyplot(plt)

    for column in ['Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Megabytes)']:
        st.write(f"**Boxplot of {column}**")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        st.pyplot(plt)

    ### Segment 4: Findings ###
    st.subheader("Findings")

    # Top 10 Handsets
    st.write("**Top 10 Handsets**")
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    st.bar_chart(top_10_handsets)

    # Top 3 Handset Manufacturers
    st.write("**Top 3 Handset Manufacturers**")
    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_3_manufacturers)

    # Top 5 Handsets for Each Manufacturer
    st.write("**Top 5 Handsets for Each of the Top 3 Manufacturers**")
    for manufacturer in top_3_manufacturers.index:
        st.write(f"Top 5 handsets for {manufacturer}")
        top_handsets_per_manufacturer = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.bar_chart(top_handsets_per_manufacturer)

    ### Segment 5: Aggregated Data ###
    st.subheader("Aggregated Data")

    # Aggregated statistics by manufacturer
    aggregated_data = df.groupby('Handset Manufacturer').agg({
        'Dur. (ms)': 'sum', 
        'Activity Duration DL (ms)': 'sum'
    }).reset_index()
    st.table(aggregated_data)

    ### Segment 6: Visualization ###
    st.subheader("Visualization")

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    corr = df.corr()
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

    user_aggregated_data = df.groupby('IMSI').agg({
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

elif options == 'User Engagement Analysis':
    st.subheader('User Engagement Analysis')
    
    
    # Convert columns with large numbers to numeric
    columns_to_convert = [
        'Dur. (ms)', 'Activity Duration DL (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'
    ]
    for column in columns_to_convert:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            st.write(f"Column missing: {column}")

    # Data Processing
    try:
        grouped_df = df.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum',
            'Activity Duration DL (ms)': 'sum',
            'Activity Duration UL (ms)': 'sum'
        }).reset_index()
    
        grouped_df['Total Traffic (Bytes)'] = grouped_df['Total DL (Bytes)'] + grouped_df['Total UL (Bytes)']
    
        # Normalization
        columns_to_normalize = ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']
        scaler = MinMaxScaler()
        grouped_df[columns_to_normalize] = scaler.fit_transform(grouped_df[columns_to_normalize])
    
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        grouped_df['Cluster'] = kmeans.fit_predict(grouped_df[columns_to_normalize])
    
        # Visualization: Total Traffic vs Total Duration
        fig, ax = plt.subplots()
        scatter = ax.scatter(grouped_df['Dur. (ms)'], grouped_df['Total Traffic (Bytes)'], c=grouped_df['Cluster'], cmap='viridis')
        ax.set_title("Customer Engagement Clusters (k=3)")
        ax.set_xlabel("Normalized Session Duration (ms)")
        ax.set_ylabel("Normalized Total Traffic (Bytes)")
        fig.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
        # Visualization: Total Traffic vs Activity Duration
        fig, ax = plt.subplots()
        scatter = ax.scatter(grouped_df['Activity Duration DL (ms)'], grouped_df['Total Traffic (Bytes)'], c=grouped_df['Cluster'], cmap='viridis')
        ax.set_title("Customer Engagement Clusters (k=3)")
        ax.set_xlabel("Normalized Activity Duration DL (ms)")
        ax.set_ylabel("Normalized Total Traffic (Bytes)")
        fig.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
        # Cluster Summary
        cluster_summary = grouped_df.groupby('Cluster').mean()
        st.write("Cluster Summary:")
        st.write(cluster_summary)
        
    except KeyError as e:
        st.write(f"Error: {e}")

elif options == 'User Experience Analysis':
    st.subheader('User Experience Analysis')

    # Filter columns
    columns = [
        'IMSI', 'Handset Type', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)'
    ]
    df_filtered = df[columns].copy()
    
    # Data Cleaning and Processing
    df_filtered = df_filtered.dropna(subset=['IMSI', 'Handset Type'])
    mean_rtt_dl = df_filtered['Avg RTT DL (ms)'].mean()
    mean_rtt_ul = df_filtered['Avg RTT UL (ms)'].mean()
    df_filtered['Avg RTT DL (ms)'].fillna(mean_rtt_dl, inplace=True)
    df_filtered['Avg RTT UL (ms)'].fillna(mean_rtt_ul, inplace=True)
    
    # Convert units
    df_filtered['Avg RTT DL (s)'] = df_filtered['Avg RTT DL (ms)'] / 1000
    df_filtered['Avg RTT UL (s)'] = df_filtered['Avg RTT UL (ms)'] / 1000
    df_filtered['TCP DL Retrans. Vol (MB)'] = df_filtered['TCP DL Retrans. Vol (Bytes)'] / (1024 * 1024)
    df_filtered['TCP UL Retrans. Vol (MB)'] = df_filtered['TCP UL Retrans. Vol (Bytes)'] / (1024 * 1024)
    
    # Compute new columns
    df_filtered['Total TCP Retransmission'] = df_filtered['TCP DL Retrans. Vol (MB)'] + df_filtered['TCP UL Retrans. Vol (MB)']
    df_filtered['Total RTT'] = df_filtered['Avg RTT DL (s)'] + df_filtered['Avg RTT UL (s)']
    df_filtered['Total Throughput'] = df_filtered['Avg Bearer TP DL (kbps)'] + df_filtered['Avg Bearer TP UL (kbps)']
    
    # Display top, bottom, and frequent values
    st.write("Top 10 TCP Retransmission Values:", df_filtered['Total TCP Retransmission'].nlargest(10))
    st.write("Bottom 10 TCP Retransmission Values:", df_filtered['Total TCP Retransmission'].nsmallest(10))
    st.write("Most Frequent TCP Retransmission Values:", df_filtered['Total TCP Retransmission'].value_counts().head(10))
    
    st.write("Top 10 RTT Values:", df_filtered['Total RTT'].nlargest(10))
    st.write("Bottom 10 RTT Values:", df_filtered['Total RTT'].nsmallest(10))
    st.write("Most Frequent RTT Values:", df_filtered['Total RTT'].value_counts().head(10))
    
    st.write("Top 10 Throughput Values:", df_filtered['Total Throughput'].nlargest(10))
    st.write("Bottom 10 Throughput Values:", df_filtered['Total Throughput'].nsmallest(10))
    st.write("Most Frequent Throughput Values:", df_filtered['Total Throughput'].value_counts().head(10))

    # Average throughput and TCP retransmission per handset type
    throughput_per_handset = df_filtered.groupby('Handset Type')['Total Throughput'].mean().reset_index()
    tcp_retrans_per_handset = df_filtered.groupby('Handset Type')['Total TCP Retransmission'].mean().reset_index()

    st.bar_chart(throughput_per_handset.set_index('Handset Type'), use_container_width=True)
    st.bar_chart(tcp_retrans_per_handset.set_index('Handset Type'), use_container_width=True)

    # K-Means clustering
    clustering_data = df_filtered[['Total TCP Retransmission', 'Total RTT', 'Total Throughput']].copy()
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['Cluster'] = kmeans.fit_predict(clustering_data_scaled)

    # Display cluster analysis
    cluster_analysis = df_filtered.groupby('Cluster').mean()[['Total TCP Retransmission', 'Total RTT', 'Total Throughput']]
    st.write(cluster_analysis)
    
    # Visualization
    st.write("Cluster Analysis Visualization")
    fig, ax = plt.subplots()
    sns.pairplot(df_filtered[['Total TCP Retransmission', 'Total RTT', 'Total Throughput', 'Cluster']], hue='Cluster', ax=ax)
    st.pyplot(fig)
