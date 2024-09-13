# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("User Engagement Analysis Dashboard")
st.markdown("This dashboard provides insights into user engagement, experience, and overall trends.")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "User Engagement", "User Experience"])

# Upload your dataset (CSV or Excel)
st.sidebar.subheader("Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success("File uploaded successfully!")

    # Overview section
    if section == "Overview":
        st.header("Data Overview")
        st.write("Dataset Preview")
        st.dataframe(df.head())  # Display the first few rows of the data
        
        st.write("Basic Statistics")
        st.write(df.describe())  # Display statistics like mean, median, etc.
    
    # User Engagement section
    if section == "User Engagement":
        st.header("User Engagement Insights")
        
        # Example: Displaying the distribution of a specific column
        column = st.selectbox("Select column to analyze", df.columns)
        
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # User Experience section
    if section == "User Experience":
        st.header("User Experience Insights")
        
        # Example: Group data by a certain column and calculate statistics
        group_column = st.selectbox("Select column to group by", df.columns)
        group_data = df.groupby(group_column).mean()
        
        st.write(f"Average metrics grouped by {group_column}")
        st.dataframe(group_data)

else:
    st.warning("Please upload a file to get started.")
