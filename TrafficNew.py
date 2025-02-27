#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


pip install kagglehub


# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('C:/Users/kerim/.cache/kagglehub/datasets/saurabhshahane/road-traffic-accidents/versions/3/RTA Dataset.csv')  # Replace backslashes with forward slashes

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())


# In[4]:


print("\nDataset Information:")
print(df.info())


# In[44]:


print("\nSummary Statistics:")
print(df.describe(include='object'))


# In[43]:


# describe categorical columns
df.describe(include='object')


# In[6]:


print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Drop duplicate rows (if any)
df = df.drop_duplicates()
print("\nDuplicate rows removed.")


# In[7]:


# Handle missing values (if any)
# For numerical columns, fill missing values with the median
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# For categorical columns, fill missing values with the mode
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

print("\nMissing values handled.")

# Convert 'Time' column to datetime format
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')  # Specify the correct format
    print("\n'Time' column converted to datetime format.")

# Add a new column for 'Hour' extracted from the 'Time' column
if 'Time' in df.columns:
    df['Hour'] = df['Time'].dt.hour
    print("\n'Hour' column added.")


# In[8]:


# Convert 'Time' column to datetime format
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')  # Specify the correct format
    print("\n'Time' column converted to datetime format.")

# Add a new column for 'Hour' extracted from the 'Time' column
if 'Time' in df.columns:
    df['Hour'] = df['Time'].dt.hour
    print("\n'Hour' column added.")


# In[25]:


# Query 1: Accidents during peak hours (8 AM - 10 AM and 5 PM - 7 PM)
peak_hours_accidents = df[(df['Hour'] >= 8) & (df['Hour'] <= 10) | (df['Hour'] >= 17) & (df['Hour'] <= 19)]
print("\nNumber of accidents during peak hours (8 AM - 10 AM and 5 PM - 7 PM):")
print(peak_hours_accidents.shape[0])


# In[32]:


# Query 2: Accidents on weekends (Saturday and Sunday)
if 'Day_of_week' in df.columns:
    weekend_accidents = df[df['Day_of_week'].isin(['Saturday', 'Sunday'])]
    print("\nNumber of accidents on weekends (Saturday and Sunday):")
    print(weekend_accidents.shape[0])


# In[33]:


# Query 3: Accidents in adverse weather conditions (Rain, Fog, Snow)
if 'Weather_conditions' in df.columns:
    adverse_weather_accidents = df[df['Weather_conditions'].isin(['Rain', 'Fog', 'Snow'])]
    print("\nNumber of accidents in adverse weather conditions (Rain, Fog, Snow):")
    print(adverse_weather_accidents.shape[0])


# In[47]:


# Query 4: Accidents with high severity (Severity = 3)
if 'Accident_severity' in df.columns:
    high_severity_accidents = df[df['Accident_severity'] == 3]
    print("\nNumber of high-severity accidents (Severity = 3):")
    print(high_severity_accidents.shape[0])


# In[46]:


def fatality_df(column, df=df, sort=False):
    """
    fetches a dataframe having category wise fatality frequency
    """
# finding out the relationship between Accident severity and a column
    df_hello = df.groupby(['Accident_severity', column]).Time.count().reset_index()

    # creating a list of all categories to plot
    rowlist = [row for row in df_hello[column]]
    sumlist = []
    for row in rowlist:
        sumlist.append(df_hello.loc[df_hello[column] == row].Time.sum())

    df_hello['sum'] = sumlist
    df_hello['ratio'] = df_hello['Time']/df_hello['sum']
    df_final = df_hello.loc[df_hello.Accident_severity=='Fatal injury']
    if sort==True:
        df_final = df_final.sort_values(by='ratio')
    return df_final

# example
fatal_collisiontype_df = fatality_df('Type_of_collision', sort=True)
fatal_collisiontype_df


# In[45]:


# Query 5: Accidents on highways
if 'Road_type' in df.columns:
    highway_accidents = df[df['Road_type'] == 'Highway']
    print("\nNumber of accidents on highways:")
    print(highway_accidents.shape[0])


# In[36]:


# Query 6: Accidents with the highest number of casualties
if 'Number_of_casualties' in df.columns:
    max_casualties_accident = df[df['Number_of_casualties'] == df['Number_of_casualties'].max()]
    print("\nAccident with the highest number of casualties:")
    print(max_casualties_accident)


# In[30]:





# In[15]:


# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_road_traffic_accidents.csv', index=False)
print("\nCleaned dataset saved to 'cleaned_road_traffic_accidents.csv'.")


# In[16]:


# Data Visualization
# Set the style for seaborn
sns.set(style="whitegrid")


# In[17]:


# 1. Distribution of accidents by time of day
plt.figure(figsize=(12, 6))
sns.histplot(df['Hour'], bins=24, kde=True, color='blue')
plt.title('Distribution of Accidents by Time of Day', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()


# In[38]:


# 2. Distribution of accidents by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_week', data=df, palette='viridis', order=df['Day_of_week'].value_counts().index)
plt.title('Distribution of Accidents by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True)
plt.show()


# In[40]:


# 3. Distribution of accidents by weather conditions
plt.figure(figsize=(12, 6))
sns.countplot(x='Weather_conditions', data=df, palette='magma', order=df['Weather_conditions'].value_counts().index)
plt.title('Distribution of Accidents by Weather Conditions', fontsize=16)
plt.xlabel('Weather Conditions', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[41]:


# 4. Correlation heatmap of numerical variables
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables', fontsize=16)
plt.show()


# In[42]:


# 5. Box plot of accidents by road type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Road_Type', y='Number_of_Casualties', data=df, palette='Set2')
plt.title('Box Plot of Accidents by Road Type', fontsize=16)
plt.xlabel('Road Type', fontsize=14)
plt.ylabel('Number of Casualties', fontsize=14)
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:


# 6. Scatter plot of accidents by location
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Accident_Severity', data=df, palette='viridis')
plt.title('Scatter Plot of Accidents by Location', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.grid(True)
plt.show()

