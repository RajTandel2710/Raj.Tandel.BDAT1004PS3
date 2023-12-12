#!/usr/bin/env python
# coding: utf-8

# # Question-1

# In[5]:


import pandas as pd


# In[6]:


#extracting the dataset
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
users = pd.read_csv(url, sep='|')


# In[7]:


mean_age_per_occupation = users.groupby('occupation')['age'].mean()
print(mean_age_per_occupation)


# In[8]:


def male_ratio(group):
    return (group == 'M').sum() / group.count()

male_ratio_per_occupation = users.groupby('occupation')['gender'].apply(male_ratio).sort_values(ascending=False)


# In[9]:


male_ratio_per_occupation


# In[10]:


age_stats_per_occupation = users.groupby('occupation')['age'].agg(['min', 'max'])


# In[11]:


age_stats_per_occupation


# In[12]:


mean_age_per_occupation_sex = users.groupby(['occupation', 'gender'])['age'].mean()
mean_age_per_occupation_sex


# In[13]:


gender_percentage_per_occupation = users.groupby('occupation')['gender'].value_counts(normalize=True).mul(100).unstack()
gender_percentage_per_occupation


# # Question-2

# In[16]:


#extracting the dataset
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"
euro12 = pd.read_csv(url,sep=',')


# In[17]:


goals = euro12['Goals']


# In[18]:


goals


# In[19]:


num_teams = euro12['Team'].nunique()


# In[33]:


num_teams


# In[34]:


num_columns = euro12.shape[1]
num_columns


# In[35]:


discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline


# In[36]:


discipline_sorted = discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False)
discipline_sorted


# In[37]:


mean_yellow_cards = euro12['Yellow Cards'].mean()
mean_yellow_cards


# In[38]:


teams_more_than_six_goals = euro12[euro12['Goals'] > 6]
teams_more_than_six_goals


# In[39]:


teams_starting_with_G = euro12[euro12['Team'].str.startswith('G')]
teams_starting_with_G


# In[40]:


first_seven_columns = euro12.iloc[:, :7]
first_seven_columns


# In[41]:


all_except_last_three_columns = euro12.iloc[:, :-3]
all_except_last_three_columns


# In[42]:


shooting_accuracy = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]
shooting_accuracy


# # Question-3

# In[20]:


import numpy as np


# In[21]:


series_1 = pd.Series(np.random.randint(1, 5, size=100))
series_2 = pd.Series(np.random.randint(1, 4, size=100))
series_3 = pd.Series(np.random.randint(10000, 30001, size=100))


# In[248]:


series_1


# In[249]:


series_2


# In[250]:


series_3


# In[251]:


data = pd.concat([series_1, series_2, series_3], axis=1)
data


# In[252]:


data.columns = ['bedrs', 'bathrs', 'price_sqr_meter']


# In[253]:


data.columns


# In[254]:


bigcolumn = pd.concat([series_1, series_2, series_3], axis=0)


# In[255]:


bigcolumn


# In[256]:


print("Is it going only until index 99?", bigcolumn.index.max() == 99)


# In[67]:


bigcolumn.reset_index(drop=True, inplace=True)


# In[68]:


bigcolumn.reset_index


# # Question-4

# In[22]:


wind= pd.read_csv('C:/Georgian/BDAT-1004/Problem Set/3/wind.txt', delim_whitespace=True)


# In[23]:


wind['Date'] = pd.to_datetime('20' + wind['Yr'].astype(str) + wind['Mo'].astype(str) + wind['Dy'].astype(str), format='%Y%m%d')
wind = wind.set_index('Date')
wind= wind.drop(['Yr', 'Mo', 'Dy'], axis=1)


# In[24]:


wind


# In[25]:


def fix_year(date):
    if date.year > 2000:
        return date - pd.DateOffset(years=100)
    return date


# In[26]:


wind.index = wind.index.map(fix_year)


# In[27]:


wind.index


# In[28]:


wind.index = pd.to_datetime(wind.index)
wind.index 


# In[29]:


missing_values_per_location = wind.isnull().sum()
missing_values_per_location


# In[30]:


non_missing_values_total = wind.notnull().sum().sum()
non_missing_values_total


# In[31]:


mean_windspeed = wind.stack().mean()
mean_windspeed


# In[32]:


loc_stats = wind.describe().T[['min', 'max', 'mean', 'std']]
loc_stats


# In[33]:


day_stats = wind.resample('D').agg(['min', 'max', 'mean', 'std'])
day_stats


# In[34]:


january_means = wind[wind.index.month == 1].resample('Y').mean()
january_means


# In[35]:


data_yearly = wind.resample('Y').mean()
data_yearly


# In[36]:


data_monthly = wind.resample('M').mean()
data_monthly


# In[37]:


data_weekly = wind.resample('W').mean()
data_weekly


# In[38]:


weekly_stats_first_52 = wind.resample('W').agg(['min', 'max', 'mean', 'std']).iloc[:52]
weekly_stats_first_52


# # Question-5

# In[39]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep='\t')


# In[40]:


chipo


# In[41]:


chipo.head(10)


# In[42]:


num_observations = chipo.shape[0]
num_observations


# In[43]:


num_columns = chipo.shape[1]
num_columns


# In[44]:


chipo.columns


# In[45]:


chipo.index


# In[46]:


most_ordered = chipo.groupby('item_name').quantity.sum().idxmax()
most_ordered


# In[47]:


qty_most_ordered = chipo.groupby('item_name').quantity.sum().max()
qty_most_ordered


# In[48]:


total_items_ordered = chipo.quantity.sum()
total_items_ordered


# In[49]:


chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:-1]))
chipo['item_price']


# In[50]:


chipo['revenue'] = chipo['item_price'] * chipo['quantity']
total_revenue = chipo['revenue'].sum()
total_revenue


# In[51]:


num_orders = chipo['order_id'].nunique()
num_orders


# In[52]:


average_revenue_per_order = total_revenue / num_orders
average_revenue_per_order


# In[53]:


num_different_items = chipo['item_name'].nunique()
num_different_items


# # Question-6

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt


# In[55]:


# load the dataset into dataframe
file_path = 'C:\\Georgian\\BDAT-1004\\Problem Set\\3\\us-marriages-divorces-1867-2014.csv'
marriages_divorces= pd.read_csv(file_path)


# In[56]:


marriages_divorces


# In[57]:


#create a line graph
plt.figure(figsize=(10, 6))

# Plotting marriages and divorces per capita
plt.plot(marriages_divorces['Year'], marriages_divorces['Marriages_per_1000'], label='Marriages')
plt.plot(marriages_divorces['Year'], marriages_divorces['Divorces_per_1000'], label='Divorces')

# Labeling axes and title
plt.xlabel('Year')
plt.ylabel('Per Capita')
plt.title('Number of Marriages and Divorces per Capita in the U.S. (1867 - 2014)')

# Adding legend
plt.legend()

# Display plot
plt.grid(True)
plt.tight_layout()
plt.show()


# # Question-7

# In[58]:


import pandas as pd
import matplotlib.pyplot as plt


# In[59]:


years = [1900, 1950, 2000]
selected_years = data[data['Year'].isin(years)]
selected_years


# In[60]:


# Creating a vertical bar chart
plt.figure(figsize=(10, 6))

bar_width = 3
index = selected_years['Year']

plt.bar(index - bar_width/2, selected_years['Marriages_per_1000'], bar_width, label='Marriages')
plt.bar(index + bar_width/2, selected_years['Divorces_per_1000'], bar_width, label='Divorces')


# Labeling axes and title
plt.xlabel('Year')
plt.ylabel('Per Capita')
plt.title('Number of Marriages and Divorces per Capita in the U.S. (1900, 1950, 2000)')

# Adding legend and ticks
plt.xticks(index, years)
plt.legend()


# # Question-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt


# In[62]:


# Load the dataset
file_path = 'C:\\Georgian\\BDAT-1004\\Problem Set\\3\\actor_kill_counts.csv' 
actor= pd.read_csv(file_path)


# In[63]:


actor


# In[64]:


# Sort the actors by their kill count in descending order

sorted_data = actor.sort_values('Count', ascending=False)
sorted_data

# Create a horizontal bar chart

plt.figure(figsize=(8, 4))

plt.barh(sorted_data['Actor'], sorted_data['Count'], color='orange')

# Labeling axes and title

plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood by Kill Count')

# Display kill count labels on the bars
for i, value in enumerate(sorted_data['Count']):
    plt.text(value, i, str(value), ha='left', va='center')
    
# Display the plot
plt.tight_layout()
plt.show()


# # Question-9

# In[65]:


import pandas as pd
import matplotlib.pyplot as plt


# In[66]:


# Load dataset into a DataFrame
file_path = 'C:\\Georgian\BDAT-1004\\Problem Set\\3\\roman-emperor-reigns.csv'
emperor = pd.read_csv(file_path)


# In[67]:


emperor


# In[68]:


# Count the number of assassinated and non-assassinated emperors

assassinated_counts = emperor['Cause_of_Death'].value_counts()
assassinated_counts


# In[69]:


# Create a pie chart

plt.figure(figsize=(8, 8))
plt.pie(assassinated_counts, labels=assassinated_counts.index, autopct='%1.1f%%', startangle=150)
plt.axis('equal')
plt.title('Fraction of Roman Emperors Assassinated')

# Display the pie chart
plt.show()


# # Question-10

# In[70]:


import pandas as pd
import matplotlib.pyplot as plt


# In[71]:


# Load the dataset
file_path = 'C:\\Georgian\\BDAT-1004\\Problem Set\\3\\arcade-revenue-vs-cs-doctorates.csv' 
revenue= pd.read_csv(file_path)
revenue


# In[73]:


# Create a scatter plot

plt.figure(figsize=(9, 6))
for year in revenue['Year'].unique():
    year_data = revenue[revenue['Year'] == year]
    plt.scatter(year_data['Total Arcade Revenue (billions)'], year_data['Computer Science Doctorates Awarded (US)'], label=year)

# Label the axes

plt.xlabel('Total Revenue earned by Arcades')
plt.ylabel('Number of Computer Science PhDs awarded')
plt.title('Relationship between Total Revenue and Computer Science PhDs')

# Showing  legend to  indicate the year for each color
plt.legend(title='Year')

# Show the scatter plot
plt.grid(True)

