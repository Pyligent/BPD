
# coding: utf-8

# ### Section 1
# ###  The City of Baltimore maintains a database of parking citations issued within the city. More information about the dataset can be found [here](https://data.baltimorecity.gov/Transportation/Parking-Citations/n4ma-fj3m). You can download the dataset as a CSV file [here](https://data.baltimorecity.gov/api/views/n4ma-fj3m/rows.csv). Unless stated otherwise, you should only consider citations written before January 1, 2019.
# 

# In[68]:


import pandas as pd
import numpy as np


# In[4]:


Baltimore_Parking_Data = pd.read_csv('./Dataset/Parking_Citations.csv')
Baltimore_Parking_Data.head()


# In[7]:


Baltimore_Parking_Data.count()


# In[8]:


Baltimore_Parking_Data['Citation'].nunique()


# In[9]:


Baltimore_Parking_Data.isnull().sum()


# ### 1. For all citations, what is the mean violation fine?

# In[15]:


mean_ViolFine = round(Baltimore_Parking_Data['ViolFine'].mean(skipna = True),9)


# In[16]:


mean_ViolFine


# ### 2. Find the police district that has the highest mean violation fine. What is that mean violation fine? Keep in mind that Baltimore is divided into nine police districts, so clean the data accordingly.

# In[19]:


Baltimore_Parking_Data['PoliceDistrict'].value_counts()


# In[21]:


Baltimore_Parking_Data['PoliceDistrict'] = Baltimore_Parking_Data['PoliceDistrict'].str.lower()


# In[22]:


Baltimore_Parking_Data['PoliceDistrict'].value_counts()


# ### Official Districts Name (https://en.wikipedia.org/wiki/Baltimore_Police_Department)
# 1-Central    
# 2-Southeast    
# 3-Eastern    
# 4-Northeast     
# 5-Northern     
# 6-Northwest     
# 7-Western    
# 8-Southwest    
# 9-Southern    

# In[24]:


# Replace notheastern to northeastern
Baltimore_Parking_Data['PoliceDistrict'] = Baltimore_Parking_Data['PoliceDistrict'].str.replace('notheastern','northeastern')


# In[25]:


Baltimore_Parking_Data['PoliceDistrict'].value_counts()


# In[26]:


BPD_District_meanVioFine = Baltimore_Parking_Data.groupby('PoliceDistrict', as_index=False)['ViolFine'].mean()


# In[29]:


BPD_District_meanVioFine.rename(columns = {'ViolFine':'mean_ViolFine'},inplace = True)
BPD_District_meanVioFine.sort_values(by=['mean_ViolFine'],inplace=True,ascending=False)
BPD_District_meanVioFine


# In[31]:


highest_mean_VioFine = BPD_District_meanVioFine['mean_ViolFine'].tolist()[0]
highest_mean_VioFine = round(highest_mean_VioFine,9)


# In[171]:


print('the highest mean violation fine:',highest_mean_VioFine)


# ### 3. First, find the total number of citations given in each year between 2004 and 2014 (inclusive). Next, using linear regression, create a function that plots the total number of citations as a function of the year. If you were to plot a line using this function, what would be the slope of that line?

# In[33]:


Baltimore_Parking_Data.dtypes


# In[44]:


Baltimore_Parking_Data['Year'] = pd.DatetimeIndex(Baltimore_Parking_Data['ViolDate']).year


# In[45]:


Baltimore_Parking_Data.head()


# In[49]:


Baltimore_Parking_Data['Year'].fillna(0, inplace=True)
Baltimore_Parking_Data['Year'] = Baltimore_Parking_Data['Year'].astype(int)
Baltimore_Parking_Data['Year'].value_counts()


# In[52]:


Citation_number = Baltimore_Parking_Data.groupby('Year', as_index=False)['Citation'].count()


# In[53]:


Citation_number.head()


# In[54]:


Citation_number_2004to2014 = Citation_number[(Citation_number['Year'] >=2004) & (Citation_number['Year']<=2014)]
Citation_number_2004to2014


# In[55]:


import matplotlib.pyplot as plt
from scipy import stats


# In[56]:


x = Citation_number_2004to2014['Year'].values
y = Citation_number_2004to2014['Citation'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
slope = round(slope,9)
print('slope: %f' % slope)


# In[58]:


plt.plot(x, y, 'o', label='2004 to 2014 Citation numbers')
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.legend()
plt.show()


# ### 4. Looking only at vehicles that have open penalty fees, what dollar amount is the 81st percentile of that group?

# In[62]:


Baltimore_Parking_Data['OpenPenalty'].value_counts()


# In[66]:


OpenPenalty_data = Baltimore_Parking_Data.filter(['OpenPenalty'],axis = 1)
OpenPenalty_data_no_null = OpenPenalty_data[OpenPenalty_data['OpenPenalty'] != 0]
OpenPenalty_data_no_null.head()


# In[73]:


OpenPenalty_data_no_null.describe()


# In[173]:


per_81 = np.percentile(OpenPenalty_data_no_null['OpenPenalty'].values,81)
print('81st percentile of Open Penalty is',round(per_81,9))


# ### 5. Find the ten vehicle makes that received the most citations during 2017. For those top ten, find all Japanese-made vehicles. What proportion of all citations were written for those vehicles? Note that the naming in Make is not consistent over the whole dataset, so you will need to clean the data before calculating your answer. Your answer should be expressed as a decimal number (i.e. 0.42, not 42).

# In[109]:


BDP_Citation_2017 = Baltimore_Parking_Data[Baltimore_Parking_Data['Year'] == 2017]
BDP_Citation_2017.head()


# In[111]:


# Fuzzy Match: Clean the Car Make Name, Short to three chats, then groupby
BDP_Citation_2017['Make'] = BDP_Citation_2017['Make'].str[0:3]
BDP_Citation_2017.head()


# In[112]:


BDP_Citation_2017_CarMaker = BDP_Citation_2017.groupby('Make', as_index=False)['Citation'].count()


# In[114]:


BDP_Citation_2017_CarMaker.sort_values(by=['Citation'],inplace=True,ascending=False)
BDP_Citation_2017_CarMaker.head(10)


# In[115]:


total_citation = len(BDP_Citation_2017['Citation']) 
total_citation


# In[124]:


# Japanese Car Maker: HON-HONDA,TOY-TOYOTA,NIS-NISSAN,ACU-ACURA
Jap_Citation = BDP_Citation_2017_CarMaker.loc[(BDP_Citation_2017_CarMaker['Make'] == 'HON') |                                                (BDP_Citation_2017_CarMaker['Make'] == 'TOY') |                                                (BDP_Citation_2017_CarMaker['Make'] == 'NIS') |                                                (BDP_Citation_2017_CarMaker['Make'] == 'ACU') ]
Jap_Citation


# In[126]:


Jap_Citation_num = sum(Jap_Citation['Citation'].tolist())
Jap_Citation_num


# In[167]:


print('all Japanese-made vehicles proportion:',round(Jap_Citation_num/total_citation,9) )


# ### 6. To answer this last question, you will need to download another dataset: The Baltimore Police Department Victim Based Crime Dataset. (CSV file is available [here](https://data.baltimorecity.gov/api/views/wsfq-mvij/rows.csv).) First, determine how many instances of auto theft ocurred in each police district during 2015. Next, determine the number of parking citations that were issued in each police district during the same year. Finally, determine the ratio of auto thefts to parking citations for each district. Out of the nine police districts, what was the highest ratio?

# In[129]:


Baltimore_Crime_Data = pd.read_csv('./Dataset/BPD_Part_1_Victim_Based_Crime_Data.csv')
Baltimore_Crime_Data.head()


# In[134]:


Baltimore_Crime_Data['Year'] = pd.DatetimeIndex(Baltimore_Crime_Data['CrimeDate']).year
Baltimore_Crime_Data.head()


# In[131]:


Baltimore_Crime_Data['District'].value_counts()


# In[132]:


Baltimore_Crime_Data['Description'].value_counts()


# In[135]:


# 2015 Data
Baltimore_Crime_2015_Data = Baltimore_Crime_Data[Baltimore_Crime_Data['Year'] == 2015]
Baltimore_Crime_2015_Data.head()


# In[136]:


Auto_Theft_2015_Data = Baltimore_Crime_2015_Data[Baltimore_Crime_2015_Data['Description'] == 'AUTO THEFT']
Auto_Theft_2015_Data.head()


# ####  6.1 how many instances of auto theft ocurred in each police district during 2015.

# In[148]:


Auto_Theft_2015_District = Auto_Theft_2015_Data.groupby('District', as_index=False)['Description'].count()
Auto_Theft_2015_District


# In[149]:


Auto_Theft_2015_District = Auto_Theft_2015_District[Auto_Theft_2015_District['District'] != 'UNKNOWN']
Auto_Theft_2015_District


# In[151]:


Auto_Theft_2015_District.columns = ['PoliceDistrict','Number of AutoTheft']
Auto_Theft_2015_District


# #### 6.2 determine the number of parking citations that were issued in each police district during the same year.

# In[140]:


BDP_Citation_2015 = Baltimore_Parking_Data[Baltimore_Parking_Data['Year'] == 2015]


# In[141]:


BDP_Citation_2015['PoliceDistrict'].value_counts()


# In[153]:


BDP_Citation_2015_District = BDP_Citation_2015.groupby('PoliceDistrict', as_index=False)['Citation'].count()
BDP_Citation_2015_District


# In[155]:


BDP_Citation_2015_District['PoliceDistrict'] = BDP_Citation_2015_District['PoliceDistrict'].str.upper()
BDP_Citation_2015_District


# #### 6.3 determine the ratio of auto thefts to parking citations for each district

# In[158]:


Auto_Theft_2015_District


# In[159]:


# Match the Police District Name REPLACE Northeast with Northeastern etc..
Auto_Theft_2015_District['PoliceDistrict'] = Auto_Theft_2015_District['PoliceDistrict'].str.replace('NORTHEAST','NORTHEASTERN')

Auto_Theft_2015_District['PoliceDistrict'] = Auto_Theft_2015_District['PoliceDistrict'].str.replace('NORTHWEST','NORTHWESTERN')

Auto_Theft_2015_District['PoliceDistrict'] = Auto_Theft_2015_District['PoliceDistrict'].str.replace('SOUTHEAST','SOUTHEASTERN')

Auto_Theft_2015_District['PoliceDistrict'] = Auto_Theft_2015_District['PoliceDistrict'].str.replace('SOUTHWEST','SOUTHWESTERN')

Auto_Theft_2015_District


# In[161]:


ratio_data = pd.merge(Auto_Theft_2015_District,BDP_Citation_2015_District, on = 'PoliceDistrict')
ratio_data


# In[162]:


ratio_data['ratio'] = ratio_data['Number of AutoTheft']/ratio_data['Citation']


# In[165]:


ratio_data.sort_values(by=['ratio'],ascending= False, inplace = True)
ratio_data


# In[166]:


Highest_ratio = ratio_data['ratio'].tolist()[0]
print('The highest ratio: ', round(Highest_ratio,9))

