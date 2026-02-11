# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:17:37 2026

Create ML to match Davis raw data to PurpleAir corrected data
(PA corrected as secondary standard)

@author: kokorn
"""
#Import helpful toolboxes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

#load in the sheet of matching pairs of PA, Davis, & regulatory data
sites = pd.read_excel("C:\\Users\\kokorn\\Documents\\GLOBE PM2.5\\PA_Davis_Matching.xlsx")

#make an overall df to hold all of our data
total_data = pd.DataFrame()

#loop through each site to load the associated data
for n in range(len(sites)):
    #get the purpleair & davis filenames
    PA_filename = sites['PurpleAir'].iloc[n] + '.csv'
    D_filename = sites['Davis'].iloc[n] + '.csv'
    #load in the data
    PA = pd.read_csv("C:\\Users\\kokorn\\Documents\\GLOBE PM2.5\\{}".format(PA_filename),index_col=0)
    D = pd.read_csv("C:\\Users\\kokorn\\Documents\\GLOBE PM2.5\\{}".format(D_filename),skiprows=5,encoding='latin1',index_col=0)
    #standardize the datetime indexes
    PA.index = pd.to_datetime(PA.index, errors='coerce', utc=True)
    D.index = pd.to_datetime(D.index, errors='coerce', utc=True) # double check later that davis is actually utc
    D.index.name = 'datetimeUTC'
    #have to merge on timestamp to align dates & times at the same location - use nearest bc different time averaging
    merged = pd.merge_asof(PA.sort_values('datetimeUTC'),D.sort_values('datetimeUTC'),on='datetimeUTC',direction='nearest', tolerance=pd.Timedelta('5min') )
    #add the location back to this to keep track of which site (repeating datetimes will occur at different sites)
    merged['Site'] = sites['Location'].iloc[n]
    #add to the rolling list of all time-merged data
    total_data = pd.concat([total_data, merged], ignore_index=True)
#------------------------------------------------------------------------------
#now apply a linear regression algorithm to match Davis to wildfire-corrected PA

#tried letting 'selector' pick features & it picked 4 temperatures + PM1 - overfitting

#drop missing values - can try interpolating later if we need the data
total_data = total_data.dropna()

#split into response variable & features
X = total_data[[ 
'PM 2.5 - ug/m³', 'High PM 2.5 - ug/m³', 'Hum - %']]  # predictors
y = total_data['wildfire']    

#split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit the model
model = LinearRegression()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#add predictions back to main chart
total_data['y_pred'] = model.predict(X)

#------------------------------------------------------------------------------
#if there's a regulatory station nearby, use this for validation





#------------------------------------------------------------------------------
#split back up by location & plot the results separately - 1 subplot per location

#initialize number of plots - subplot for each location
fig, axs = plt.subplots(n, 1, figsize=(8, 4 * n))
#If there is only one subplot, make it a list to handle indexing
if n == 1:
    axs = [axs]

for loc, df_loc in total_data.groupby('location'):
    #scatter the raw davis & PA corrected data
    plt.scatter(df_loc['wildfire'], df_loc['PM 2.5 - ug/m³'], color='black',label = 'Raw Davis')
    #then scatter the 'corrected' davis & PA corrected data
    plt.scatter(df_loc['wildfire'], df_loc['y_pred'], color='grey',label = 'Corrected Davis' )
    plt.title('{}'.format(loc))
    plt.legend()
    plt.show()
    