# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:51:53 2026

Pull PA data from pyrsig-0.11.1 (may be incompatible with other versions)
No framework to limit to just outdoor sensors other than keeping the bounding box small
Need to filter by exact lat/lon from PA map (not sensor ID)
Downloads parallelized for speed

@author: okorn
"""

#Import helpful toolboxes
import pyrsig
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

#-----------------------------------
#list your pa api key (should not charge it through pyrsig)
API_KEY = '056020E1-0BF4-11F0-81BE-42010A80001F'  

#have to specify sensors by lat/lon instead of ID or site name
lats = [37.57338, 37.1038] #kristen, margaret
lons = [-122.343153, -76.387]
buffer = 0.001 #degrees

#set the full date range we want to pull from
start_date = datetime(2026, 2, 1)
end_date   = datetime.utcnow()

#-----------------------------------
#download a single day of data at a time
def download_day(day_str):
    day = datetime.strptime(day_str, "%Y-%m-%d")
    next_day = day + timedelta(days=1)
    attempt = 0
    while attempt < 3:
        try:
            #use the api to pull
            rsigapi = pyrsig.RsigApi(bdate=day.strftime("%Y-%m-%d"),edate=next_day.strftime("%Y-%m-%d"))
            rsigapi.purpleair_kw["api_key"] = API_KEY
            #pull just the corrected EPA values
            df = rsigapi.to_dataframe("purpleair.pm25_corrected")
            return day_str, df
        except Exception as e:
            attempt += 1
            time.sleep(5)
    # Failed after 3 attempts
    return day_str, None

#-----------------------------------
#parallelized download loop (for speed)
if __name__ == "__main__":
    #get list of days to keep track
    all_days = []
    day = start_date
    while day < end_date:
        all_days.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)

    #now do the actual download
    all_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_day = {executor.submit(download_day, d): d for d in all_days}
        for future in as_completed(future_to_day):
            day_str = future_to_day[future]
            try:
                day_str, df = future.result()
                if df is not None:
                    all_results.append((day_str, df))
                else:
                    continnue
            except Exception as e:
                print(f"Error downloading {day_str}: {e}")

    #-----------------------------------
    #now set up the bounding boxes for each sensor
    bbox_data = [[] for _ in range(len(lats))]
    
    for day_str, df in all_results:
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            bbox = (lon-buffer, lat-buffer, lon+buffer, lat+buffer)
            df_bbox = df[(df["LONGITUDE(deg)"] >= bbox[0]) &
                         (df["LONGITUDE(deg)"] <= bbox[2]) &
                         (df["LATITUDE(deg)"] >= bbox[1]) &
                         (df["LATITUDE(deg)"] <= bbox[3])]
            bbox_data[i].append(df_bbox)
    
    #-----------------------------------
    #save new csv for each sensor
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if bbox_data[i]:
            final_df = pd.concat(bbox_data[i], ignore_index=True)
            #now save to file
            savepath = 'C:\\Users\\okorn\\Documents\\GLOBE PM2.5\\PA_{}_{}.csv'.format(lat,lon)
            final_df.to_csv(savepath, index=False)
