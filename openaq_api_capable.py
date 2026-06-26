# -*- coding: utf-8 -*-
"""
Fetch OpenAQ V3 data for specific known sensors,
pivot parameters (PM2.5, Humidity), apply US EPA correction, and save to CSV.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Configuration
# -----------------------------
path = 'C:\\Users\\kokorn\\Documents\\ECIPES Sensor Database\\OpenAQ\\'
BASE_URL = "https://api.openaq.org/v3"

# OpenAQ V3 REQUIRES an API key. 
OPENAQ_API_KEY = "a8a6c37b6b0e479af479a23dab0d4c636a9386db20ad036b39ce994e942de9b8" 

# The 5 exact locations we want (Hampton NASA + 4 CAPABLE sensors)
TARGET_LOCATION_IDS = [1120, 6348872, 6397024, 6397025, 6406930]

START_DATE = datetime(2026, 1, 1) # Edit this to your desired start date
CHUNK_DAYS = 30  
DELAY_BETWEEN_REQUESTS = 1.0  
MAX_WORKERS = 3  

HEADERS = {
    "Accept": "application/json",
    "X-API-Key": OPENAQ_API_KEY
}

# -----------------------------
# Step 1: Get Metadata for Target Locations
# -----------------------------
print(f"Fetching metadata for {len(TARGET_LOCATION_IDS)} target locations...")

sensor_meta = {}
for loc_id in TARGET_LOCATION_IDS:
    url_loc = f"{BASE_URL}/locations/{loc_id}"
    resp = requests.get(url_loc, headers=HEADERS)
    
    if resp.status_code == 200:
        data = resp.json().get('results', [])
        if data:
            loc = data[0]
            loc_sensors = loc.get('sensors', [])
            
            # Map parameter names to their specific V3 sensor IDs
            param_map = {}
            for s in loc_sensors:
                param_name = s['parameter']['name'].lower()
                param_map[param_name] = s['id']

            sensor_meta[loc_id] = {
                "name": loc.get('name'),
                "lat": loc['coordinates']['latitude'],
                "lon": loc['coordinates']['longitude'],
                "sensors": param_map
            }
            print(f" ✅ Found Location {loc_id}: {loc.get('name')} | Params: {list(param_map.keys())}")
    else:
        print(f" Failed to fetch metadata for Location {loc_id}: {resp.status_code} {resp.text}")

# Save the sitelist
os.makedirs(path, exist_ok=True)
if sensor_meta:
    key_df = pd.DataFrame.from_dict(sensor_meta, orient='index')
    key_df.index.name = 'locationId'
    key_df.to_csv(os.path.join(path, 'OpenAQ_sitelist_filtered.csv'), index=True)

# -----------------------------
# Step 2: Fetch historical data
# -----------------------------
def fetch_openaq_history(loc_id, meta):
    print(f"  Starting historical pull for {loc_id} ({meta['name']})...")
    sensor_rows = []
    
    # We only care about pm2.5 and humidity for the EPA correction
    target_params = ['pm25', 'relativehumidity', 'humidity']
    
    for param_name in target_params:
        if param_name not in meta['sensors']:
            continue
            
        sensor_id = meta['sensors'][param_name]
        
        current_start = START_DATE
        final_end_date = datetime.utcnow()

        while current_start < final_end_date:
            current_end = current_start + timedelta(days=CHUNK_DAYS)
            if current_end > final_end_date:
                current_end = final_end_date

            page = 1
            while True:
                params = {
                    "datetime_from": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "datetime_to": current_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 1000,  # Fixed: OpenAQ V3 maximum limit is 1000
                    "page": page
                }

                url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
                
                try:
                    r = requests.get(url, params=params, headers=HEADERS)
                    if r.status_code == 200:
                        results = r.json().get('results', [])
                        if results:
                            for d in results:
                                sensor_rows.append({
                                    "locationId": loc_id,
                                    "timestamp": d['period']['datetimeFrom']['utc'],
                                    "parameter": param_name,
                                    "value": d['value'],
                                    "latitude": meta["lat"],
                                    "longitude": meta["lon"]
                                })
                            
                            # If we got exactly 1000 rows, there might be more on the next page
                            if len(results) == 1000:
                                page += 1
                                time.sleep(DELAY_BETWEEN_REQUESTS)
                                continue # Loop again for the next page
                            else:
                                print(f"  Loc {loc_id} ({param_name}): +{len(sensor_rows)} total rows so far (up to {current_end.strftime('%Y-%m-%d')})")
                                break # Done with this time chunk
                        else:
                            break # No data returned, move to next chunk
                    elif r.status_code == 429:
                        print(f"  Rate limited! Pausing for 5 seconds...")
                        time.sleep(5)
                        continue # Retry the exact same request
                    else:
                        print(f"  API Error {r.status_code} for sensor {sensor_id}: {r.text}")
                        break # Break out of pagination on error
                        
                except Exception as e:
                    print(f"  Location {loc_id} / Sensor {sensor_id} Exception: {e}")
                    break

            current_start = current_end
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return sensor_rows
# -----------------------------
# Step 3: Run parallel processing
# -----------------------------
all_data = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_openaq_history, loc, meta): loc for loc, meta in sensor_meta.items()}
    for future in as_completed(futures):
        all_data.extend(future.result())

# -----------------------------
# Step 4: Pivot, Apply EPA Correction & Save
# -----------------------------
if all_data:
    df_raw = pd.DataFrame(all_data)
    
    print("\nPivoting dataset to align timestamps...")
    df_pivot = df_raw.pivot_table(
        index=['locationId', 'timestamp', 'latitude', 'longitude'],
        columns='parameter',
        values='value'
    ).reset_index()
    
    # Normalize column names based on what OpenAQ returned
    if 'pm25' in df_pivot.columns:
        df_pivot.rename(columns={'pm25': 'pm25_raw'}, inplace=True)
    if 'relativehumidity' in df_pivot.columns:
        df_pivot.rename(columns={'relativehumidity': 'rhum'}, inplace=True)
    elif 'humidity' in df_pivot.columns:
        df_pivot.rename(columns={'humidity': 'rhum'}, inplace=True)

    # Check if we have the necessary columns for the calculation
    if 'pm25_raw' in df_pivot.columns and 'rhum' in df_pivot.columns:
        df_pivot['pm25_raw'] = pd.to_numeric(df_pivot['pm25_raw'], errors='coerce')
        df_pivot['rhum'] = pd.to_numeric(df_pivot['rhum'], errors='coerce')
        
        # EPA Correction Formula: 0.534 * PM2.5 - 0.0844 * RH + 5.604
        df_pivot['pm25_epa_calculated'] = (0.534 * df_pivot['pm25_raw']) - (0.0844 * df_pivot['rhum']) + 5.604
        
        # Clean up negative values that can result from the formula at very low concentrations
        df_pivot.loc[df_pivot['pm25_epa_calculated'] < 0, 'pm25_epa_calculated'] = 0
    else:
        print("Note: Could not calculate EPA PM2.5 (Missing PM2.5 or Humidity data in the pulled dataset).")

    # Save to CSV
    csv_path = os.path.join(path, "openaq_history_epa_corrected.csv")
    df_pivot.to_csv(csv_path, index=False)
    print(f"\nDone! Saved {len(df_pivot)} timestamped rows total to {csv_path}.")
else:
    print("\nNo historical data found for these locations in the specified timeframe.")