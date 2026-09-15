import pandas as pd
from pathlib import Path
csv_path = Path('measurements/20260506-152956_bike-pusher/detections.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print('Columns:', df.columns.tolist())
    if 'cam_X' in df.columns:
        print('First 5 rows of cam_X and cam_Y:')
        print(df[['cam_X', 'cam_Y']].head())
        print('Number of NaNs:', df['cam_X'].isna().sum())
    else:
        print('cam_X is MISSING!')
else:
    print('detections.csv does not exist for this folder!')