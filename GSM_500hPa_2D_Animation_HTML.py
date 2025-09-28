import requests
import xarray as xr
import numpy as np
import cartopy.io.shapereader as shpreader
from scipy.interpolate import griddata
from shapely.geometry import LineString, MultiLineString, Polygon
import os
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- STEP 1: アニメーション設定 ---
print("STEP 1: Setting up animation parameters...")
JST = timezone(timedelta(hours=9))

def get_latest_gfs_run():
    now_utc = datetime.now(timezone.utc)
    latest_run_time = now_utc - timedelta(hours=7)
    run_hour = (latest_run_time.hour // 6) * 6
    run_date = latest_run_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    return run_date

BASE_DATE = get_latest_gfs_run()
print(f"Latest available GFS run detected: {BASE_DATE.strftime('%Y-%m-%d %H:%M')} UTC")

# ★★★ 修正点: アニメーション用の9フレームに戻す ★★★
KEYFRAME_HOURS = [-48, -36, -24, -12, 0, 12, 24, 36, 48]
DOW_MAP = {'Mon': '月', 'Tue': '火', 'Wed': '水', 'Thu': '木', 'Fri': '金', 'Sat': '土', 'Sun': '日'}

# --- STEP 2: キーフレームの気象データ取得と処理 ---
print("\nSTEP 2: Downloading and processing keyframe data...")
keyframe_data_list_500hpa = []
keyframe_data_list_850t = []
keyframe_data_list_mslp = []

plot_x_min, plot_x_max = 60, 180
plot_y_min, plot_y_max = 0, 70
grid_x, grid_y = np.mgrid[plot_x_min:plot_x_max:121j, plot_y_min:plot_y_max:71j]

for hour in KEYFRAME_HOURS:
    run_date = BASE_DATE
    forecast_hour = hour
    if hour < 0:
        run_date = BASE_DATE + timedelta(hours=hour)
        forecast_hour = 0
    
    date_str = run_date.strftime("%Y%m%d")
    run_hour_str = f"{run_date.hour:02d}"
    f_hour_str = f"{forecast_hour:03d}"
    
    url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_str}/{run_hour_str}/atmos/gfs.t{run_hour_str}z.pgrb2.0p25.f{f_hour_str}"
    file_name = "temp_gfs_data.grib2"
    
    print(f"  Downloading keyframe for T{hour:+}h...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"  Download successful. Processing {file_name}...")

        ds_500hpa = xr.open_dataset(file_name, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500}}, decode_times=False)
        gh = ds_500hpa['gh']
        grid_z_500hpa = griddata(np.column_stack((np.meshgrid(gh.longitude.values, gh.latitude.values)[0].ravel(), np.meshgrid(gh.longitude.values, gh.latitude.values)[1].ravel())), gh.values.ravel(), (grid_x, grid_y), method='cubic')
        keyframe_data_list_500hpa.append(grid_z_500hpa)
        
        ds_850t = xr.open_dataset(file_name, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 850}}, decode_times=False)
        temp_k = ds_850t['t']
        temp_c = temp_k - 273.15
        grid_z_850t = griddata(np.column_stack((np.meshgrid(temp_c.longitude.values, temp_c.latitude.values)[0].ravel(), np.meshgrid(temp_c.longitude.values, temp_c.latitude.values)[1].ravel())), temp_c.values.ravel(), (grid_x, grid_y), method='cubic')
        keyframe_data_list_850t.append(grid_z_850t)
        
        ds_mslp = xr.open_dataset(file_name, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea'}}, decode_times=False)
        mslp = ds_mslp['prmsl'] / 100
        grid_z_mslp = griddata(np.column_stack((np.meshgrid(mslp.longitude.values, mslp.latitude.values)[0].ravel(), np.meshgrid(mslp.longitude.values, mslp.latitude.values)[1].ravel())), mslp.values.ravel(), (grid_x, grid_y), method='cubic')
        keyframe_data_list_mslp.append(grid_z_mslp)
        
        print(f"  Processing for T{hour:+}h complete.")

    except Exception as e:
        print(f"  A critical error occurred: {e}")

    finally:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"  Removed temporary file: {file_name}")

if not all(len(lst) >= 2 for lst in [keyframe_data_list_500hpa, keyframe_data_list_850t, keyframe_data_list_mslp]):
    print("\nError: Not enough data to create animation. Exiting.")
    exit()

# --- STEP 3: Plotly用フレームの生成 ---
print("\nSTEP 3: Generating frames for Plotly animation...")
frames = []
for k in range(len(KEYFRAME_HOURS)):
    frame_time_utc = BASE_DATE + timedelta(hours=KEYFRAME_HOURS[k])
    time_label = frame_time_utc.astimezone(JST).strftime(f'%m/%d({DOW_MAP.get(frame_time_utc.astimezone(JST).strftime("%a"), "")}) %H:%M JST')
    
    frame_data = [
        go.Contour(z=keyframe_data_list_500hpa[k].T),
        go.Contour(z=keyframe_data_list_850t[k].T),
        go.Contour(z=keyframe_data_list_mslp[k].T)
    ]
    frames.append(go.Frame(data=frame_data, name=time_label, traces=[0, 1, 2])) 

# --- STEP 4: Plotlyフィギュアの作成とレイアウト設定 ---
print("STEP 4: Creating Plotly figure and layout...")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    # ★★★ 修正点: 各図の名称を追加 ★★★
    subplot_titles=(
        '<b>500hPa 高度分布図 (m)</b>',
        '<b>850hPa 気温分布図 (℃)</b>',
        '<b>地上気圧分布図 (hPa)</b>'
    )
)

# 地上気圧用のカスタムカラースケール
zmin_mslp = 980
zmax_mslp = 1030
mid_point_hpa = 1012
mid_point_norm = (mid_point_hpa - zmin_mslp) / (zmax_mslp - zmin_mslp)

custom_colorscale_mslp = [
    [0.0, 'darkblue'],
    [mid_point_norm, 'white'],
    [1.0, 'gold']
]

# 1. 500hPa 高度
fig.add_trace(go.Contour(x=grid_x[:,0], y=grid_y[0,:], z=keyframe_data_list_500hpa[0].T, colorscale='Plasma', zmin=5300, zmax=5900, colorbar=dict(title='m', x=1.01, y=0.83, len=0.3), contours=dict(start=5100, end=6000, size=60, showlabels=True, labelfont=dict(color='black')), hovertemplate='高度: %{z:.0f}m<extra></extra>'), row=1, col=1)

# 2. 850hPa 気温
fig.add_trace(go.Contour(x=grid_x[:,0], y=grid_y[0,:], z=keyframe_data_list_850t[0].T, colorscale='RdBu_r', zmin=-21, zmax=30, colorbar=dict(title='℃', x=1.01, y=0.5, len=0.3), contours=dict(start=-30, end=30, size=3, showlabels=True, labelfont=dict(color='black')), hovertemplate='気温: %{z:.1f}℃<extra></extra>'), row=2, col=1)

# 3. 地上気圧
fig.add_trace(go.Contour(x=grid_x[:,0], y=grid_y[0,:], z=keyframe_data_list_mslp[0].T, colorscale=custom_colorscale_mslp, zmin=zmin_mslp, zmax=zmax_mslp, colorbar=dict(title='hPa', x=1.01, y=0.17, len=0.3), contours=dict(start=960, end=1040, size=4, showlabels=True, labelfont=dict(color='black')), hovertemplate='気圧: %{z:.1f}hPa<extra></extra>'), row=3, col=1)

# ★★★ 修正点: アニメーション用にfig.framesを適用 ★★★
fig.frames = frames

print("STEP 4b: Adding coastlines...")
coastlines_shp = shpreader.natural_earth(resolution='50m', category='physical', name='coastline')
domain_poly = Polygon([(plot_x_min, plot_y_min), (plot_x_max, plot_y_min), (plot_x_max, plot_y_max), (plot_x_min, plot_y_max)])
for geometry in shpreader.Reader(coastlines_shp).geometries():
    clipped_geom = domain_poly.intersection(geometry)
    if not clipped_geom.is_empty:
        geoms = [clipped_geom] if clipped_geom.geom_type == 'LineString' else clipped_geom.geoms
        for line in geoms:
            lon, lat = line.xy
            for r in range(1, 4):
                fig.add_trace(go.Scatter(x=list(lon), y=list(lat), mode='lines', line=dict(color='black', width=1.5), showlegend=False), row=r, col=1)

slider_steps = [{'label': frame.name, 'method': 'animate', 'args': [[frame.name], {'mode': 'immediate'}]} for frame in fig.frames]

fig.update_layout(
    title_text=f'<b>GFS気象予測アニメーション</b><br>Base Time: {BASE_DATE.strftime("%Y-%m-%d %H:%M")} UTC',
    height=1600,
    **{f'yaxis{i}': dict(range=[plot_y_min, plot_y_max], showticklabels=False, title='') for i in range(1, 4)},
    **{f'xaxis{i}': dict(range=[plot_x_min, plot_x_max]) for i in range(1, 4)},
    xaxis3_title='<b>経度</b>',
    **{f'yaxis{i}_scaleanchor': f'x{i}' for i in range(1, 4)},
    **{f'yaxis{i}_scaleratio': 1 for i in range(1, 4)},
    # ★★★ 修正点: アニメーションUIを再追加 ★★★
    updatemenus=[
        {'type': 'buttons', 'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': False}]}, {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}], 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'},
        {'type': 'dropdown', 'buttons': [{'label': f'{s}x', 'method': 'animate', 'args': [None, {'frame': {'duration': 500/s, 'redraw': True}}]} for s in [0.5, 0.75, 1.0, 1.5, 2.0]], 'active': 2, 'x': 0.25, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}
    ],
    sliders=[{'steps': slider_steps, 'x': 0.1, 'xanchor': 'left', 'y': 0, 'yanchor': 'top', 'currentvalue': {'prefix': '<b>時刻: </b>', 'visible': True, 'xanchor': 'right'}}]
)

# 各サブプロットの左上に日時を注釈として追加
initial_time_label = fig.frames[0].name
annotations = [
    dict(text=f'<b>{initial_time_label}</b>', align='left', showarrow=False, xref='x domain' if i == 1 else f'x{i} domain', yref='y domain' if i == 1 else f'y{i} domain', x=0.01, y=0.99) for i in range(1, 4)
]
fig.update_layout(annotations=annotations)

# スライダーのステップに注釈更新を追加
for step in slider_steps:
    step['args'][1]['layout'] = {'annotations': [
        dict(text=step['label'], align='left', showarrow=False, xref='x domain' if i == 1 else f'x{i} domain', yref='y domain' if i == 1 else f'y{i} domain', x=0.01, y=0.99) for i in range(1, 4)
    ]}

# --- STEP 5: HTMLファイルとして保存 ---
print("\nSTEP 5: Saving interactive animation to HTML file...")
output_filename = "gfs_final_animation.html"
config = {'responsive': True}
fig.write_html(output_filename, config=config)

print(f"\n🎉 Success! Open '{output_filename}' in your web browser for the final animation.")