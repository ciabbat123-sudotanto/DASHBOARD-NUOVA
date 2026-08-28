import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("Scaricamento dati dalle API Open-Meteo in corso...")

    # 1. DOWNLOAD DATI
    url_ensemble = (
        "https://ensemble-api.open-meteo.com/v1/ensemble?"
        "latitude=45.4643&longitude=9.1895&"
        "hourly=rain,temperature_850hPa&"
        "models=gem_global_ensemble,ecmwf_ifs025_ensemble,ncep_gefs05,google_weathernext2_ensemble"
    )

    url_lam = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=45.4643&longitude=9.1895&"
        "hourly=cape,wind_direction_1000hPa,wind_direction_850hPa,wind_direction_500hPa,"
        "wind_speed_1000hPa,wind_speed_850hPa,wind_speed_500hPa,temperature_2m,rain,wind_gusts_10m&"
        "models=italia_meteo_arpae_icon_2i,meteofrance_arome_france,dwd_icon_d2&"
        "forecast_days=1"
    )

    res_ens = requests.get(url_ensemble).json()
    res_lam = requests.get(url_lam).json()

    # ELABORAZIONE ENSEMBLE
    hourly_ens = res_ens.get("hourly", {})
    df_ens = pd.DataFrame(hourly_ens)
    if "time" in df_ens.columns:
        df_ens["time"] = pd.to_datetime(df_ens["time"])

    temp_850_cols = [c for c in df_ens.columns if c.startswith("temperature_850hPa")]
    rain_cols = [c for c in df_ens.columns if c.startswith("rain")]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Temp 850hPa (Supermedia, Medie Modelli, Sopra/Sotto media)",
            "% Spaghi con Pioggia ≥ 0.4 mm/h",
            "Intensità Media Pioggia Spaghi Attivi (mm/h)"
        )
    )

    model_names = ["gem_global_ensemble", "ecmwf_ifs025_ensemble", "ncep_gefs05", "google_weathernext2_ensemble"]
    model_means_list = []

    for m in model_names:
        m_cols = [c for c in temp_850_cols if m in c]
        if m_cols:
            m_df = df_ens[m_cols]
            m_mean = m_df.mean(axis=1)
            model_means_list.append(m_mean)

            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_mean, mode='lines',
                line=dict(color='#64748b', width=1.5)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_df.quantile(0.9, axis=1), mode='lines',
                line=dict(color='#dc2626', width=1.2, dash='dot')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_df.quantile(0.1, axis=1), mode='lines',
                line=dict(color='#2563eb', width=1.2, dash='dash')
            ), row=1, col=1)

    if model_means_list:
        supermedia = pd.concat(model_means_list, axis=1).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=df_ens['time'], y=supermedia, mode='lines',
            line=dict(color='#000000', width=3.5)
        ), row=1, col=1)

    # PIOGGIA ENSEMBLE
    rain_data = df_ens[rain_cols]
    total_spaghi = rain_data.shape[1]

    if total_spaghi > 0:
        spaghi_active = (rain_data >= 0.4)
        pct_active = (spaghi_active.sum(axis=1) / total_spaghi) * 100.0
        mean_active_rain = rain_data[spaghi_active].mean(axis=1).fillna(0)
    else:
        pct_active = pd.Series(0, index=df_ens.index)
        mean_active_rain = pd.Series(0, index=df_ens.index)

    fig.add_trace(go.Scatter(
        x=df_ens['time'], y=pct_active, mode='lines', fill='tozeroy',
        line=dict(color='#0284c7', width=2), fillcolor='rgba(2, 132, 199, 0.15)'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_ens['time'], y=mean_active_rain, mode='lines', fill='tozeroy',
        line=dict(color='#9333ea', width=2), fillcolor='rgba(147, 51, 234, 0.15)'
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="#1e293b", family="Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=900,
        width=800,
        showlegend=False
    )

    fig.write_image("grafico.png", scale=2)

    # 2. ELABORAZIONE SCHEDE LAM
    hourly_lam = res_lam.get("hourly", {})
    df_lam = pd.DataFrame(hourly_lam)
    if "time" in df_lam.columns:
        df_lam["time"] = pd.to_datetime(df_lam["time"])

    lam_models = ["italia_meteo_arpae_icon_2i", "meteofrance_arome_france", "dwd_icon_d2"]

    # TEMPERATURA 2M
    temp_cols = [f"temperature_2m_{m}" for m in lam_models if f"temperature_2m_{m}" in df_lam.columns]
    if temp_cols:
        temp_data = df_lam[temp_cols]
        temp_mean_daily = round(temp_data.values.mean(), 1)
        temp_max_avg = round(temp_data.max(axis=0).mean(), 1)
        temp_min_avg = round(temp_data.min(axis=0).mean(), 1)
    else:
        temp_mean_daily, temp_max_avg, temp_min_avg = 0.0, 0.0, 0.0

    # PRECIPITAZIONI LAM
    rain_lam_cols = [f"rain_{m}" for m in lam_models if f"rain_{m}" in df_lam.columns]
    total_lam = len(rain_lam_cols)

    if total_lam > 0:
        rain_lam_df = df_lam[rain_lam_cols]
        
        # 1. Rischio Pioggia 24h (% modelli con picco >= 0.4 mm/h)
        models_with_peak = sum((rain_lam_df[c] >= 0.4).any() for c in rain_lam_cols)
        risk_rain_pct = int((models_with_peak / total_lam) * 100)

        # 2. Accumulo Totale 24h: Media dell'accumulo totale giornaliero dei modelli
        total_acc_mean = round(rain_lam_df.sum(axis=0).mean(), 1)

        # 3. Finestra Temporale: Considera tutte le ore in cui ALMENO UN MODELLO vede pioggia >= 0.1 mm/h
        active_hours_mask = (rain_lam_df >= 0.1).any(axis=1)
        active_times = df_lam.loc[active_hours_mask, 'time']

        if not active_times.empty:
            first_hour = active_times.iloc[0].strftime('%H:%M')
            last_hour = active_times.iloc[-1].strftime('%H:%M')
            rain_window = f"{first_hour} - {last_hour}"
        else:
            rain_window = "Nessuna precipitazione"
    else:
        risk_rain_pct = 0
        total_acc_mean = 0.0
        rain_window = "N/A"

    # CALCOLO SHEAR VENTO
    def calc_shear(speed1, dir1, speed2, dir2):
        rad1, rad2 = np.radians(dir1), np.radians(dir2)
        u1, v1 = speed1 * np.sin(rad1), speed1 * np.cos(rad1)
        u2, v2 = speed2 * np.sin(rad2), speed2 * np.cos(rad2)
        return np.sqrt((u2 - u1)**2 + (v2 - v1)**2)

    # INDICE TURBOLOSO ORARIO & PICCHI MAX
    hourly_turb_list = []
    max_rain_peak_val, max_rain_peak_time = 0.0, "N/A"
    max_gust_val, max_gust_time = 0.0, "N/A"

    for m in lam_models:
        r_col = f"rain_{m}"
        g_col = f"wind_gusts_10m_{m}"

        if r_col in df_lam.columns:
            r_arr = df_lam[r_col].values
            max_idx = np.argmax(r_arr)
            if r_arr[max_idx] > max_rain_peak_val:
                max_rain_peak_val = r_arr[max_idx]
                max_rain_peak_time = df_lam['time'].iloc[max_idx].strftime('%H:%M')

        if g_col in df_lam.columns:
            g_arr = df_lam[g_col].values
            max_idx = np.argmax(g_arr)
            if g_arr[max_idx] > max_gust_val:
                max_gust_val = g_arr[max_idx]
                max_gust_time = df_lam['time'].iloc[max_idx].strftime('%H:%M')

        cape_col = f"cape_{m}"
        s1000, d1000 = f"wind_speed_1000hPa_{m}", f"wind_direction_1000hPa_{m}"
        s850, d850 = f"wind_speed_850hPa_{m}", f"wind_direction_850hPa_{m}"
        s500, d500 = f"wind_speed_500hPa_{m}", f"wind_direction_500hPa_{m}"

        if cape_col in df_lam.columns and s1000 in df_lam.columns:
            cape_hourly = df_lam[cape_col].values
            sh_1000_850 = calc_shear(df_lam[s1000], df_lam[d1000], df_lam[s850], df_lam[d850])
            sh_850_500 = calc_shear(df_lam[s850], df_lam[d850], df_lam[s500], df_lam[d500])
            shear_tot_hourly = (sh_1000_850 + sh_850_500).values
            rain_hourly = df_lam[r_col].values
            
            # Calcolo orario dell'indice turboloso
            idx_hourly = (cape_hourly * shear_tot_hourly * rain_hourly) / 100.0
            hourly_turb_list.append(idx_hourly)

    if hourly_turb_list:
        # Media tra i modelli per ciascuna ora, poi si prende il PICCO MASSIMO ORARIO
        hourly_turb_means = np.mean(np.array(hourly_turb_list), axis=0)
        final_turb_index = round(float(np.max(hourly_turb_means)), 1)
    else:
        final_turb_index = 0.0

    # SOGLIE ALLERTA
    if final_turb_index >= 20:
        level_code, level_label, level_color = 3, "Livello 3 (Rosso)", "#dc2626"
        alert_msg = "ATTENZIONE TEMPORALI FORTI!"
    elif final_turb_index >= 10:
        level_code, level_label, level_color = 2, "Livello 2 (Arancione)", "#ea580c"
        alert_msg = "Previsti temporali moderati!"
    elif final_turb_index >= 5:
        level_code, level_label, level_color = 1, "Livello 1 (Giallo)", "#d97706"
        alert_msg = "Previsti temporali!"
    else:
        level_code, level_label, level_color = 0, "Livello 0 (Verde)", "#16a34a"
        alert_msg = ""

    banner_html = f"""
    <div class="alert-banner alert-lvl-{level_code}">
        <div class="alert-title">{alert_msg}</div>
        <div class="alert-details">
            <span><b>Pioggia max:</b> {max_rain_peak_val:.1f} mm/h (ore {max_rain_peak_time})</span>
            <span><b>Raffica max:</b> {max_gust_val:.1f} km/h (ore {max_gust_time})</span>
        </div>
    </div>
    """ if level_code >= 1 else ""

    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsioni Meteo Ciabatta</title>
    <style>
        :root {{
            --bg-color: #ffffff;
            --card-bg: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-red: #dc2626;
            --accent-blue: #2563eb;
            --border-color: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 900px;
            margin: 0 auto;
        }}
        header {{ text-align: center; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); }}
        header h1 {{ font-size: 1.6rem; font-weight: 700; color: #0f172a; }}

        .plots-container {{
            background: #ffffff; border-radius: 12px; padding: 8px;
            border: 1px solid var(--border-color); text-align: center;
        }}
        .plots-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }}

        .cards-container {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
        .card {{
            background: var(--card-bg); border-radius: 12px; padding: 14px;
            border: 1px solid var(--border-color); display: flex; flex-direction: column; justify-content: space-between;
        }}
        .card-title {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }}
        .card-main-val {{ font-size: 1.6rem; font-weight: 800; margin: 4px 0; }}
        .card-subtext {{ font-size: 0.75rem; color: var(--text-muted); }}
        .temp-sub {{ display: flex; gap: 8px; font-size: 0.85rem; font-weight: 600; }}
        .temp-max {{ color: var(--accent-red); }}
        .temp-min {{ color: var(--accent-blue); }}

        .alert-banner {{
            border-radius: 12px; padding: 14px; color: #ffffff;
            display: flex; flex-direction: column; gap: 6px;
        }}
        .alert-lvl-1 {{ background: #d97706; }}
        .alert-lvl-2 {{ background: #ea580c; }}
        .alert-lvl-3 {{ background: #dc2626; }}
        .alert-title {{ font-size: 1.1rem; font-weight: 800; }}
        .alert-details {{ font-size: 0.85rem; display: flex; flex-direction: column; gap: 2px; }}

        footer {{ text-align: center; font-size: 0.75rem; color: var(--text-muted); margin-top: 12px; }}
    </style>
</head>
<body>
    <header><h1>Previsioni Meteo Ciabatta</h1></header>
    
    <div class="cards-container">
        <div class="card">
            <div class="card-title">Temperatura 24h</div>
            <div class="card-main-val">{temp_mean_daily}°C</div>
            <div class="temp-sub">
                <span class="temp-max">Max: {temp_max_avg}°C</span>
                <span class="temp-min">Min: {temp_min_avg}°C</span>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Rischio Pioggia 24h</div>
            <div class="card-main-val" style="color: #0284c7;">{risk_rain_pct}%</div>
            <div class="card-subtext">Modelli LAM ≥ 0.4 mm/h</div>
        </div>
        <div class="card">
            <div class="card-title">Accumulo Totale 24h</div>
            <div class="card-main-val" style="color: #9333ea;">{total_acc_mean} <span style="font-size: 0.9rem;">mm</span></div>
            <div class="card-subtext">Finestra: <b>{rain_window}</b></div>
        </div>
        <div class="card">
            <div class="card-title">Indice Turboloso</div>
            <div class="card-main-val" style="color: {level_color};">{final_turb_index:.1f}</div>
            <div class="card-subtext">Stato: <b style="color: {level_color};">{level_label}</b></div>
        </div>
    </div>

    {banner_html}

    <div class="plots-container">
        <img src="grafico.png" alt="Grafici Meteo Ensemble">
    </div>

    <footer>Dati forniti da Open-Meteo API</footer>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Immagine grafico.png e file index.html generati con successo.")

if __name__ == "__main__":
    main()
