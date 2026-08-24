import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    print("Scaricamento dati dalle API Open-Meteo in corso...")

    # ----------------------------------------------------
    # 1. DOWNLOAD DATI DA OPEN-METEO API
    # ----------------------------------------------------
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

    # --- ELABORAZIONE DATI ENSEMBLE ---
    hourly_ens = res_ens.get("hourly", {})
    df_ens = pd.DataFrame(hourly_ens)
    if "time" in df_ens.columns:
        df_ens["time"] = pd.to_datetime(df_ens["time"])

    temp_850_cols = [c for c in df_ens.columns if c.startswith("temperature_850hPa")]
    rain_cols = [c for c in df_ens.columns if c.startswith("rain")]

    # Layout Subplot Plotly
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        subplot_titles=(
            "Temperatura 850hPa Ensemble (Supermedia, Medie Modelli, Sopramedia e Sottomedia)",
            "% Spaghi con Pioggia ≥ 0.4 mm/h",
            "Intensità Media Pioggia tra Spaghi Attivi (≥ 0.4 mm/h)"
        )
    )

    # ----------------------------------------------------
    # GRAFICO ENSEMBLE 850HPA
    # ----------------------------------------------------
    model_names = ["gem_global_ensemble", "ecmwf_ifs025_ensemble", "ncep_gefs05", "google_weathernext2_ensemble"]

    model_means_list = []
    p90_list = []
    p10_list = []

    for m in model_names:
        m_cols = [c for c in temp_850_cols if m in c]
        if m_cols:
            m_df = df_ens[m_cols]
            m_mean = m_df.mean(axis=1)
            model_means_list.append(m_mean)
            p90_list.append(m_df.quantile(0.9, axis=1))
            p10_list.append(m_df.quantile(0.1, axis=1))

            # MEDIA DI OGNI MODELLO SINGOLO (IN GRIGIO)
            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_mean,
                mode='lines',
                name=f'Media {m}',
                line=dict(color='#64748b', width=1.5),
                hoverinfo='x+y+name'
            ), row=1, col=1)

            # SOPRAMEDIA DI OGNI MODELLO (IN ROSSO)
            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_df.quantile(0.9, axis=1),
                mode='lines',
                name=f'Sopramedia (P90) {m}',
                line=dict(color='#dc2626', width=1.2, dash='dot'),
                hoverinfo='x+y+name'
            ), row=1, col=1)

            # SOTTOMEDIA DI OGNI MODELLO (IN BLU)
            fig.add_trace(go.Scatter(
                x=df_ens['time'], y=m_df.quantile(0.1, axis=1),
                mode='lines',
                name=f'Sottomedia (P10) {m}',
                line=dict(color='#2563eb', width=1.2, dash='dash'),
                hoverinfo='x+y+name'
            ), row=1, col=1)

    # SUPERMEDIA (MEDIA DELLE MEDIE IN NERO PIÙ SPESSA)
    if model_means_list:
        supermedia = pd.concat(model_means_list, axis=1).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=df_ens['time'], y=supermedia,
            mode='lines',
            name='SUPERMEDIA (Media delle Medie)',
            line=dict(color='#000000', width=4.0),
            hoverinfo='x+y+name'
        ), row=1, col=1)

    # ----------------------------------------------------
    # SOTTO-GRAFICI PIOGGIA ENSEMBLE
    # ----------------------------------------------------
    rain_data = df_ens[rain_cols]
    total_spaghi = rain_data.shape[1]

    if total_spaghi > 0:
        spaghi_active = (rain_data >= 0.4)
        pct_active = (spaghi_active.sum(axis=1) / total_spaghi) * 100.0
        mean_active_rain = rain_data[spaghi_active].mean(axis=1).fillna(0)
    else:
        pct_active = pd.Series(0, index=df_ens.index)
        mean_active_rain = pd.Series(0, index=df_ens.index)

    # Sotto-Grafico A: % Spaghi >= 0.4 mm/h
    fig.add_trace(go.Scatter(
        x=df_ens['time'], y=pct_active,
        mode='lines', name='% Spaghi ≥ 0.4 mm/h',
        fill='tozeroy',
        line=dict(color='#0284c7', width=2),
        fillcolor='rgba(2, 132, 199, 0.15)'
    ), row=2, col=1)

    # Sotto-Grafico B: Media mm spaghi attivi
    fig.add_trace(go.Scatter(
        x=df_ens['time'], y=mean_active_rain,
        mode='lines', name='Media mm/h (Spaghi Attivi)',
        fill='tozeroy',
        line=dict(color='#9333ea', width=2),
        fillcolor='rgba(147, 51, 234, 0.15)'
    ), row=2, col=2)

    # Styling Plotly Chiaro / Sfondo Bianco (Senza Legenda)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="#1e293b", family="Segoe UI, sans-serif"),
        margin=dict(l=30, r=30, t=50, b=30),
        height=700,
        showlegend=False
    )

    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # ----------------------------------------------------
    # 2. ELABORAZIONE SCHEDE SINTETICHE LAM (24H) & ALLERTE
    # ----------------------------------------------------
    hourly_lam = res_lam.get("hourly", {})
    df_lam = pd.DataFrame(hourly_lam)
    if "time" in df_lam.columns:
        df_lam["time"] = pd.to_datetime(df_lam["time"])

    lam_models = ["italia_meteo_arpae_icon_2i", "meteofrance_arome_france", "dwd_icon_d2"]

    # A. Temperatura 24h
    temp_cols = [f"temperature_2m_{m}" for m in lam_models if f"temperature_2m_{m}" in df_lam.columns]
    if temp_cols:
        temp_data = df_lam[temp_cols]
        temp_mean_daily = round(temp_data.values.mean(), 1)
        temp_max_avg = round(temp_data.max(axis=0).mean(), 1)
        temp_min_avg = round(temp_data.min(axis=0).mean(), 1)
    else:
        temp_mean_daily, temp_max_avg, temp_min_avg = 0.0, 0.0, 0.0

    # B. Rischio Pioggia 24h
    rain_lam_cols = [f"rain_{m}" for m in lam_models if f"rain_{m}" in df_lam.columns]
    models_with_peak = 0
    total_lam = len(rain_lam_cols)

    if total_lam > 0:
        for c in rain_lam_cols:
            if (df_lam[c] >= 0.4).any():
                models_with_peak += 1
        risk_rain_pct = int((models_with_peak / total_lam) * 100)
    else:
        risk_rain_pct = 0

    # C. Accumulo Totale & Finestra Temporale
    if rain_lam_cols:
        rain_lam_df = df_lam[rain_lam_cols]
        total_acc_mean = round(rain_lam_df.sum(axis=1).mean(), 1)

        avg_hourly_rain = rain_lam_df.mean(axis=1)
        active_hours = df_lam[avg_hourly_rain >= 0.4]['time']

        if not active_hours.empty:
            first_hour = active_hours.iloc[0].strftime('%H:%M')
            last_hour = active_hours.iloc[-1].strftime('%H:%M')
            rain_window = f"{first_hour} - {last_hour}"
        else:
            rain_window = "Nessuna precipitazione"
    else:
        total_acc_mean = 0.0
        rain_window = "N/A"

    # D. Indice Turboloso e Shear
    def calc_shear(speed1, dir1, speed2, dir2):
        rad1, rad2 = np.radians(dir1), np.radians(dir2)
        u1, v1 = speed1 * np.sin(rad1), speed1 * np.cos(rad1)
        u2, v2 = speed2 * np.sin(rad2), speed2 * np.cos(rad2)
        return np.sqrt((u2 - u1)**2 + (v2 - v1)**2)

    turb_indices = []
    max_rain_peak_val = 0.0
    max_rain_peak_time = "N/A"
    max_gust_val = 0.0
    max_gust_time = "N/A"

    for m in lam_models:
        cape_col = f"cape_{m}"
        r_col = f"rain_{m}"
        g_col = f"wind_gusts_10m_{m}"

        s1000, d1000 = f"wind_speed_1000hPa_{m}", f"wind_direction_1000hPa_{m}"
        s850, d850 = f"wind_speed_850hPa_{m}", f"wind_direction_850hPa_{m}"
        s500, d500 = f"wind_speed_500hPa_{m}", f"wind_direction_500hPa_{m}"

        if cape_col in df_lam.columns:
            cape_max = df_lam[cape_col].max()

            sh_1000_850 = calc_shear(df_lam[s1000], df_lam[d1000], df_lam[s850], df_lam[d850])
            sh_850_500 = calc_shear(df_lam[s850], df_lam[d850], df_lam[s500], df_lam[d500])
            shear_tot_max = np.max(sh_1000_850 + sh_850_500)

            water_mass_24h = df_lam[r_col].sum()
            idx = (cape_max * shear_tot_max * water_mass_24h) / 100000.0
            turb_indices.append(idx)

        # Picco Pioggia
        if r_col in df_lam.columns:
            r_arr = df_lam[r_col].values
            max_idx = np.argmax(r_arr)
            if r_arr[max_idx] > max_rain_peak_val:
                max_rain_peak_val = r_arr[max_idx]
                max_rain_peak_time = df_lam['time'].iloc[max_idx].strftime('%H:%M')

        # Picco Raffiche
        if g_col in df_lam.columns:
            g_arr = df_lam[g_col].values
            max_idx = np.argmax(g_arr)
            if g_arr[max_idx] > max_gust_val:
                max_gust_val = g_arr[max_idx]
                max_gust_time = df_lam['time'].iloc[max_idx].strftime('%H:%M')

    final_turb_index = float(np.mean(turb_indices)) if turb_indices else 0.0

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

    # ----------------------------------------------------
    # 3. TEMPLATE HTML COMPLETO CON SFONDO BIANCO
    # ----------------------------------------------------
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
            --accent-green: #16a34a;
            --border-color: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            min-height: 100vh;
        }}
        header {{ text-align: center; padding-bottom: 12px; border-bottom: 1px solid var(--border-color); }}
        header h1 {{ font-size: 2rem; font-weight: 700; color: #0f172a; }}

        .dashboard-grid {{ display: grid; grid-template-columns: 3fr 1fr; gap: 24px; }}
        .plots-container {{
            background: #ffffff; border-radius: 12px; padding: 16px;
            border: 1px solid var(--border-color); box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .cards-container {{ display: flex; flex-direction: column; gap: 16px; }}
        .card {{
            background: var(--card-bg); border-radius: 12px; padding: 20px;
            border: 1px solid var(--border-color); box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            display: flex; flex-direction: column; justify-content: space-between; min-height: 150px;
        }}
        .card-title {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 600; }}
        .card-main-val {{ font-size: 2.2rem; font-weight: 800; margin: 8px 0; }}
        .card-subtext {{ font-size: 0.85rem; color: var(--text-muted); }}
        .temp-sub {{ display: flex; gap: 12px; font-size: 0.95rem; font-weight: 600; }}
        .temp-max {{ color: var(--accent-red); }}
        .temp-min {{ color: var(--accent-blue); }}

        .alert-banner {{
            border-radius: 12px; padding: 20px 24px; display: flex;
            justify-content: space-between; align-items: center; color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .alert-lvl-1 {{ background: linear-gradient(90deg, #d97706, #f59e0b); color: #fff; }}
        .alert-lvl-2 {{ background: linear-gradient(90deg, #ea580c, #f97316); }}
        .alert-lvl-3 {{ background: linear-gradient(90deg, #dc2626, #ef4444); animation: pulse 2s infinite; }}
        .alert-title {{ font-size: 1.3rem; font-weight: 800; }}
        .alert-details {{ display: flex; gap: 24px; font-size: 0.95rem; }}

        footer {{ text-align: center; font-size: 0.8rem; color: var(--text-muted); margin-top: auto; padding-top: 12px; }}
        @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.85; }} 100% {{ opacity: 1; }} }}
        @media (max-width: 1024px) {{
            .dashboard-grid {{ grid-template-columns: 1fr; }}
            .alert-banner {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
        }}
    </style>
</head>
<body>
    <header><h1>Previsioni Meteo Ciabatta</h1></header>
    <div class="dashboard-grid">
        <div class="plots-container">{plotly_html}</div>
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
                <div class="card-subtext">Modelli LAM con picco ≥ 0.4 mm/h</div>
            </div>
            <div class="card">
                <div class="card-title">Accumulo Totale 24h</div>
                <div class="card-main-val" style="color: #9333ea;">{total_acc_mean} <span style="font-size: 1rem;">mm</span></div>
                <div class="card-subtext">Finestra: <b>{rain_window}</b></div>
            </div>
            <div class="card">
                <div class="card-title">Indice Turboloso</div>
                <div class="card-main-val" style="color: {level_color};">{final_turb_index:.1f}</div>
                <div class="card-subtext">Stato: <b style="color: {level_color};">{level_label}</b></div>
            </div>
        </div>
    </div>
    {banner_html}
    <footer>Dati forniti da Open-Meteo API</footer>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("File index.html generato con successo.")

if __name__ == "__main__":
    main()

