from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuración
# =========================
@dataclass
class Config:
    # Simulación
    start_date: str = "2025-01-01"     # YYYY-MM-DD
    days: int = 14                     # días a simular
    interval_minutes: int = 10         # contador real: 10-min
    seed: int = 42

    # Vivienda / consumo
    occupants: int = 4
    weekend_factor: float = 1.15

    # Caudal base (litros/hora) -> se convierte a L/min internamente
    base_leak_lph: float = 0.2         # fuga mínima/base

    # Probabilidades (aprox) a nivel HORA; se reescalan a 10-min (÷6)
    shower_prob_per_hour: float = 0.08
    tap_use_prob_per_hour: float = 0.25
    washing_machine_prob_per_day: float = 0.35
    dishwasher_prob_per_day: float = 0.30

    # Detección nocturna y fugas
    night_start: int = 0               # 00:00
    night_end: int = 5                 # 05:00 (incluye 05:xx)
    night_use_threshold_lpm: float = 0.2

    leak_threshold_lpm: float = 0.3    # fuga si flow_lpm > umbral sostenido
    leak_min_intervals: int = 12       # 12×10min = 2 horas

    # Salidas
    out_dir: str = "data_out"


# =========================
# Utilidades
# =========================
def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=sábado, 6=domingo


def _hour_profile_multiplier(hour: int) -> float:
    """
    Perfil diario simple:
    - valle madrugada
    - pico mañana (6-9)
    - valle medio día
    - pico tarde-noche (18-22)
    """
    if 0 <= hour <= 5:
        return 0.55
    if 6 <= hour <= 9:
        return 1.35
    if 10 <= hour <= 17:
        return 0.85
    if 18 <= hour <= 22:
        return 1.40
    return 0.95


def _simulate_interval(dt: datetime, cfg: Config, daily_flags: dict) -> dict:
    """
    Simula un intervalo de (cfg.interval_minutes) minutos.
    Devuelve un dict con flow_lpm, liters y eventos.
    """
    hour = dt.hour
    profile_mult = _hour_profile_multiplier(hour)
    weekend_mult = cfg.weekend_factor if _is_weekend(dt) else 1.0

    # Base: fuga mínima (convertir L/h a L/min) + pequeña aleatoriedad
    base_lpm = cfg.base_leak_lph / 60.0
    flow_lpm = random.uniform(max(0.01, 0.6 * base_lpm), 1.4 * base_lpm)

    # Ajuste por ocupantes (leve)
    occ_mult = 0.9 + 0.1 * (cfg.occupants / 4)

    # Probabilidades escaladas a intervalo 10-min (6 intervalos por hora)
    shower_event = random.random() < (cfg.shower_prob_per_hour / 6) * profile_mult
    tap_event = random.random() < (cfg.tap_use_prob_per_hour / 6) * profile_mult

    if shower_event:
        # Ducha real ~6-12 L/min durante el intervalo
        flow_lpm += random.uniform(6, 12) * occ_mult

    if tap_event:
        # Grifo ~1-4 L/min durante el intervalo
        flow_lpm += random.uniform(1, 4) * occ_mult

    # Lavadora / lavavajillas: como eventos diarios (probabilidad distribuida en 144 intervalos/día)
    wm_event = False
    if (not daily_flags["washing_machine_done"]) and (random.random() < cfg.washing_machine_prob_per_day / 144):
        wm_event = True
        daily_flags["washing_machine_done"] = True
        flow_lpm += random.uniform(8, 15)

    dw_event = False
    if (not daily_flags["dishwasher_done"]) and (random.random() < cfg.dishwasher_prob_per_day / 144):
        dw_event = True
        daily_flags["dishwasher_done"] = True
        flow_lpm += random.uniform(5, 10)

    # Aplica perfil + fin de semana
    flow_lpm *= weekend_mult
    # Perfil multiplicativo adicional (suaviza picos)
    flow_lpm *= (0.85 + 0.15 * profile_mult)

    liters_interval = flow_lpm * cfg.interval_minutes

    return {
        "timestamp": dt,
        "flow_lpm": round(flow_lpm, 3),
        "liters": round(liters_interval, 3),
        "shower": shower_event,
        "tap_use": tap_event,
        "washing_machine": wm_event,
        "dishwasher": dw_event,
        "weekend": _is_weekend(dt),
        "profile_mult": round(profile_mult, 3),
        "weekend_mult": round(weekend_mult, 3),
    }


# =========================
# Generación de datos
# =========================
def generate_timeseries(cfg: Config) -> pd.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    start = datetime.fromisoformat(cfg.start_date)
    step = timedelta(minutes=cfg.interval_minutes)
    total_steps = int((cfg.days * 24 * 60) / cfg.interval_minutes)

    rows = []
    current_day = start.date()
    daily_flags = {"washing_machine_done": False, "dishwasher_done": False}

    dt = start
    for _ in range(total_steps):
        if dt.date() != current_day:
            current_day = dt.date()
            daily_flags = {"washing_machine_done": False, "dishwasher_done": False}

        rows.append(_simulate_interval(dt, cfg, daily_flags))
        dt += step

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    return df


# =========================
# Detección de anomalías
# =========================
def detect_anomalies(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    # 1) Consumo nocturno (sospechoso)
    df["night_use"] = (
        (df["hour"] >= cfg.night_start)
        & (df["hour"] <= cfg.night_end)
        & (df["flow_lpm"] > cfg.night_use_threshold_lpm)
    )

    # 2) Fuga: flow_lpm > umbral de forma sostenida (racha)
    df["leak_flag"] = (df["flow_lpm"] > cfg.leak_threshold_lpm).astype(int)

    # Rachas consecutivas de leak_flag==1
    streak = []
    count = 0
    for v in df["leak_flag"].to_numpy():
        if v == 1:
            count += 1
        else:
            count = 0
        streak.append(count)

    df["leak_streak"] = streak
    df["leak_detected"] = df["leak_streak"] >= cfg.leak_min_intervals

    return df


# =========================
# Guardar CSV
# =========================
def save_outputs(df: pd.DataFrame, cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    serie_csv = out / f"consumo_agua_{cfg.start_date}_dias{cfg.days}_{cfg.interval_minutes}min.csv"
    df.to_csv(serie_csv, index=False)

    daily = (
        df.groupby("date", as_index=False)["liters"]
        .sum()
        .rename(columns={"liters": "liters_day"})
    )
    daily_csv = out / f"consumo_agua_diario_{cfg.start_date}_dias{cfg.days}.csv"
    daily.to_csv(daily_csv, index=False)

    anomalies_csv = out / "consumo_con_anomalias.csv"
    df.to_csv(anomalies_csv, index=False)

    return {"serie_csv": serie_csv, "daily_csv": daily_csv, "anomalies_csv": anomalies_csv}


# =========================
# Gráficas
# =========================
def plot_outputs(df: pd.DataFrame, cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Serie temporal: litros por intervalo
    plt.figure()
    plt.plot(df["timestamp"], df["liters"])
    plt.title("Consumo de agua por intervalo (10-min)")
    plt.xlabel("Tiempo")
    plt.ylabel("Litros por intervalo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p1 = out / "serie_temporal_consumo_10min.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    # 2) Consumo diario
    daily = df.groupby("date")["liters"].sum()
    plt.figure()
    plt.plot(daily.index.astype(str), daily.values)
    plt.title("Consumo diario de agua")
    plt.xlabel("Fecha")
    plt.ylabel("Litros/día")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p2 = out / "consumo_diario.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    # 3) Perfil horario promedio (litros/intervalo promedio por hora)
    hourly = df.groupby("hour")["liters"].mean()
    plt.figure()
    plt.plot(hourly.index, hourly.values)
    plt.title("Perfil horario promedio (litros por 10-min)")
    plt.xlabel("Hora del día")
    plt.ylabel("Litros promedio por intervalo")
    plt.xticks(range(0, 24, 1))
    plt.tight_layout()
    p3 = out / "perfil_horario_promedio.png"
    plt.savefig(p3, dpi=200)
    plt.close()

    # 4) Marcado de posibles fugas: % de intervalos con leak_flag por día
    leak_daily = df.groupby("date")["leak_flag"].mean() * 100
    plt.figure()
    plt.plot(leak_daily.index.astype(str), leak_daily.values)
    plt.title("Porcentaje de intervalos con caudal > umbral (posible fuga)")
    plt.xlabel("Fecha")
    plt.ylabel("% intervalos (leak_flag)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p4 = out / "indicador_fuga_por_dia.png"
    plt.savefig(p4, dpi=200)
    plt.close()

    return {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


# =========================
# Main
# =========================
def main():
    cfg = Config(
        start_date="2025-01-01",
        days=14,
        interval_minutes=10,
        occupants=4,
        seed=42,
    )

    df = generate_timeseries(cfg)
    df = detect_anomalies(df, cfg)

    outputs = save_outputs(df, cfg)
    plots = plot_outputs(df, cfg)

    print("✅ Generado correctamente:")
    print(f"- CSV serie:      {outputs['serie_csv']}")
    print(f"- CSV diario:     {outputs['daily_csv']}")
    print(f"- CSV anomalías:  {outputs['anomalies_csv']}")
    print(f"- Gráficas:       {Path(cfg.out_dir).resolve()}")
    for k, v in plots.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()