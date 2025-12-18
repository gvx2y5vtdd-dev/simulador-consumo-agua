from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    # Simulación
    start_date: str = "2025-01-01"     # YYYY-MM-DD
    days: int = 7                      # días a simular
    interval_minutes: int = 60         # 60 = horario (IoT típico). Puedes usar 10 para 10-min.
    seed: int = 42

    # Vivienda / consumo
    base_leak_lph: float = 0.2         # fuga/base (litros por hora)
    occupants: int = 4                 # ocupantes
    weekend_factor: float = 1.15       # más consumo fin de semana

    # Eventos por hora (aprox, ajustable)
    shower_prob_per_hour: float = 0.08
    tap_use_prob_per_hour: float = 0.25
    washing_machine_prob_per_day: float = 0.35
    dishwasher_prob_per_day: float = 0.30

    # Carpeta de salida
    out_dir: str = "data_out"


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=sábado, 6=domingo


def _hour_profile_multiplier(hour: int) -> float:
    """
    Perfil diario simple:
    - picos mañana (6-9) y noche (19-22)
    - valle madrugada
    """
    if 0 <= hour <= 5:
        return 0.45
    if 6 <= hour <= 9:
        return 1.25
    if 10 <= hour <= 17:
        return 0.85
    if 18 <= hour <= 22:
        return 1.30
    return 0.90


def _simulate_hour(dt: datetime, cfg: Config, daily_flags: dict) -> tuple[float, dict]:
    """
    Devuelve consumo de esta hora (litros) + metadatos de eventos.
    """
    hour = dt.hour
    mult = _hour_profile_multiplier(hour)
    weekend_mult = cfg.weekend_factor if _is_weekend(dt) else 1.0

    # Base (fuga / consumo mínimo)
    liters = cfg.base_leak_lph

    # Ducha: evento con consumo alto
    shower_event = random.random() < (cfg.shower_prob_per_hour * (0.9 if hour < 6 else 1.0))
    if shower_event:
        # 35–80 L por ducha aproximado
        liters += random.uniform(35, 80) * (0.9 + 0.1 * cfg.occupants / 4)

    # Uso grifo: eventos pequeños repetidos
    tap_event = random.random() < cfg.tap_use_prob_per_hour
    if tap_event:
        # 1–12 L en la hora (sumatoria)
        liters += random.uniform(1, 12) * (0.9 + 0.1 * cfg.occupants / 4)

    # Lavadora (una vez al día con probabilidad)
    wm_event = False
    if not daily_flags["washing_machine_done"] and random.random() < (cfg.washing_machine_prob_per_day / 24):
        wm_event = True
        daily_flags["washing_machine_done"] = True
        liters += random.uniform(45, 110)

    # Lavavajillas (una vez al día con probabilidad)
    dw_event = False
    if not daily_flags["dishwasher_done"] and random.random() < (cfg.dishwasher_prob_per_day / 24):
        dw_event = True
        daily_flags["dishwasher_done"] = True
        liters += random.uniform(12, 25)

    # Aplica multiplicadores de perfil y fin de semana
    liters *= mult * weekend_mult

    meta = {
        "shower": shower_event,
        "tap_use": tap_event,
        "washing_machine": wm_event,
        "dishwasher": dw_event,
        "weekend": _is_weekend(dt),
        "profile_mult": mult,
        "weekend_mult": weekend_mult,
    }
    return max(liters, 0.0), meta


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
        # resetea flags al cambiar de día
        if dt.date() != current_day:
            current_day = dt.date()
            daily_flags = {"washing_machine_done": False, "dishwasher_done": False}

        liters, meta = _simulate_hour(dt, cfg, daily_flags)

        rows.append(
            {
                "timestamp": dt.isoformat(timespec="minutes"),
                "liters": round(liters, 3),
                "flow_lph": round(liters * (60 / cfg.interval_minutes), 3),  # l/h equivalente
                **meta,
            }
        )
        dt += step

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df


def save_outputs(df: pd.DataFrame, cfg: Config) -> tuple[Path, Path]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"consumo_agua_{cfg.start_date}_dias{cfg.days}_{cfg.interval_minutes}min.csv"
    df.to_csv(csv_path, index=False)

    daily = (
        df.groupby("date", as_index=False)["liters"]
        .sum()
        .rename(columns={"liters": "liters_day"})
    )
    daily_csv_path = out / f"consumo_agua_diario_{cfg.start_date}_dias{cfg.days}.csv"
    daily.to_csv(daily_csv_path, index=False)

    return csv_path, daily_csv_path


def plot_outputs(df: pd.DataFrame, cfg: Config) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Serie temporal (litros por intervalo)
    plt.figure()
    plt.plot(df["timestamp"], df["liters"])
    plt.title("Consumo de agua por intervalo")
    plt.xlabel("Tiempo")
    plt.ylabel("Litros por intervalo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "serie_temporal_consumo.png", dpi=200)
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
    plt.savefig(out / "consumo_diario.png", dpi=200)
    plt.close()

    # 3) Perfil horario promedio
    hourly = df.groupby("hour")["liters"].mean()
    plt.figure()
    plt.plot(hourly.index, hourly.values)
    plt.title("Perfil horario promedio")
    plt.xlabel("Hora del día")
    plt.ylabel("Litros promedio por intervalo")
    plt.xticks(range(0, 24, 1))
    plt.tight_layout()
    plt.savefig(out / "perfil_horario_promedio.png", dpi=200)
    plt.close()


def main():
    cfg = Config(
        start_date="2025-01-01",
        days=14,
        interval_minutes=60,  # cambia a 10 si quieres “contador” 10-min
        occupants=4,
        seed=42,
    )

    df = generate_timeseries(cfg)
    csv_path, daily_csv_path = save_outputs(df, cfg)
    plot_outputs(df, cfg)

    print("✅ Generado:")
    print(f"- CSV (serie):   {csv_path}")
    print(f"- CSV (diario):  {daily_csv_path}")
    print(f"- Gráficas en:   {cfg.out_dir}/")


if __name__ == "__main__":
    main()
