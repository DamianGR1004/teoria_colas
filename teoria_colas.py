"""
  Simulación M/M/1 — Sistema de Transporte Público
  Investigación de Operaciones | Teoría de Colas

Módulos:
  1. Generador de datos sintéticos (rutas y horas del día)
  2. Modelo analítico M/M/1
  3. Métricas de desempeño
  4. Visualizaciones comparativas entre rutas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# 1. CONFIGURACIÓN DE RUTAS (datos sintéticos)

@dataclass
class Ruta:
    nombre: str
    color: str
    # λ: pasajeros que llegan por minuto, según franja horaria
    lambda_por_franja: Dict[str, float]   # llegadas/min
    # μ: pasajeros atendidos por autobús por minuto (capacidad efectiva)
    mu: float                             # servicio/min
    # frecuencia: cuántos minutos entre autobús y autobús
    frecuencia_min: float


# Tres rutas con perfiles distintos de demanda
RUTAS: List[Ruta] = [
    Ruta(
        nombre="Ruta 1 — Centro",
        color="#2C7BB6",
        lambda_por_franja={
            "Madrugada (00-06)": 0.5,
            "Mañana pico (07-09)": 6.0,
            "Media mañana (10-12)": 2.5,
            "Mediodía (12-14)": 3.5,
            "Tarde (15-17)": 2.8,
            "Tarde pico (18-20)": 5.5,
            "Noche (21-23)": 1.2,
        },
        mu=8.0,          # atiende hasta 8 pasajeros/min cuando llega
        frecuencia_min=8,
    ),
    Ruta(
        nombre="Ruta 2 — Universidad",
        color="#D7191C",
        lambda_por_franja={
            "Madrugada (00-06)": 0.2,
            "Mañana pico (07-09)": 7.5,   # muy alta por estudiantes
            "Media mañana (10-12)": 4.0,
            "Mediodía (12-14)": 5.0,
            "Tarde (15-17)": 4.5,
            "Tarde pico (18-20)": 6.8,
            "Noche (21-23)": 0.8,
        },
        mu=7.5,
        frecuencia_min=10,                 # frecuencia más baja → más saturada
    ),
    Ruta(
        nombre="Ruta 3 — Periférico",
        color="#1A9641",
        lambda_por_franja={
            "Madrugada (00-06)": 0.1,
            "Mañana pico (07-09)": 3.0,
            "Media mañana (10-12)": 1.5,
            "Mediodía (12-14)": 2.0,
            "Tarde (15-17)": 1.8,
            "Tarde pico (18-20)": 3.2,
            "Noche (21-23)": 0.6,
        },
        mu=6.0,
        frecuencia_min=15,
    ),
]

FRANJAS = list(RUTAS[0].lambda_por_franja.keys())


# 2. MODELO ANALÍTICO M/M/1

class ModeloMM1:
    """
    Sistema M/M/1 estándar.

    Supuestos:
      - Llegadas: proceso de Poisson con tasa λ (pasajeros/min)
      - Servicio: exponencial con tasa μ (pasajeros/min)
      - Un servidor (el autobús cuando llega a la parada)
      - Cola infinita, disciplina FIFO
    """

    def __init__(self, lam: float, mu: float):
        self.lam = lam    # tasa de llegada
        self.mu = mu      # tasa de servicio

    @property
    def rho(self) -> float:
        """Factor de utilización ρ = λ/μ. Debe ser < 1 para estabilidad."""
        return self.lam / self.mu

    @property
    def estable(self) -> bool:
        return self.rho < 1.0

    @property
    def Lq(self) -> float:
        """Número promedio de clientes en cola."""
        if not self.estable:
            return float("inf")
        return self.rho**2 / (1 - self.rho)

    @property
    def L(self) -> float:
        """Número promedio de clientes en el sistema."""
        if not self.estable:
            return float("inf")
        return self.rho / (1 - self.rho)

    @property
    def Wq(self) -> float:
        """Tiempo promedio de espera en cola (minutos)."""
        if not self.estable:
            return float("inf")
        return self.Lq / self.lam

    @property
    def W(self) -> float:
        """Tiempo promedio en el sistema (minutos)."""
        if not self.estable:
            return float("inf")
        return self.L / self.lam

    def prob_espera_mayor_que(self, t: float) -> float:
        """
        P(Wq > t): probabilidad de que un pasajero espere MÁS de t minutos.

        Fórmula: P(Wq > t) = ρ · exp(−μ(1−ρ)t)
        """
        if not self.estable:
            return 1.0
        return self.rho * np.exp(-self.mu * (1 - self.rho) * t)

    def resumen(self) -> dict:
        return {
            "λ (llegadas/min)": round(self.lam, 3),
            "μ (servicio/min)": round(self.mu, 3),
            "ρ (utilización)":  round(self.rho, 3),
            "Lq (cola prom.)":  round(self.Lq, 3) if self.estable else "∞",
            "Wq (espera prom., min)": round(self.Wq, 3) if self.estable else "∞",
            "Estable": self.estable,
        }


# 3. CÁLCULO DE MÉTRICAS POR RUTA Y FRANJA

def calcular_metricas(rutas: List[Ruta]) -> Dict[str, Dict[str, dict]]:
    """
    Devuelve métricas M/M/1 para cada ruta y cada franja horaria.
    """
    resultados = {}
    for ruta in rutas:
        resultados[ruta.nombre] = {}
        for franja, lam in ruta.lambda_por_franja.items():
            modelo = ModeloMM1(lam=lam, mu=ruta.mu)
            resultados[ruta.nombre][franja] = {
                "modelo": modelo,
                "rho": modelo.rho,
                "Wq": modelo.Wq if modelo.estable else None,
                "Lq": modelo.Lq if modelo.estable else None,
                "P_mayor_5min": modelo.prob_espera_mayor_que(5),
                "estable": modelo.estable,
            }
    return resultados


# 4. VISUALIZACIONES

def graficar_comparativa(rutas: List[Ruta], metricas: dict):
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#F8F8F6")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :])   # Utilización ρ (ancho completo)
    ax2 = fig.add_subplot(gs[1, 0])   # Espera promedio Wq
    ax3 = fig.add_subplot(gs[1, 1])   # Pasajeros en cola Lq
    ax4 = fig.add_subplot(gs[2, :])   # P(espera > t) — curvas continuas

    franjas_cortas = [f.split("(")[0].strip() for f in FRANJAS]
    x = np.arange(len(FRANJAS))
    ancho = 0.28

    # ── Gráfica 1: Utilización ρ por franja ──
    for i, ruta in enumerate(rutas):
        rhos = [metricas[ruta.nombre][f]["rho"] for f in FRANJAS]
        bars = ax1.bar(x + i * ancho, rhos, ancho, label=ruta.nombre,
                       color=ruta.color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Etiqueta de valor en barras > 0.8
        for bar, r in zip(bars, rhos):
            if r >= 0.8:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                         f"{r:.2f}", ha="center", va="bottom", fontsize=7.5,
                         color="#C0392B", fontweight="bold")

    ax1.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1.2, label="Límite de estabilidad (ρ=1)")
    ax1.axhline(0.8, color="#E67E22", linestyle=":", linewidth=1, label="Zona de riesgo (ρ=0.8)")
    ax1.set_xticks(x + ancho)
    ax1.set_xticklabels(franjas_cortas, fontsize=9)
    ax1.set_ylabel("Utilización ρ = λ/μ", fontsize=10)
    ax1.set_title("Utilización del sistema por franja horaria\n(ρ ≥ 1 = sistema inestable / colapso)", fontsize=11, pad=10)
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.set_ylim(0, 1.3)
    ax1.set_facecolor("#FAFAFA")
    ax1.grid(axis="y", alpha=0.3, linewidth=0.6)

    # ── Gráfica 2: Tiempo de espera promedio Wq ──
    for i, ruta in enumerate(rutas):
        wqs = [metricas[ruta.nombre][f]["Wq"] or 0 for f in FRANJAS]
        ax2.bar(x + i * ancho, wqs, ancho, label=ruta.nombre,
                color=ruta.color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax2.axhline(5, color="#E74C3C", linestyle="--", linewidth=1, label="Umbral 5 min")
    ax2.set_xticks(x + ancho)
    ax2.set_xticklabels(franjas_cortas, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Tiempo de espera (min)", fontsize=10)
    ax2.set_title("Espera promedio en cola (Wq)", fontsize=11, pad=8)
    ax2.legend(fontsize=8)
    ax2.set_facecolor("#FAFAFA")
    ax2.grid(axis="y", alpha=0.3, linewidth=0.6)

    # ── Gráfica 3: Pasajeros en cola Lq ──
    for i, ruta in enumerate(rutas):
        lqs = [metricas[ruta.nombre][f]["Lq"] or 0 for f in FRANJAS]
        ax3.bar(x + i * ancho, lqs, ancho, label=ruta.nombre,
                color=ruta.color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax3.set_xticks(x + ancho)
    ax3.set_xticklabels(franjas_cortas, fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Pasajeros en cola (Lq)", fontsize=10)
    ax3.set_title("Longitud promedio de cola (Lq)", fontsize=11, pad=8)
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#FAFAFA")
    ax3.grid(axis="y", alpha=0.3, linewidth=0.6)

    # ── Gráfica 4: P(espera > t) — hora pico mañana ──
    franja_pico = "Mañana pico (07-09)"
    t_vals = np.linspace(0, 20, 300)

    for ruta in rutas:
        modelo: ModeloMM1 = metricas[ruta.nombre][franja_pico]["modelo"]
        if modelo.estable:
            probs = [modelo.prob_espera_mayor_que(t) for t in t_vals]
        else:
            probs = [1.0] * len(t_vals)

        ax4.plot(t_vals, probs, color=ruta.color, linewidth=2.2,
                 label=f"{ruta.nombre}  (ρ={modelo.rho:.2f})")

        # Marcar Wq en la curva
        if modelo.estable:
            wq = modelo.Wq
            p_wq = modelo.prob_espera_mayor_que(wq)
            ax4.annotate(f"Wq={wq:.1f}m",
                         xy=(wq, p_wq),
                         xytext=(wq + 1.5, p_wq + 0.05),
                         fontsize=8, color=ruta.color,
                         arrowprops=dict(arrowstyle="->", color=ruta.color, lw=0.8))

    # Línea de umbral P=0.20 (política operativa: máx. 20% de pasajeros esperan > X min)
    ax4.axhline(0.20, color="#7F8C8D", linestyle=":", linewidth=1.1, label="Política: P ≤ 0.20")
    ax4.axvline(5, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.7, label="Umbral: 5 min")

    ax4.set_xlabel("Tiempo de espera t (minutos)", fontsize=10)
    ax4.set_ylabel("P(Wq > t)", fontsize=10)
    ax4.set_title(f"Probabilidad de esperar más de t minutos — {franja_pico}\n"
                  "(¿cuántos pasajeros esperan más de lo aceptable?)", fontsize=11, pad=8)
    ax4.legend(fontsize=8.5, loc="upper right")
    ax4.set_xlim(0, 20)
    ax4.set_ylim(0, 1.05)
    ax4.set_facecolor("#FAFAFA")
    ax4.grid(alpha=0.3, linewidth=0.6)

    plt.suptitle("Sistema M/M/1 — Análisis comparativo de rutas de transporte público",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig("analisis_rutas_mm1.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("✅  Gráfica guardada: analisis_rutas_mm1.png")
    plt.show()


# 5. REPORTE EN CONSOLA

def imprimir_reporte(rutas: List[Ruta], metricas: dict):
    FRANJA_PICO = "Mañana pico (07-09)"
    print("\n" + "═" * 65)
    print("  REPORTE M/M/1 — TRANSPORTE PÚBLICO")
    print("═" * 65)

    for ruta in rutas:
        print(f"\n{'─'*60}")
        print(f"  {ruta.nombre}  |  μ={ruta.mu} pas/min  |  frecuencia={ruta.frecuencia_min} min")
        print(f"{'─'*60}")
        print(f"  {'Franja':<28} {'ρ':>6} {'Wq(min)':>9} {'Lq':>7} {'P(>5min)':>10}")
        print(f"  {'─'*26} {'─'*6} {'─'*9} {'─'*7} {'─'*10}")

        for franja in FRANJAS:
            m = metricas[ruta.nombre][franja]
            rho_s = f"{m['rho']:.3f}"
            wq_s  = f"{m['Wq']:.2f}" if m["Wq"] is not None else "  ∞"
            lq_s  = f"{m['Lq']:.2f}" if m["Lq"] is not None else "  ∞"
            p5_s  = f"{m['P_mayor_5min']:.3f}"
            alerta = " ⚠" if m["rho"] >= 0.85 else ""
            print(f"  {franja:<28} {rho_s:>6} {wq_s:>9} {lq_s:>7} {p5_s:>10}{alerta}")

    # Ranking de saturación (hora pico mañana)
    print(f"\n{'═'*65}")
    print(f"  RANKING DE SATURACIÓN — {FRANJA_PICO}")
    print(f"{'═'*65}")
    ranking = sorted(
        rutas,
        key=lambda r: metricas[r.nombre][FRANJA_PICO]["rho"],
        reverse=True
    )
    for pos, ruta in enumerate(ranking, 1):
        m = metricas[ruta.nombre][FRANJA_PICO]
        estado = "🔴 CRÍTICA" if m["rho"] >= 1 else ("🟠 ALTA" if m["rho"] >= 0.8 else "🟢 OK")
        print(f"  {pos}. {ruta.nombre:<30} ρ={m['rho']:.3f}  {estado}")

    print(f"\n  ⚠  = utilización ≥ 0.85 (riesgo de colapso)")
    print("═" * 65 + "\n")


# ─────────────────────────────────────────────
# 6. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Calculando métricas M/M/1...")
    metricas = calcular_metricas(RUTAS)

    imprimir_reporte(RUTAS, metricas)

    print("Generando visualizaciones...")
    graficar_comparativa(RUTAS, metricas)