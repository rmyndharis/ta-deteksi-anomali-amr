"""
Gambar Studi Kasus Per Fasa
============================
Arus (atau tegangan) rata-rata harian per fasa untuk beberapa kasus P2TL
di sekitar tanggal laporan, dari data/AP2T/dataset_lp_p2tl_fasa.csv.
Menunjukkan pola satu fasa tidak terukur selama masa sebelum penertiban
dan pemulihannya setelah penertiban.

Output: experiments/analysis_hibrida/gambar_kasus_fasa.png (panel 2x2)
        experiments/analysis_hibrida/gambar_kasus_<id>.png (per kasus)

Jalankan dari folder code/:  ./venv/bin/python gambar_kasus_fasa.py [ID ...]

Penulis: Yudhi Armyndharis (220401010272)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
FASA_FILE = os.path.join(os.path.dirname(BASE), "data", "AP2T", "dataset_lp_p2tl_fasa.csv")
OUT = os.path.join(BASE, "experiments", "analysis_hibrida")
os.makedirs(OUT, exist_ok=True)

# (kasus, besaran yang digambar)
KASUS = [("A073", "arus"), ("A081", "arus"), ("A093", "arus"), ("A107", "tegangan")]
WARNA = {"L1": "#2a78d6", "L2": "#eb6834", "L3": "#1baf7a"}
GAYA = {"L1": "-", "L2": "--", "L3": "-."}
BULAN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
         7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}


def tanggal_id(v, _pos=None):
    """Label tanggal sumbu x dengan nama bulan Indonesia: 15 Feb 2017."""
    d = mdates.num2date(v)
    return f"{d.day:02d} {BULAN[d.month]} {d.year}"


def angka_koma(v, _pos=None):
    """Label angka sumbu y dengan koma desimal: 0,8."""
    return f"{v:g}".replace(".", ",")


def harian(df, cid, besaran):
    d = df[df["case_id"] == cid].copy()
    kolom = {"arus": ["current_l1", "current_l2", "current_l3"],
             "tegangan": ["voltage_l1", "voltage_l2", "voltage_l3"]}[besaran]
    h = d.groupby(d["timestamp"].dt.normalize())[kolom].mean()
    h.columns = ["L1", "L2", "L3"]
    h = h.asfreq("D")                      # hari tanpa data dibiarkan kosong (garis terputus)
    return h, pd.to_datetime(d["tgl_lap_p2tl"].iloc[0])


def gambar_panel(ax, h, tgl, cid, besaran):
    for f in ["L1", "L2", "L3"]:
        ax.plot(h.index, h[f], color=WARNA[f], linestyle=GAYA[f], linewidth=1.8, label=f)
    ax.axvline(tgl, color="#444444", linestyle=":", linewidth=1.5)
    ax.set_ylim(bottom=0)
    ymaks = ax.get_ylim()[1]
    ax.text(tgl, 0.97 * ymaks, "tanggal laporan P2TL ", fontsize=8, color="#444444",
            va="top", ha="right", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    ax.set_title(f"Kasus {cid}: {besaran} rata-rata harian per fasa", fontsize=10)
    ax.set_ylabel("Arus (A, sisi sekunder)" if besaran == "arus" else "Tegangan (V, sisi sekunder)", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(tanggal_id))
    ax.yaxis.set_major_formatter(FuncFormatter(angka_koma))
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)


def main():
    df = pd.read_csv(FASA_FILE, low_memory=False, dtype={"meter_id": str, "case_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    kasus = KASUS if len(sys.argv) < 2 else [(c, "arus") for c in sys.argv[1:]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (cid, besaran) in zip(axes.ravel(), kasus):
        h, tgl = harian(df, cid, besaran)
        gambar_panel(ax, h, tgl, cid, besaran)
    garis, label = axes[0, 0].get_legend_handles_labels()
    fig.legend(garis, label, title="Fasa", loc="lower center", ncol=3, fontsize=9, title_fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(os.path.join(OUT, "gambar_kasus_fasa.png"), dpi=150, bbox_inches="tight")
    plt.close()

    for cid, besaran in kasus:
        h, tgl = harian(df, cid, besaran)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        gambar_panel(ax, h, tgl, cid, besaran)
        ax.legend(title="Fasa", fontsize=8, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"gambar_kasus_{cid}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        pre, post = h[h.index <= tgl], h[h.index > tgl]
        print(f"  {cid} ({besaran}): rata-rata pre L1/L2/L3 = {pre.mean().round(2).tolist()}"
              + (f" | post = {post.mean().round(2).tolist()}" if len(post) else " | post = (tidak ada data)"))
    print(f"  Gambar disimpan di {OUT}")


if __name__ == "__main__":
    main()
