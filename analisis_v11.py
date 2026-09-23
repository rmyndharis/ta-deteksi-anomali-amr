"""
Analisis Lanjutan Bab IV (artefak aktif, tanpa pelatihan ulang)
================================================================
Semua perhitungan memakai model dan skor yang aktif di models/ dan output/
(hasil `python main.py --population`) pada set uji kasus-kontrol yang sama
(seluruh meter uji tidak pernah dilihat model). Dua pembanding tanpa
pelatihan direplikasi dari CSV mentah dengan normalisasi per meter yang sama:
deviasi-median kWh (bl: skor = -rata-rata kWh ter-skala per hari) dan deviasi
dua kanal (bl2: skor = -rata-rata gabungan kWh dan arus ter-skala per hari,
kanal yang sama dengan skor model; ditambahkan 9 September 2026).

1. KETIDAKPASTIAN BERBASIS KLASTER
   Window dari kasus/meter yang sama berkorelasi, sehingga bootstrap per
   window terlalu optimistis. Resampling dilakukan per klaster: case_id
   untuk window positif (pre-P2TL) dan meter_id untuk window negatif.
   Protokol: (a) pre vs seluruh meter kontrol + post-P2TL (UTAMA),
   (b) pre vs kontrol saja, (c) pre vs post. Juga: recall window pada
   anggaran positif palsu kontrol, AUC parsial (FPR <= 5/10/20%),
   AUC tingkat pelanggan (rata-rata skor per kasus vs per meter kontrol),
   recall tingkat kasus (>= 1 hari > P95) dan pada anggaran FP tingkat meter.
1c. ATURAN URUTAN DAFTAR PRIORITAS: 117 unit uji (32 kasus, 85 meter kontrol)
   diurutkan dengan aturan yang dipakai dashboard (skor rata-rata) dan
   aturan lama (persentase hari tertandai lalu skor rata-rata) serta kedua
   pembanding; kasus yang masuk K = 6, 12, 23 teratas (precision@K, recall@K).
   Dihitung dua kali: tanpa filter (117 unit) dan dengan filter bawaan
   tampilan dashboard (minimal 7 hari dinilai dan minimal satu hari
   tertandai), yang menyisakan lebih sedikit unit (kolom n_unit).
2. DETEKTABILITAS PER KASUS (Lampiran C): rasio energi dan arus harian
   pre/post, skor rata-rata, fraksi hari tertandai, golongan P2TL.
3. NORMALISASI PROSPEKTIF: median/IQR tiap hari dihitung HANYA dari riwayat
   meter sebelum tanggal window (minimal MIN_HIST_DAYS hari), set uji diskor
   ulang dengan model yang sama; pembanding adil = skor seluruh riwayat pada
   subset window yang sama.

Output: experiments/analysis_v11/
  ringkasan_v11.json, tabel_auc_cluster.csv, tabel_recall_fp_cluster.csv,
  tabel_pauc.csv, tabel_level_meter.csv, tabel_topk_peringkat.csv,
  tabel_detektabilitas_kasus.csv, tabel_prospektif.csv, forest_auc.png,
  skor_prospektif.pkl
Jalankan dari code/:  ./venv/bin/python -u analisis_v11.py 2>&1 | tee logs/v11_06_analisis.log
Opsi: --skip-prospektif. Env: B_BOOT (default 2000), TA_SMOKE_OUT (folder output).

Penulis: Yudhi Armyndharis (220401010272)
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)
from config import (INTERVAL_MINUTES, SEQUENCE_LENGTH, ELECTRICAL_FEATURES,
                    MODEL_FEATURES, SCALED_CLIP, RAW_DATA_FILE)
assert INTERVAL_MINUTES == 30 and SEQUENCE_LENGTH == 48, "set_grid 30 dulu"
from artefak import load_detection, load_model_and_threshold

OUT = os.environ.get("TA_SMOKE_OUT", os.path.join(BASE, "experiments", "analysis_v11"))
os.makedirs(OUT, exist_ok=True)

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
_ap.add_argument("--skip-prospektif", action="store_true", help="lewati bagian 3")
args = _ap.parse_args()

B_BOOT = int(os.environ.get("B_BOOT", 2000))
MIN_HIST_DAYS = 7
MIN_HARI_DASHBOARD = 7        # filter bawaan "Minimal hari dinilai" pada halaman Daftar Prioritas
FP_BUDGETS = [0.05, 0.10, 0.20]
BUDGETS_WINDOW = [0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20]


def auc(pos, neg):
    """AUC-ROC via statistik Mann-Whitney."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot_auc(pos, pos_cl, neg, neg_cl, B=B_BOOT, seed=0):
    """Bootstrap klaster: klaster positif dan negatif di-resample dengan penggantian."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pg = {k: pos[np.asarray(pos_cl) == k] for k in np.unique(pos_cl)}
    ng = {k: neg[np.asarray(neg_cl) == k] for k in np.unique(neg_cl)}
    pk, nk = list(pg), list(ng)
    v = np.empty(B)
    for b in range(B):
        ps = np.concatenate([pg[k] for k in rng.choice(pk, len(pk))])
        ns = np.concatenate([ng[k] for k in rng.choice(nk, len(nk))])
        v[b] = auc(ps, ns)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def window_boot_auc(pos, neg, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_prop(flags, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed); flags = np.asarray(flags, float)
    v = [rng.choice(flags, len(flags)).mean() for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pauc(pos, neg, fmax):
    """AUC parsial pada FPR <= fmax (ternormalisasi McClish: 0,5 = acak, 1 = sempurna)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y) / len(pos); fp = np.cumsum(1 - y) / len(neg)
    m = fp <= fmax
    f = np.concatenate([[0], fp[m], [fmax]]); t = np.concatenate([[0], tp[m], [tp[m][-1] if m.any() else 0]])
    area = np.trapezoid(t, f) if hasattr(np, "trapezoid") else np.trapz(t, f)
    amin, amax = fmax ** 2 / 2, fmax
    return float(0.5 * (1 + (area - amin) / (amax - amin)))


def confusion(pos_flag, neg_flag):
    tp = int(pos_flag.sum()); fn = int(len(pos_flag) - tp)
    fp = int(neg_flag.sum()); tn = int(len(neg_flag) - fp)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def robust_params(ref):
    med = ref.median(); iqr = ref.quantile(0.75) - ref.quantile(0.25); std = ref.std()
    scale = iqr.where(iqr > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
    return med, scale


# ============================================================== data dasar
t0 = time.time()
ti, thr = load_detection()

# baseline deviasi-median dari CSV mentah (grid 30 menit, robust scaling per meter)
from preprocessing import load_dataset, resample_to_grid
raw = load_dataset(RAW_DATA_FILE)
grid = resample_to_grid(raw)
grid["meter_id"] = grid["meter_id"].astype(str)
grid["date"] = grid["timestamp"].dt.strftime("%Y-%m-%d")

bl_rows = []
for m, g in grid.groupby("meter_id", sort=False):
    med, sc = robust_params(g[["kWh", "current"]])
    ks = ((g["kWh"] - med["kWh"]) / sc["kWh"]).clip(-SCALED_CLIP, SCALED_CLIP)
    cs = ((g["current"] - med["current"]) / sc["current"]).clip(-SCALED_CLIP, SCALED_CLIP)
    day = pd.DataFrame({"date": g["date"], "ks": ks, "cs": cs, "kwh": g["kWh"], "cur": g["current"]}).groupby("date").agg(
        n=("ks", "size"), nn=("ks", "count"), ks=("ks", "mean"), cs=("cs", "mean"), kwh=("kwh", "sum"), cur=("cur", "mean"))
    day = day[(day["n"] == SEQUENCE_LENGTH) & (day["nn"] == SEQUENCE_LENGTH)]
    bl_rows.append(pd.DataFrame({"meter_id": str(m), "date": day.index, "bl": -day["ks"].to_numpy(),
                                 "bl2": -(day["ks"].to_numpy() + day["cs"].to_numpy()) / 2.0,
                                 "kwh": day["kwh"].to_numpy(), "cur": day["cur"].to_numpy()}))
bl = pd.concat(bl_rows, ignore_index=True)
ti = ti.merge(bl, on=["meter_id", "date"], how="left")
assert ti["bl"].notna().all() and ti["bl2"].notna().all(), "replikasi baseline tidak selaras dengan window model"
SKOR = [("score", "model"), ("bl", "baseline"), ("bl2", "baseline2")]   # model, deviasi-median kWh, deviasi dua kanal

pre = ti[ti["periode"] == "pre_p2tl"]; post = ti[ti["periode"] == "post_p2tl"]
kon = ti[ti["grp"] == "kontrol"]
PROTO = {"utama_kontrol+post": pd.concat([post, kon]), "pre_vs_kontrol": kon, "pre_vs_post": post}
ringkas = {"n_window": int(len(ti)), "n_pre": int(len(pre)), "n_post": int(len(post)),
           "n_kontrol_window": int(len(kon)), "n_kontrol_meter": int(kon["meter_id"].nunique()),
           "n_kasus_pre": int(pre["case_id"].nunique()),
           "n_meter_kasus": int(ti[ti["grp"] == "p2tl"]["meter_id"].nunique()),
           "threshold_p95": thr, "B_bootstrap": B_BOOT}
print(f"Set uji: {len(ti):,} window | pre {len(pre)} ({pre['case_id'].nunique()} kasus) | post {len(post)} "
      f"| kontrol {len(kon)} ({kon['meter_id'].nunique()} meter) | ambang {thr:.4f}")

# ============================================================== 1. klaster
print("\n=== 1. AUC dengan bootstrap klaster (case_id positif, meter_id negatif) ===")
rows = []
for nama, neg in PROTO.items():
    for skor, lab in SKOR:
        a = auc(pre[skor], neg[skor])
        lo, hi = cluster_boot_auc(pre[skor], pre["case_id"], neg[skor], neg["meter_id"])
        wlo, whi = window_boot_auc(pre[skor], neg[skor])
        rows.append({"protokol": nama, "skor": lab, "n_pos": len(pre), "n_neg": len(neg),
                     "n_klaster_pos": pre["case_id"].nunique(), "n_klaster_neg": neg["meter_id"].nunique(),
                     "auc": a, "ci_klaster_lo": lo, "ci_klaster_hi": hi, "ci_window_lo": wlo, "ci_window_hi": whi})
        print(f"  {nama:20s} {lab:8s} AUC {a:.4f}  CI klaster [{lo:.4f}, {hi:.4f}]  (CI window [{wlo:.4f}, {whi:.4f}])")
tab_auc = pd.DataFrame(rows); tab_auc.to_csv(os.path.join(OUT, "tabel_auc_cluster.csv"), index=False)

# confusion matrix pada ambang P95 (angka Tabel 4.1) untuk model dan baseline
cm = {}
neg = PROTO["utama_kontrol+post"]
thr_bl = float(np.quantile(kon["bl"], 1 - float(kon["flag"].mean()))) if False else None
c = confusion(pre["flag"].to_numpy(), neg["flag"].to_numpy())
c.update({"n": int(len(pre) + len(neg)), "auc": auc(pre["score"], neg["score"]),
          "flag_rate_pre": float(pre["flag"].mean()), "flag_rate_post": float(post["flag"].mean()),
          "flag_rate_kontrol": float(kon["flag"].mean()), "n_flag_kontrol": int(kon["flag"].sum()),
          "akurasi": float((c["tp"] + c["tn"]) / (len(pre) + len(neg))),
          "akurasi_semua_normal": float(len(neg) / (len(pre) + len(neg)))})
cm["model"] = c
print(f"  model P95: n={c['n']} TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} "
      f"P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f} AUC={c['auc']:.4f} "
      f"FPR kontrol={c['flag_rate_kontrol']:.4f} akurasi={c['akurasi']:.4f} (semua-normal {c['akurasi_semua_normal']:.4f})")
ringkas["confusion_p95"] = cm

# recall window pada anggaran FP (negatif = window kontrol) dengan CI klaster
print("\n  Recall window pre pada anggaran FP kontrol (CI klaster per kasus, ambang tetap):")
rec_rows = []
for b in BUDGETS_WINDOW:
    for skor, lab in SKOR:
        t = np.quantile(kon[skor], 1 - b); fl = (pre[skor] > t).astype(float).to_numpy()
        rng = np.random.default_rng(0); cl = pre["case_id"].to_numpy(); ks = np.unique(cl)
        v = [np.concatenate([fl[cl == k] for k in rng.choice(ks, len(ks))]).mean() for _ in range(B_BOOT)]
        rec_rows.append({"fp_budget": b, "skor": lab, "ambang": float(t), "recall": float(fl.mean()),
                         "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5))})
tab_rec = pd.DataFrame(rec_rows); tab_rec.to_csv(os.path.join(OUT, "tabel_recall_fp_cluster.csv"), index=False)
print(tab_rec[tab_rec["fp_budget"].isin([0.05, 0.10, 0.20])].round(3).to_string(index=False))

# AUC parsial
prow = []
for fm in [0.05, 0.10, 0.20, 1.0]:
    prow.append({"protokol": "utama_kontrol+post", "fpr_maks": fm,
                 "model": pauc(pre["score"], neg["score"], fm), "baseline": pauc(pre["bl"], neg["bl"], fm),
                 "baseline2": pauc(pre["bl2"], neg["bl2"], fm)})
ptab = pd.DataFrame(prow); ptab.to_csv(os.path.join(OUT, "tabel_pauc.csv"), index=False)
print("\n  AUC parsial ternormalisasi (0,5 = acak):\n" + ptab.round(3).to_string(index=False))

# level meter / kasus
print("\n=== 1b. Tingkat pelanggan (kasus vs meter kontrol) ===")
lm = {}
ca = pre.groupby("case_id").agg(meter_id=("meter_id", "first"), skor=("score", "mean"), bl=("bl", "mean"),
                                bl2=("bl2", "mean"), n_hari=("score", "size"), ada_flag=("flag", "max"),
                                fraksi_flag=("flag", "mean"))
ka = kon.groupby("meter_id").agg(skor=("score", "mean"), bl=("bl", "mean"), bl2=("bl2", "mean"),
                                 n_hari=("score", "size"), ada_flag=("flag", "max"), fraksi_flag=("flag", "mean"))
UNIT = [("model", "skor"), ("baseline", "bl"), ("baseline2", "bl2")]
for lab, col in UNIT:
    a = auc(ca[col], ka[col]); rng = np.random.default_rng(0)
    v = [auc(rng.choice(ca[col], len(ca)), rng.choice(ka[col], len(ka))) for _ in range(B_BOOT)]
    lm[f"auc_level_meter_{lab}"] = {"auc": a, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                    "n_kasus": int(len(ca)), "n_kontrol": int(len(ka))}
    print(f"  AUC tingkat pelanggan {lab:8s}: {a:.4f} CI [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")
rk = float(ca["ada_flag"].mean()); lo, hi = boot_prop(ca["ada_flag"]); fk = float(ka["ada_flag"].mean())
lm["recall_kasus_p95_min1hari"] = {"recall": rk, "ci95": [lo, hi], "n_kasus_tertangkap": int(ca["ada_flag"].sum()),
                                  "n_kasus": int(len(ca)), "fpr_meter_kontrol_min1hari": fk,
                                  "n_meter_kontrol_tertandai": int(ka["ada_flag"].sum())}
print(f"  Recall tingkat kasus (>=1 hari > P95): {int(ca['ada_flag'].sum())}/{len(ca)} = {rk:.3f} CI [{lo:.3f}, {hi:.3f}]; "
      f"meter kontrol dengan >=1 hari tertandai: {int(ka['ada_flag'].sum())}/{len(ka)} = {fk:.3f}")
lm_rows = []
for b in FP_BUDGETS:
    for lab, col in UNIT:
        t = np.quantile(ka[col], 1 - b); fl = (ca[col] > t).astype(float); lo, hi = boot_prop(fl)
        lm_rows.append({"fp_budget_meter": b, "skor": lab, "ambang": float(t), "recall_kasus": float(fl.mean()),
                        "n_kasus_tertangkap": int(fl.sum()), "ci_lo": lo, "ci_hi": hi,
                        "n_meter_kontrol_tertandai": int((ka[col] > t).sum())})
tab_lm = pd.DataFrame(lm_rows); tab_lm.to_csv(os.path.join(OUT, "tabel_level_meter.csv"), index=False)
print(tab_lm.round(3).to_string(index=False))
ringkas["level_meter"] = lm

# ============================================================== 1c. aturan urutan daftar prioritas
print("\n=== 1c. Aturan urutan daftar prioritas pada set uji (32 kasus vs 85 meter kontrol) ===")
units = pd.concat([ca.assign(jenis="kasus"), ka.assign(jenis="kontrol")], ignore_index=True)
ATURAN = {"skor_rata_rata_model": ["skor"],                       # dipakai dashboard sejak 9 September 2026
          "fraksi_hari_tertandai_lalu_skor": ["fraksi_flag", "skor"],   # aturan lama dashboard
          "deviasi_median_kwh": ["bl"], "deviasi_dua_kanal": ["bl2"]}
K_LIST = [6, 12, 23]                                               # sekitar 5, 10, 20% dari 117 unit
# filter bawaan halaman Daftar Prioritas (dashboard.py): minimal 7 hari dinilai dan hanya meter yang tertandai
FILTER = {"tanpa": units,
          "bawaan_dashboard": units[(units["n_hari"] >= MIN_HARI_DASHBOARD) & (units["fraksi_flag"] > 0)]}
tk_rows = []
for f_nama, u in FILTER.items():
    n_kasus_u = int((u["jenis"] == "kasus").sum())
    for nama, kunci in ATURAN.items():
        urut = u.sort_values(kunci, ascending=[False] * len(kunci), kind="mergesort")
        a_unit = auc(u.loc[u["jenis"] == "kasus", kunci[0]], u.loc[u["jenis"] == "kontrol", kunci[0]])
        for K in K_LIST:
            ditemukan = int((urut.head(K)["jenis"] == "kasus").sum())
            tk_rows.append({"filter": f_nama, "n_unit": int(len(u)), "n_kasus": n_kasus_u,
                            "n_kontrol": int(len(u) - n_kasus_u), "aturan": nama, "K": K,
                            "kasus_ditemukan": ditemukan, "precision_at_k": ditemukan / K,
                            "recall_at_k": ditemukan / n_kasus_u, "auc_unit_kunci_utama": a_unit})
tab_tk = pd.DataFrame(tk_rows); tab_tk.to_csv(os.path.join(OUT, "tabel_topk_peringkat.csv"), index=False)
print(tab_tk.round(3).to_string(index=False))
ringkas["peringkat_topk"] = tab_tk.to_dict(orient="records")

# ============================================================== 2. detektabilitas per kasus
print("\n=== 2. Detektabilitas per kasus (Lampiran C) ===")
meta = pd.read_csv(RAW_DATA_FILE, usecols=["case_id", "meter_id", "gol_p2tl", "tarip", "daya", "tgl_lap_p2tl"],
                   low_memory=False).drop_duplicates("case_id")
meta["case_id"] = meta["case_id"].astype(str)
cs = []
for c_id, g in ti[ti["grp"] == "p2tl"].groupby("case_id"):
    p = g[g["periode"] == "pre_p2tl"]; q = g[g["periode"] == "post_p2tl"]
    if len(p) == 0:
        continue
    cs.append({"case_id": c_id, "n_pre": len(p), "n_post": len(q),
               "kwh_harian_pre": p["kwh"].mean(), "kwh_harian_post": q["kwh"].mean() if len(q) else np.nan,
               "arus_pre": p["cur"].mean(), "arus_post": q["cur"].mean() if len(q) else np.nan,
               "skor_rata_pre": p["score"].mean(), "skor_baseline_pre": p["bl"].mean(),
               "fraksi_hari_tertandai_p95": p["flag"].mean()})
cs = pd.DataFrame(cs).merge(meta, on="case_id", how="left")
with np.errstate(divide="ignore", invalid="ignore"):
    cs["rasio_kwh_pre_post"] = cs["kwh_harian_pre"] / cs["kwh_harian_post"]
    cs["rasio_arus_pre_post"] = cs["arus_pre"] / cs["arus_post"]
cs = cs.replace([np.inf, -np.inf], np.nan).sort_values("skor_rata_pre", ascending=False)
cs.to_csv(os.path.join(OUT, "tabel_detektabilitas_kasus.csv"), index=False)
ok = cs.dropna(subset=["rasio_kwh_pre_post"])
sup = ok[ok["rasio_kwh_pre_post"] < 0.8]; nosup = ok[ok["rasio_kwh_pre_post"] >= 0.8]
det = {"n_kasus_uji": int(len(cs)), "n_kasus_pre_post_lengkap": int(len(ok)),
       "n_energi_turun_lt80pct": int(len(sup)), "n_tidak_turun": int(len(nosup)),
       "n_pre_lebih_tinggi_gt100pct": int((ok["rasio_kwh_pre_post"] > 1.0).sum()),
       "n_pre_lebih_tinggi_gt120pct": int((ok["rasio_kwh_pre_post"] > 1.2).sum()),
       "skor_rata_kasus_turun": float(sup["skor_rata_pre"].mean()) if len(sup) else None,
       "skor_rata_kasus_tidak_turun": float(nosup["skor_rata_pre"].mean()) if len(nosup) else None,
       "tersentuh_p95_kasus_turun": f"{int((sup['fraksi_hari_tertandai_p95']>0).sum())}/{len(sup)}",
       "tersentuh_p95_kasus_tidak_turun": f"{int((nosup['fraksi_hari_tertandai_p95']>0).sum())}/{len(nosup)}",
       "tersentuh_p95_tidak_dapat_dibandingkan": f"{int((cs.loc[~cs.index.isin(ok.index), 'fraksi_hari_tertandai_p95']>0).sum())}/{len(cs)-len(ok)}",
       "spearman_skor_vs_rasio_kwh": float(ok[["skor_rata_pre", "rasio_kwh_pre_post"]].corr("spearman").iloc[0, 1]),
       "spearman_baseline_vs_rasio_kwh": float(ok[["skor_baseline_pre", "rasio_kwh_pre_post"]].corr("spearman").iloc[0, 1]),
       "komposisi_golongan_kasus": meta["gol_p2tl"].value_counts().to_dict()}
ringkas["detektabilitas"] = det
print(json.dumps(det, indent=1))

# ============================================================== 3. prospektif
if not args.skip_prospektif:
    print(f"\n=== 3. Normalisasi prospektif (riwayat >= {MIN_HIST_DAYS} hari sebelum tanggal window) ===")
    from model import mask_energy_channels, MASKED_IDX
    model, _ = load_model_and_threshold()

    def scale_prospective(g):
        """Skala tiap hari dengan median/IQR dari riwayat SEBELUM hari itu."""
        g = g.sort_values("timestamp").copy()
        day = g["timestamp"].dt.normalize()
        out = np.full((len(g), len(ELECTRICAL_FEATURES)), np.nan, dtype=float)
        vals = g[ELECTRICAL_FEATURES]
        for dt in day.unique():
            hist = vals[day < dt].dropna()
            if hist["kWh"].size == 0 or (day[day < dt].nunique() < MIN_HIST_DAYS):
                continue
            med, sc = robust_params(hist)
            idx = (day == dt).to_numpy()
            out[idx] = ((vals[idx] - med) / sc).clip(-SCALED_CLIP, SCALED_CLIP).to_numpy()
        g[ELECTRICAL_FEATURES] = out
        return g

    parts = [scale_prospective(g) for _, g in grid.groupby("meter_id", sort=False)]
    gp = pd.concat(parts, ignore_index=True)
    gp["hour"] = gp["hour"] / 23.0; gp["day_of_week"] = gp["day_of_week"] / 6.0
    from preprocessing import create_sequences
    Xp, yp, infop = create_sequences(gp)
    infop["meter_id"] = infop["meter_id"].astype(str); infop["date"] = pd.to_datetime(infop["date"]).dt.strftime("%Y-%m-%d")
    Xr = model.predict(mask_energy_channels(Xp), batch_size=256, verbose=0)
    infop["score_prosp"] = (Xr[:, :, MASKED_IDX] - Xp[:, :, MASKED_IDX]).mean(axis=(1, 2))
    kidx = MODEL_FEATURES.index("kWh"); cidx = MODEL_FEATURES.index("current")
    infop["bl_prosp"] = -Xp[:, :, kidx].mean(axis=1)
    infop["bl2_prosp"] = -Xp[:, :, [kidx, cidx]].mean(axis=(1, 2))
    sub = ti.merge(infop[["meter_id", "date", "score_prosp", "bl_prosp", "bl2_prosp"]], on=["meter_id", "date"], how="inner")
    print(f"  Window dengan riwayat cukup: {len(sub):,} dari {len(ti):,} "
          f"(pre {int((sub['periode']=='pre_p2tl').sum())}, post {int((sub['periode']=='post_p2tl').sum())}, kontrol {int((sub['grp']=='kontrol').sum())})")
    with open(os.path.join(OUT, "skor_prospektif.pkl"), "wb") as f:
        pickle.dump(sub, f)
    spre = sub[sub["periode"] == "pre_p2tl"]; spost = sub[sub["periode"] == "post_p2tl"]; skon = sub[sub["grp"] == "kontrol"]
    COLS = [("score", "model_seluruh_riwayat"), ("score_prosp", "model_prospektif"),
            ("bl", "baseline_seluruh_riwayat"), ("bl_prosp", "baseline_prospektif"),
            ("bl2", "baseline2_seluruh_riwayat"), ("bl2_prosp", "baseline2_prospektif")]
    prows = []
    for nama, negp in [("utama_kontrol+post", pd.concat([spost, skon])), ("pre_vs_kontrol", skon), ("pre_vs_post", spost)]:
        for col, lab in COLS:
            a = auc(spre[col], negp[col]); lo, hi = cluster_boot_auc(spre[col], spre["case_id"], negp[col], negp["meter_id"])
            prows.append({"protokol": nama, "skor": lab, "n_pos": len(spre), "n_neg": len(negp), "auc": a, "ci_klaster_lo": lo, "ci_klaster_hi": hi})
            print(f"  {nama:20s} {lab:26s} AUC {a:.4f} CI [{lo:.4f}, {hi:.4f}]")
    for b in [0.05, 0.10]:
        for col, lab in COLS:
            t = np.quantile(skon[col], 1 - b)
            prows.append({"protokol": f"recall_pre@fp{int(b*100)}%_kontrol", "skor": lab, "n_pos": len(spre), "n_neg": len(skon),
                          "auc": float((spre[col] > t).mean()), "ci_klaster_lo": np.nan, "ci_klaster_hi": np.nan})
    cap = spre.groupby("case_id")[[c for c, _ in COLS]].mean(); kap = skon.groupby("meter_id")[[c for c, _ in COLS]].mean()
    for col, lab in COLS:
        a = auc(cap[col], kap[col]); rng = np.random.default_rng(0)
        v = [auc(rng.choice(cap[col], len(cap)), rng.choice(kap[col], len(kap))) for _ in range(B_BOOT)]
        prows.append({"protokol": "level_meter_kasus_vs_kontrol", "skor": lab, "n_pos": len(cap), "n_neg": len(kap), "auc": a,
                      "ci_klaster_lo": float(np.percentile(v, 2.5)), "ci_klaster_hi": float(np.percentile(v, 97.5))})
        print(f"  tingkat pelanggan {lab:26s} AUC {a:.4f}")
    tab_p = pd.DataFrame(prows); tab_p.to_csv(os.path.join(OUT, "tabel_prospektif.csv"), index=False)
    ringkas["prospektif"] = {"min_hist_days": MIN_HIST_DAYS, "n_window_subset": int(len(sub)),
                             "n_pre": int(len(spre)), "n_post": int(len(spost)), "n_kontrol": int(len(skon)),
                             "tabel": tab_p.to_dict(orient="records")}

# ============================================================== forest plot AUC
fig, ax = plt.subplots(figsize=(8.5, 5.2))
items = []
for nama, lab in [("utama_kontrol+post", "Window: pre vs kontrol + post (utama)"),
                  ("pre_vs_kontrol", "Window: pre vs kontrol"), ("pre_vs_post", "Window: pre vs post")]:
    for skor, sl in [("model", "LSTM-AE"), ("baseline", "deviasi median"), ("baseline2", "deviasi dua kanal")]:
        r = tab_auc[(tab_auc["protokol"] == nama) & (tab_auc["skor"] == skor)].iloc[0]
        items.append((f"{lab} [{sl}]", r["auc"], r["ci_klaster_lo"], r["ci_klaster_hi"]))
for lab, key in [("Tingkat pelanggan [LSTM-AE]", "auc_level_meter_model"), ("Tingkat pelanggan [deviasi median]", "auc_level_meter_baseline"),
                 ("Tingkat pelanggan [deviasi dua kanal]", "auc_level_meter_baseline2")]:
    items.append((lab, lm[key]["auc"], lm[key]["ci95"][0], lm[key]["ci95"][1]))
ys = np.arange(len(items))[::-1]
for y, (lab, a, lo, hi) in zip(ys, items):
    ax.plot([lo, hi], [y, y], color="steelblue" if "LSTM" in lab else "gray", lw=2)
    ax.plot(a, y, "o", color="steelblue" if "LSTM" in lab else "gray")
ax.axvline(0.5, color="red", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=9)
ax.set_xlabel("AUC-ROC (titik) dan interval kepercayaan 95% (garis)")
ax.set_title("AUC dengan Ketidakpastian Berbasis Klaster (kasus dan meter)")
ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "forest_auc.png"), dpi=150, bbox_inches="tight"); plt.close()

with open(os.path.join(OUT, "ringkasan_v11.json"), "w") as f:
    json.dump(ringkas, f, indent=2, default=float)
print(f"\nSELESAI ({time.time()-t0:.0f} detik). Output di: {OUT}")
