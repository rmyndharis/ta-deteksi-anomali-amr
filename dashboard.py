"""
Prototype Dashboard Interaktif
================================
Dashboard interaktif menggunakan Streamlit untuk memvisualisasikan
hasil deteksi anomali.

Komponen utama:
1. Grafik time-series konsumsi energi listrik pelanggan
2. Indikator deteksi anomali pada titik data tertentu
3. Ringkasan metrik evaluasi model
4. Grafik per pelanggan dengan window terdeteksi
5. Daftar prioritas pemeriksaan (tabel peringkat meter populasi)

Jalankan: streamlit run dashboard.py
Parameter URL: ?page=overview|timeseries|pelanggan|prioritas
               &anonim=1  -> ID meter disamarkan (M-0001, M-0002, ...) untuk
                             tangkapan layar naskah; pemetaan dibuat dari
                             urutan ID asli dan tidak disimpan ke berkas.
               &meter=<id> (halaman pelanggan), &cari=<teks> (halaman prioritas)

Penulis: Yudhi Armyndharis (220401010272)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Konfigurasi path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_FEATURES,
    RAW_DATA_FILE
)


@st.cache_data
def load_data():
    """Load semua data yang dibutuhkan dashboard."""
    data = {}

    # Load detection results
    detection_file = os.path.join(OUTPUT_DIR, "detection_results.pkl")
    if os.path.exists(detection_file):
        with open(detection_file, "rb") as f:
            data["detection"] = pickle.load(f)

    # Load evaluation metrics
    metrics_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.pkl")
    if os.path.exists(metrics_file):
        with open(metrics_file, "rb") as f:
            data["metrics"] = pickle.load(f)

    # Load dataset untuk visualisasi per pelanggan
    if os.path.exists(RAW_DATA_FILE):
        from preprocessing import load_dataset
        data["raw"] = load_dataset(RAW_DATA_FILE)

    # Daftar prioritas: skor deteksi pada seluruh window meter populasi 2025
    # (hasil rank_population.py). Ambang sama dengan halaman lain.
    ranking_file = os.path.join(OUTPUT_DIR, "population_ranking.pkl")
    if os.path.exists(ranking_file):
        with open(ranking_file, "rb") as f:
            rk = pickle.load(f)
        thr = data["detection"]["threshold"] if "detection" in data else None
        data["prioritas"] = build_priority_table(rk, thr)
        # Meter yang dikecualikan aturan register nol (rank_population.py):
        # bukan bagian peringkat, ditampilkan sebagai daftar pemeriksaan register.
        nol = rk.get("meter_register_nol")
        if nol is not None and len(nol):
            data["register_nol"] = nol.copy()

    # Peta ID samaran (M-0001, ...) dari gabungan semua ID meter yang muncul
    # di dashboard, diurutkan, sehingga satu meter selalu mendapat nama yang sama.
    ids = set()
    if "raw" in data:
        ids |= set(data["raw"]["meter_id"].astype(str))
    if "prioritas" in data:
        ids |= set(data["prioritas"]["tabel"]["ID Meter"].astype(str))
    if "register_nol" in data:
        ids |= set(data["register_nol"]["meter_id"].astype(str))
    data["anonim"] = {m: f"M-{i + 1:04d}" for i, m in enumerate(sorted(ids))}
    return data


def anonim_aktif():
    """Mode ID samaran: ?anonim=1 pada URL atau kotak centang di sidebar."""
    return bool(st.session_state.get("anonim", False))


def nama_meter(data, meter_id):
    """ID meter seperti ditampilkan: asli, atau samaran bila mode anonim aktif."""
    meter_id = str(meter_id)
    if anonim_aktif():
        return data["anonim"].get(meter_id, meter_id)
    return meter_id


STATUS_OPTIONS = ["Belum ditindaklanjuti", "Dijadwalkan", "Diperiksa: ada temuan",
                  "Diperiksa: normal", "Diabaikan"]
STATUS_FILE = os.path.join(OUTPUT_DIR, "tindak_lanjut_prioritas.csv")


def build_priority_table(rk, threshold=None):
    """Susun tabel daftar prioritas dari artefak population_ranking.pkl.

    Peringkat mengikuti rank_population.py: skor rata-rata lalu
    persentase hari tertandai (skor > ambang), keduanya dari model
    populasi. Hari tertandai terakhir dihitung dari window per meter.
    """
    per_meter = rk["per_meter"].copy()
    info = rk["info_windows"]
    scores = rk["scores"]
    if isinstance(scores, dict):                       # format lama (dua model)
        scores = scores.get("populasi30", next(iter(scores.values())))
        per_meter = per_meter.rename(columns={
            "rank_populasi30": "rank", "pctl_populasi30": "pctl",
            "mean_pop": "mean_score", "flagrate_pop": "flag_rate"})
    scores = np.asarray(scores, dtype=float)
    if threshold is None:
        threshold = float(rk.get("threshold", np.percentile(scores, 95)))
    flag = scores > threshold

    per_window = pd.DataFrame({
        "meter_id": info["meter_id"].astype(str).to_numpy(),
        "date": pd.to_datetime(info["date"]).dt.date.to_numpy(),
        "flag": flag,
    })
    last_flag = (per_window[per_window["flag"]]
                 .groupby("meter_id")["date"].max().rename("hari_terakhir"))
    periode = (per_window["date"].min(), per_window["date"].max())

    t = pd.DataFrame({
        "Peringkat": per_meter["rank"].astype(int),
        "ID Meter": per_meter["meter_id"].astype(str),
        "Golongan Tarif": per_meter["golongan_tarif"].fillna("-").astype(str),
        "Daya (VA)": per_meter["daya"].map(
            lambda v: "-" if pd.isna(v) else f"{int(v):,}".replace(",", ".")),
        "Hari Dinilai": per_meter["n_windows"].astype(int),
        "Hari Tertandai (%)": (per_meter["flag_rate"] * 100).round(1),
        "Skor Rata-rata": per_meter["mean_score"].astype(float).round(3),
        "Persentil": per_meter["pctl"].round(1),
        "Riwayat P2TL": np.where(per_meter["p2tl_linked"], "Ya", "-"),
        "Kasus 2025/2026": np.where(per_meter["kasus_2025_26"], "Ya", "-"),
        # meter yang dikeluarkan dari data latih: meter set uji (kasus dan kontrol) serta meter beriwayat
        # P2TL lain di luar set uji; artefak lama (sebelum 9 September 2026) belum memiliki kolom ini
        "Dikecualikan dari Latih": np.where(per_meter.get("dikecualikan_latih", per_meter["p2tl_linked"]), "Ya", "-"),
    })
    if "fraksi_slot_kwh_nol" in per_meter.columns:
        # Persentase slot kWh bernilai nol: konteks bagi petugas bila peringkat
        # tinggi disebabkan hari-hari berenergi nol (meter > 50% sudah dikecualikan).
        t["Slot kWh Nol (%)"] = (per_meter["fraksi_slot_kwh_nol"].astype(float) * 100).round(1)
    t = t.merge(last_flag, left_on="ID Meter", right_index=True, how="left")
    t = t.rename(columns={"hari_terakhir": "Hari Tertandai Terakhir"})
    t["Hari Tertandai Terakhir"] = pd.to_datetime(t["Hari Tertandai Terakhir"])
    t["Status Tindak Lanjut"] = STATUS_OPTIONS[0]
    t = t.sort_values("Peringkat").reset_index(drop=True)
    return {"tabel": t, "ambang": float(threshold), "periode": periode,
            "n_window": int(len(per_window))}


def load_status():
    """Status tindak lanjut yang tersimpan (meter_id -> status)."""
    if os.path.exists(STATUS_FILE):
        df = pd.read_csv(STATUS_FILE, dtype=str)
        return dict(zip(df["meter_id"], df["status"]))
    return {}


def save_status(status_map):
    rows = [{"meter_id": m, "status": s, "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M")}
            for m, s in sorted(status_map.items())]
    pd.DataFrame(rows, columns=["meter_id", "status", "diperbarui"]).to_csv(STATUS_FILE, index=False)


def main():
    st.set_page_config(
        page_title="Dashboard Deteksi Anomali Energi Listrik",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Dashboard Deteksi Anomali Konsumsi Energi Listrik")
    st.caption("LSTM Autoencoder | Tugas Akhir - Yudhi Armyndharis")

    data = load_data()

    if not data:
        st.error("Data belum tersedia. Jalankan pipeline terlebih dahulu menggunakan `python main.py`.")
        return

    # Sidebar
    st.sidebar.header("Navigasi")
    # Halaman dapat dipilih lewat URL (?page=overview|timeseries|pelanggan|prioritas)
    # agar tangkapan layar dapat dibuat ulang secara deterministik.
    pages = [
        "📊 Overview & Metrik",
        "📈 Time-Series & Anomali",
        "👤 Per Pelanggan",
        "📋 Daftar Prioritas"
    ]
    slug = {"overview": 0, "timeseries": 1, "pelanggan": 2, "prioritas": 3}
    page = st.sidebar.radio("Pilih Halaman:", pages,
                            index=slug.get(st.query_params.get("page", ""), 0))
    # Mode ID samaran (untuk tangkapan layar naskah): ?anonim=1 atau kotak centang
    if "anonim" not in st.session_state:
        st.session_state["anonim"] = st.query_params.get("anonim", "0") in ("1", "true", "ya")
    st.sidebar.checkbox("Samarkan ID meter (M-0001, ...)", key="anonim",
                        help="Nomor seri meter diganti nama samaran yang tetap; "
                             "status tindak lanjut tetap disimpan dengan ID asli.")

    if page == "📊 Overview & Metrik":
        render_overview(data)
    elif page == "📈 Time-Series & Anomali":
        render_timeseries(data)
    elif page == "👤 Per Pelanggan":
        render_per_customer(data)
    elif page == "📋 Daftar Prioritas":
        render_prioritas(data)


def render_overview(data):
    """Halaman overview dan metrik evaluasi."""
    st.header("Ringkasan Metrik Evaluasi Model")

    if "metrics" in data:
        metrics = data["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", f"{metrics['precision']:.4f}")
        col2.metric("Recall", f"{metrics['recall']:.4f}")
        col3.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        if metrics.get("auc_roc") is not None:
            col4.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

        st.divider()

        # Confusion Matrix
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = metrics["confusion_matrix"]
            fig = px.imshow(
                cm,
                labels=dict(x="Prediksi", y="Label sebenarnya", color="Jumlah"),
                x=["Normal", "Anomali"],
                y=["Normal", "Anomali"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Detail Metrik")
            st.write(f"**True Positive (TP):** {metrics['true_positive']:,}")
            st.write(f"**True Negative (TN):** {metrics['true_negative']:,}")
            st.write(f"**False Positive (FP):** {metrics['false_positive']:,}")
            st.write(f"**False Negative (FN):** {metrics['false_negative']:,}")

            st.divider()
            st.write("**Interpretasi:**")
            st.write(f"- Precision {metrics['precision']:.2f}: "
                     f"{metrics['precision']*100:.0f}% prediksi anomali benar")
            st.write(f"- Recall {metrics['recall']:.2f}: "
                     f"{metrics['recall']*100:.0f}% anomali berhasil terdeteksi")

        # Tampilkan gambar ROC jika ada
        roc_img = os.path.join(FIGURE_DIR, "roc_curve.png")
        if os.path.exists(roc_img):
            st.subheader("Kurva ROC")
            st.image(roc_img)


def render_timeseries(data):
    """Halaman visualisasi time-series dan deteksi anomali."""
    st.header("Visualisasi Time-Series dan Deteksi Anomali")

    if "detection" in data:
        det = data["detection"]
        re_test = det["re_test"]
        threshold = det["threshold"]
        predictions = det["predictions"]

        # Distribusi skor deteksi (residual berarah, Persamaan 3.1) pada data uji
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=re_test, nbinsx=100, name="Skor deteksi data uji",
            marker_color="darkorange", opacity=0.7
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Ambang P95 = {threshold:.4f}")
        fig.update_layout(
            title="Distribusi Skor Deteksi pada Data Uji",
            xaxis_title="Skor deteksi (tebakan model dikurangi kenyataan)",
            yaxis_title="Frekuensi",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot anomali scatter
        fig2 = go.Figure()
        normal_mask = predictions == 0
        anomaly_mask = predictions == 1

        fig2.add_trace(go.Scatter(
            x=np.where(normal_mask)[0], y=re_test[normal_mask],
            mode="markers", name="Normal",
            marker=dict(color="steelblue", size=3, opacity=0.5)
        ))
        fig2.add_trace(go.Scatter(
            x=np.where(anomaly_mask)[0], y=re_test[anomaly_mask],
            mode="markers", name="Anomali",
            marker=dict(color="red", size=6, opacity=0.8)
        ))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig2.update_layout(
            title="Skor Deteksi per Window Harian (data uji)",
            xaxis_title="Indeks window harian",
            yaxis_title="Skor deteksi",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Statistik
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Window Harian", f"{len(predictions):,}")
        col2.metric("Window Tertandai Anomali", f"{predictions.sum():,}")
        col3.metric("Ambang (P95)", f"{threshold:.6f}")


def render_per_customer(data):
    """Halaman analisis per pelanggan."""
    st.header("Analisis Per Pelanggan")

    if "raw" not in data:
        st.warning("Data mentah tidak tersedia untuk visualisasi per pelanggan.")
        return

    df = data["raw"]
    meter_ids = sorted(df["meter_id"].astype(str).unique())

    default_meter = st.query_params.get("meter")   # ?meter=<meter_id asli>
    selected_meter = st.selectbox(
        "Pilih Pelanggan:", meter_ids,
        index=meter_ids.index(default_meter) if default_meter in meter_ids else 0,
        format_func=lambda m: nama_meter(data, m))

    if selected_meter:
        meter_data = df[df["meter_id"].astype(str) == selected_meter].sort_values("timestamp")

        # Time-series plot
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=(
                "Konsumsi Energi (kWh)", "Tegangan (V)",
                "Arus (A)", "Faktor Daya"
            )
        )

        # Titik berlabel P2TL (ground truth, bukan hasil deteksi model)
        normal = meter_data[meter_data["anomaly_label"] == 0]
        anomaly = meter_data[meter_data["anomaly_label"] == 1]

        for col_idx, col_name in enumerate(["kWh", "voltage", "current", "power_factor"]):
            row = col_idx // 2 + 1
            col = col_idx % 2 + 1

            fig.add_trace(go.Scatter(
                x=normal["timestamp"], y=normal[col_name],
                mode="lines", name="Normal",
                line=dict(color="steelblue", width=1),
                opacity=0.6, showlegend=(col_idx == 0)
            ), row=row, col=col)

            if len(anomaly) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly["timestamp"], y=anomaly[col_name],
                    mode="markers", name="Label P2TL (ground truth)",
                    marker=dict(color="orange", size=4),
                    showlegend=(col_idx == 0)
                ), row=row, col=col)

        # Window yang DIDETEKSI model sebagai anomali (arsir merah)
        det_info = data.get("detection", {}).get("test_info")
        n_detected = 0
        if det_info is not None:
            det_meter = det_info[
                (det_info["meter_id"].astype(str) == str(selected_meter))
                & (det_info["prediction"] == 1)
            ]
            n_detected = len(det_meter)
            for _, w in det_meter.head(200).iterrows():
                fig.add_vrect(
                    x0=w["start_time"], x1=w["end_time"],
                    fillcolor="red", opacity=0.10, line_width=0,
                    row="all", col="all"
                )

        fig.update_layout(height=600, title_text=f"Data Pelanggan: {nama_meter(data, selected_meter)}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Arsir merah = window harian yang dideteksi model sebagai anomali "
                   "(periode uji). Titik oranye = label P2TL (ground truth).")

        # Statistik pelanggan
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Pembacaan", f"{len(meter_data):,}")
        col2.metric("Pembacaan Berlabel P2TL", f"{meter_data['anomaly_label'].sum():,}")
        col3.metric("Window Terdeteksi", f"{n_detected:,}")
        col4.metric("Rata-rata kWh", f"{meter_data['kWh'].mean():.3f}")


def render_prioritas(data):
    """Halaman daftar prioritas pemeriksaan (tabel peringkat meter)."""
    st.header("Daftar Prioritas Pemeriksaan")

    if "prioritas" not in data:
        st.warning("Berkas peringkat populasi (output/population_ranking.pkl) belum ada. "
                   "Jalankan `python rank_population.py` terlebih dahulu.")
        return

    pr = data["prioritas"]
    tabel = pr["tabel"].copy()
    status_map = load_status()
    tabel["Status Tindak Lanjut"] = tabel["ID Meter"].map(status_map).fillna(STATUS_OPTIONS[0])
    # ID asli disimpan pada kolom tersembunyi agar status tetap terkait meter
    # yang benar walaupun ID yang ditampilkan adalah samaran.
    tabel["_id_asli"] = tabel["ID Meter"]
    tabel["ID Meter"] = tabel["_id_asli"].map(lambda m: nama_meter(data, m))
    n_nol = len(data["register_nol"]) if "register_nol" in data else 0

    st.caption(
        f"Peringkat seluruh meter populasi AMR {pr['periode'][0].year} yang memiliki window harian lengkap, "
        f"diurutkan berdasarkan skor rata-rata (persentase hari yang ditandai anomali pada ambang {pr['ambang']:.3f} "
        f"sebagai keterangan). Data {pr['periode'][0]} sampai {pr['periode'][1]}, {pr['n_window']:,} window harian. "
        + (f"{n_nol:,} meter yang register kWh-nya hampir selalu nol tidak diberi peringkat (lihat bagian bawah). "
           if n_nol else "")
        + "Kolom Riwayat P2TL menandai meter yang tercatat sebagai kasus P2TL pada sumber data, sedangkan Dikecualikan dari "
          "Latih menandai meter yang dikeluarkan dari data latih (meter set uji, kasus dan kontrol, serta meter beriwayat P2TL "
          "lain di luar set uji). "
        + "Daftar ini adalah keluaran prototipe dan belum divalidasi di lapangan; keputusan pemeriksaan tetap pada petugas."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Meter Dinilai", f"{len(tabel):,}")
    col2.metric("Meter dengan Hari Tertandai", f"{int((tabel['Hari Tertandai (%)'] > 0).sum()):,}")
    col3.metric("Ambang (P95)", f"{pr['ambang']:.4f}")
    col4.metric("Sudah Ditindaklanjuti", f"{int((tabel['Status Tindak Lanjut'] != STATUS_OPTIONS[0]).sum()):,}")

    st.subheader("Filter")
    f1, f2, f3, f4, f5 = st.columns([2, 2, 1, 1, 1])
    cari = f1.text_input("Cari ID meter", value=st.query_params.get("cari", ""))
    tarif_opsi = sorted(tabel["Golongan Tarif"].unique())
    tarif = f2.multiselect("Golongan tarif", tarif_opsi, default=[])
    min_hari = f3.number_input("Minimal hari dinilai", min_value=1, max_value=366, value=7,
                               help="Meter dengan sedikit hari dinilai dapat menempati peringkat tinggi secara kebetulan.")
    hanya_tertandai = f4.checkbox("Hanya yang tertandai", value=True)
    n_teratas = f5.number_input("Tampilkan teratas", min_value=min(10, len(tabel)), max_value=len(tabel),
                                value=min(100, len(tabel)), step=10)

    view = tabel
    if cari:
        view = view[view["ID Meter"].str.contains(cari.strip(), case=False, na=False)]
    if tarif:
        view = view[view["Golongan Tarif"].isin(tarif)]
    view = view[view["Hari Dinilai"] >= int(min_hari)]
    if hanya_tertandai:
        view = view[view["Hari Tertandai (%)"] > 0]
    view = view.sort_values("Peringkat").head(int(n_teratas)).reset_index(drop=True)

    st.subheader(f"Tabel Peringkat ({len(view):,} meter ditampilkan)")
    st.caption("Klik judul kolom untuk mengurutkan. Kolom Status Tindak Lanjut dapat diubah, lalu tekan Simpan.")
    edited = st.data_editor(
        view,
        hide_index=True,
        key="editor_prioritas",
        disabled=[c for c in view.columns if c != "Status Tindak Lanjut"],
        column_order=[c for c in view.columns if c != "_id_asli"],
        column_config={
            "Peringkat": st.column_config.NumberColumn(format="%d"),
            "Daya (VA)": st.column_config.TextColumn(),
            "Hari Dinilai": st.column_config.NumberColumn(format="%d"),
            "Hari Tertandai (%)": st.column_config.NumberColumn(format="%.1f"),
            "Skor Rata-rata": st.column_config.NumberColumn(format="%.3f"),
            "Persentil": st.column_config.NumberColumn(format="%.1f"),
            "Slot kWh Nol (%)": st.column_config.NumberColumn(
                format="%.1f", help="Persentase slot 30 menit yang energinya tercatat nol; "
                                    "meter di atas 50% tidak diberi peringkat."),
            "Hari Tertandai Terakhir": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Status Tindak Lanjut": st.column_config.SelectboxColumn(
                options=STATUS_OPTIONS, required=True),
            "_id_asli": None,
        },
    )

    b1, b2 = st.columns([1, 1])
    if b1.button("Simpan status tindak lanjut"):
        baru = dict(zip(edited["_id_asli"], edited["Status Tindak Lanjut"]))
        status_map.update({m: s for m, s in baru.items() if s != STATUS_OPTIONS[0] or m in status_map})
        save_status(status_map)
        st.success(f"Status tersimpan ke {os.path.basename(STATUS_FILE)} "
                   f"({len(status_map)} meter).")
    b2.download_button(
        "Unduh daftar (CSV)",
        data=edited.drop(columns=["_id_asli"]).to_csv(index=False).encode("utf-8"),
        file_name=f"daftar_prioritas_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    kasus = tabel[tabel["Kasus 2025/2026"] == "Ya"]
    if len(kasus) > 0:
        with st.expander(f"Posisi {len(kasus)} kasus P2TL 2025/2026 dalam peringkat (pemeriksaan wajar)"):
            st.dataframe(
                kasus[["Peringkat", "ID Meter", "Golongan Tarif", "Hari Dinilai",
                       "Hari Tertandai (%)", "Skor Rata-rata", "Persentil"]],
                hide_index=True,
            )
            st.caption("Kasus yang terjadi pada periode data peringkat; posisinya menunjukkan "
                       "seberapa tinggi kasus nyata muncul dalam daftar.")

    if n_nol:
        nol = data["register_nol"].copy()
        with st.expander(f"{n_nol:,} meter dikecualikan aturan register nol (perlu pemeriksaan register/pemasangan, bukan skor model)"):
            st.caption("Lebih dari separuh slot energi meter-meter ini tercatat nol sepanjang periode data. "
                       "Energi nol yang terus-menerus adalah tanda register tidak mencatat, bukan pola konsumsi, "
                       "sehingga meter ini tidak diberi peringkat dan disarankan diperiksa registernya terlebih dahulu.")
            tampil = pd.DataFrame({
                "ID Meter": nol["meter_id"].astype(str).map(lambda m: nama_meter(data, m)),
                "Golongan Tarif": nol["golongan_tarif"].fillna("-").astype(str),
                "Daya (VA)": nol["daya"].map(lambda v: "-" if pd.isna(v) else f"{int(v):,}".replace(",", ".")),
                "Register Energi": nol["register_energi"].astype(str),
                "Slot kWh Nol (%)": (nol["fraksi_slot_kwh_nol"].astype(float) * 100).round(1),
                "Riwayat P2TL": np.where(nol["p2tl_linked"], "Ya", "-"),
                "Dikecualikan dari Latih": np.where(nol.get("dikecualikan_latih", nol["p2tl_linked"]), "Ya", "-"),
            })
            st.dataframe(tampil, hide_index=True)


if __name__ == "__main__":
    main()
