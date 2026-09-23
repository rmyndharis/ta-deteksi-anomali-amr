# Tahap klasifikasi hibrida (29 Agustus 2026, versi ringkas)

Menindaklanjuti saran bimbingan (penyeimbangan data latih: random oversampling, SMOTE)
dan temuan bahwa sebagian pelanggaran K2 hanya terlihat pada data per fasa.
Versi ringkas untuk naskah S1: satu pengklasifikasi (regresi logistik), tiga perlakuan
(tanpa, random oversampling, SMOTE), empat indikator, satu pembagian 5-fold per meter.

Alur:
1. `buat_dataset_fasa.py`: `data/AP2T/lp_anomali.csv` + `lp_kontrol.csv` ->
   `data/AP2T/dataset_lp_p2tl_fasa.csv` (36 kasus, 88 kontrol, tegangan/arus per fasa;
   baris, label, dan kWh identik dengan `dataset_lp_p2tl.csv`).
2. `fitur_indikator.py`: empat indikator per hari (fasa_mati, teg_hilang, rasio_register,
   utilisasi) digabung dengan skor window LSTM-AE dari `output/detection_results.pkl`
   -> `data/fitur_harian_hibrida.csv` (4.959 window set uji yang sama dengan Bab IV).
3. `klasifikasi_hibrida.py`: himpunan fitur AE / Indikator / Hibrida x perlakuan
   tanpa / random_over / smote, regresi logistik, StratifiedGroupKFold 5-fold per meter.
   Keluaran: `tabel_hibrida.csv`, `tabel_cakupan_indikator.csv`, `ringkasan_hibrida.json`,
   `gambar_roc_hibrida.png`, `gambar_efek_penyeimbangan.png` di `experiments/analysis_hibrida/`.
4. `gambar_kasus_fasa.py`: arus/tegangan harian per fasa kasus A073, A081, A093, A107.

Menjalankan semuanya (dari folder code/, venv sudah ada):

    ./venv/bin/pip install imbalanced-learn
    bash run_hibrida.sh

Versi lengkap sebelumnya (8 indikator, random forest, 5 perlakuan, 5 seed) disimpan di
`experiments/backup_hibrida_v1/` (kode) dan `experiments/_to_delete/analysis_hibrida_lama_8indikator/`
(hasil; folder _to_delete boleh dihapus). Eksplorasi awal ada di
`experiments/eksplorasi_20260829_fitur_lintas/`.
