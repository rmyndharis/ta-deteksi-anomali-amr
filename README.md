# ta-deteksi-anomali-amr

Kode program Tugas Akhir "Penerapan Masked LSTM Autoencoder dalam Sistem Deteksi Anomali
Konsumsi Energi Listrik untuk Mendukung Penertiban Pemakaian Tenaga Listrik"
(Yudhi Armyndharis, PJJ Informatika, Universitas Siber Asia, 2026).

## Isi

Jalur utama, dijalankan lewat `run_v11.sh`:
`config.py`, `set_grid.py`, `build_training_population.py`, `main.py`, `preprocessing.py`,
`model.py`, `train.py`, `detect.py`, `evaluate.py`, `visualize.py`, `evaluasi_sintetis.py`,
`analisis_v11.py`, `artefak.py`, `rank_population.py`.

Tahap klasifikasi hibrida, dijalankan lewat `run_hibrida.sh`:
`buat_dataset_fasa.py`, `fitur_indikator.py`, `klasifikasi_hibrida.py`, `gambar_kasus_fasa.py`,
`cek_variasi_fold.py`.

Lainnya: `suku_profil_harapan.py` (penguraian skor deteksi), `dashboard.py` (prototipe dashboard
Streamlit), `buat_bundel_colab.py` (menyusun bundel untuk Google Colab), `requirements.txt`.
Peta berkas dan urutan menjalankannya ada di `README_KODE.md`, catatan tahap hibrida di
`README_hibrida.md`, catatan versi pipeline di `README_v11.md` dan `CATATAN_RERUN_REGISTER.md`.

## Data

Data tidak disertakan. Pipeline membaca ekspor load profile meter AMR, data induk meter, dan
rekapitulasi P2TL dari folder `data/` yang tidak ada di repositori ini karena memuat data
pelanggan. Data itu dipakai atas surat persetujuan tertulis dari perusahaan penyedia tenaga
listrik tempat penelitian dilakukan, dan hanya hasil olahan yang telah dianonimkan yang
disajikan pada naskah.

## Menjalankan

```
pip install -r requirements.txt
bash run_v11.sh        # praproses, pelatihan, deteksi, evaluasi, pemeringkatan populasi
bash run_hibrida.sh    # tahap klasifikasi hibrida
streamlit run dashboard.py
```

Eksekusi resmi yang angkanya dipakai pada naskah dilakukan di Google Colab dengan GPU T4
(notebook `TA_CRISP-DM_Colab.ipynb`, 10 September 2026):
https://colab.research.google.com/drive/1QtED9rc8R_wGKZyM2zZQlqN3m9cDULgc
