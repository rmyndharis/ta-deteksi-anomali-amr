# Peta kode yang dipakai naskah TA (v37, 13 September 2026, notebook lama dipindahkan)

## Angka resmi yang dipakai naskah: jalan Colab 10 September 2026 (`experiments/20260910_1709_v11_colab`)

Populasi 2.586 meter (2.781 - 105 data uji/riwayat P2TL - 90 interval 60 menit); aturan register nol mengeluarkan 165 meter, 66 meter tanpa hari lengkap gugur; data latih 119.622 window dari 2.355 meter (validasi 236 meter / 12.044 window, pelatihan 107.578 window / 2.119 meter). Pelatihan berhenti epoch 47, bobot terbaik epoch 37 (train loss 0,4511, val loss 0,4785), 1.597,9 detik GPU T4. Ambang P95 = 0,7149 (naskah: 0,715). Uji kasus-kontrol 4.959 window (668 pre / 658 post / 3.633 kontrol): CM TN 4.125 / FP 166 / FN 625 / TP 43; Precision 0,206, Recall 0,064, F1 0,098, AUC 0,579 (IK klaster 0,51-0,66); 10 dari 32 kasus dan 38 dari 85 meter kontrol pernah tertandai; AUC tingkat meter 0,613; pembanding deviasi-median dua kanal AUC 0,609 (tingkat meter 0,638); normalisasi masa lalu (4.246 window) 0,579 -> 0,522; sintetis AUC 0,618/0,767/0,854/0,910; hibrida SMOTE AUC 0,626 (P 0,201, R 0,413, F1 0,271), indikator saja 0,627, tanpa penyeimbangan 0,574; peringkat 2.453 meter, 1.458 pernah tertandai, 172 meter register nol. Seluruh angka ini dihitung ulang dari berkas arsip pada audit 12 September 2026 dan cocok dengan naskah.

Notebook: `../colab/TA_CRISP_DM_Colab.ipynb` = salinan notebook Colab (`MyDrive/TA/TA_CRISP-DM_Colab.ipynb`) berisi keluaran jalan resmi 10 September; sel Markdown diselaraskan dengan istilah naskah pada 12 September (kode dan keluaran tidak diubah). `../colab/TA_CRISP-DM_Colab.ipynb` = sumber 9 September tanpa keluaran (lama). `Penjelasan_Sistem_TA.ipynb` di folder ini = notebook penjelasan LAMA yang keluarannya dari arsip 8 September (ambang 0,758, AUC 0,587); jangan dibuka saat sidang.

## Perubahan 12 September 2026 (pemeriksaan tambahan, tanpa pelatihan ulang)
- `suku_profil_harapan.py` (Lampiran A.22): memecah skor deteksi (Persamaan 3.1) menjadi suku deviasi dua kanal
  (pembanding) dan suku profil harapan (sumbangan LSTM-AE) pada 4.959 window uji arsip resmi, lalu menilai AUC
  tiap suku (bootstrap klaster 2.000 ulangan, seed 0, protokol analisis_v11.py). Hasil ke
  `experiments/analysis_v11/suku_profil_harapan.json` (salinan: `../colab/angka_tambahan_0912_suku_profil.json`):
  AUC skor 0,579 (IK 0,51-0,66), deviasi 0,609 (0,53-0,69), suku profil harapan 0,444 (0,38-0,51); tingkat meter
  0,613 / 0,638 / 0,428; rata-rata suku profil harapan pre 0,175, post 0,182, kontrol 0,236; korelasi suku dengan
  deviasi -0,37, dengan kWh aktual +0,36, tegangan -0,30, faktor daya +0,30. Dipakai naskah pada Subbab 4.8 (sejak 20 Sep 2026; sebelumnya 4.9)
  (pendalaman mengapa skor deteksi di bawah pembanding, permintaan pembimbing) dan paparan.
- Daftar Pustaka naskah: [10] tesis Ashari ITB diganti Taruna dkk. 2025 (IEEE Access, akses terbuka), [20] artikel
  Jokar dkk. 2016 diganti disertasi Jokar 2015 (UBC, akses terbuka; Bab 5 = artikel). Docstring evaluasi_sintetis.py
  hanya menyebut nomor [11] dan [20], jadi tidak berubah.

## Perubahan 11 September 2026 (komentar saja, tidak ada perubahan perilaku)
Penilaian menyeluruh naskah menemukan beberapa komentar yang isinya bertabrakan dengan naskah.
Hanya teks komentar yang diubah, jadi tidak perlu menjalankan ulang apa pun dan seluruh angka tetap.
- `config.py`: komentar ambang tidak lagi menyebut P95 sebagai nilai konvensional standar, karena Bab II
  sengaja tidak mengklaim itu dan menyebutnya kompromi antara penurunan sedang yang terlewat dan hari
  normal yang harus ditinjau petugas.
- `config.py`: dua kata "terbukti" diganti, yaitu alasan grid 30 menit (sekarang memakai alasan Bab III,
  data 15 menit dapat digabung sedangkan 60 menit tidak dapat dipecah) dan autoencoder identitas yang
  "dapat", bukan "terbukti", menyalin input.
- `evaluasi_sintetis.py`: rujukan pada docstring dibetulkan dari [19] menjadi [20], dan ditambahkan catatan
  bahwa Tabel 4.2 hanya melaporkan T1, T4, dan T5 sedangkan T2 dan T3 tersimpan di `tabel_sintetis.csv`.
- `dashboard.py`: komentar daftar prioritas tidak lagi menyebut "Model B", istilah yang tidak ada di naskah.
- Ke-21 listing Lampiran A pada naskah sudah disamakan lagi baris demi baris dengan berkas di sini.
- Sore harinya, empat berkas praproses di `data/` disalin ulang dari arsip resmi karena masih berisi
  jalan lama: `processed_sequences.pkl` dan `population_windows.pkl` tadinya 119.639 window, sekarang
  119.622; `population_scalers.pkl` dan `population_register_map.csv` juga ikut disamakan. Isi `output/`
  dan `models/` memang sudah cocok sejak awal, jadi tidak ada angka naskah yang berubah. Sesudah ini
  seluruh folder kerja berasal dari satu arsip yang sama, aman untuk demo maupun jalan ulang.


## Perubahan 10 September 2026 (eksekusi resmi baru + pembersihan identitas)
- Eksekusi resmi naskah sekarang `experiments/20260910_1709_v11_colab` (Colab T4, koreksi tipe data
  `id_pelanggan` sudah ikut). select_meters memberi "Dikecualikan P2TL: 173 meter" dan 2.586 meter populasi;
  pelatihan 47 epoch (terbaik 37), 1.597,9 detik, ambang P95 = 0,7149.
- `output/`, `models/`, `data/fitur_harian_hibrida.csv`, `experiments/analysis_v11/`, `experiments/analysis_hibrida/`
  dan `../gambar/` disalin dari arsip itu. Arsip 8 dan 9 September tetap disimpan sebagai riwayat.
- Kolom identitas pelanggan (nama, alamat, telepon, no KTP) dibuang dari keempat berkas `../data/AP2T/*.csv`
  sebelum eksekusi ini; jumlah baris tidak berubah dan tidak ada kode yang memakai kolom itu.
- Seluruh angka Bab III, Bab IV, Bab V, kedua abstrak, Tabel C.1, dan Tabel D.2 naskah sudah diperbarui
  dari arsip ini (225 suntingan, 11 September 2026 dini hari UTC). Catatan koreksi tipe data pada
  pengantar Lampiran dihapus karena koreksinya kini sudah termasuk dalam eksekusi resmi.
- Kontaminasi tingkat pelanggan sesudah koreksi: 10 meter pengganti milik 10 pelanggan kasus uji
  (528 window pasca penertiban), turun dari 12 meter / 545 window pada eksekusi 9 September.

## Perubahan 9 September 2026 malam (koreksi setelah jalan resmi, tanpa melatih ulang)
- build_training_population.select_meters dan rank_population.baca_master: master meter dibaca dengan
  `dtype={"id_mtr": str, "id_pelanggan": str}`. Sebelumnya kolom `id_pelanggan` terbaca sebagai bilangan pecahan
  (160 sel kosong), sehingga `astype(str)` menghasilkan bentuk `...0` yang tidak pernah cocok dengan `IDPEL` pada
  `p2tl_2025_alll.csv`. Akibatnya pengecualian tingkat pelanggan untuk laporan P2TL 2025 diam-diam kosong pada jalan
  resmi 9 September (log Colab "Dikecualikan P2TL: 171 meter", seluruhnya dari ID meter). Dampaknya persis dua meter
  pengganti pelanggan kasus 2025 (17 window pelatihan) yang sudah diungkap naskah sebagai keterbatasan. Dengan koreksi
  ini select_meters memberi "Dikecualikan P2TL: 173 meter" dan 2.586 meter populasi (jalan resmi: 171 dan 2.588).
  Arsip `experiments/20260909_0220_v11_colab` dan seluruh angka naskah TIDAK diubah (tidak ada pelatihan ulang);
  pada jalan ulang berikutnya kedua meter itu ikut dikecualikan dan angka Bab IV harus diperbarui utuh dari arsip baru.
  (SUDAH DILAKSANAKAN: lihat bagian 10 September 2026 di atas.)
- Keterkaitan tingkat pelanggan yang lebih luas (bukan bug, keputusan protokol): 12 meter populasi adalah meter pengganti
  milik pelanggan 12 kasus uji (545 window, satu meter di validasi) dan 24 meter (termasuk 12 tadi, 1.223 window) milik
  pelanggan pada rekap 115 kasus, semuanya dipasang setelah tanggal laporan kasus sehingga datanya pasca penertiban.
  Naskah (keterbatasan butir 3) menyebut pemisahan latih dan uji berlaku per meter.
- Lampiran A naskah memuat kedua baris koreksi di atas; Tabel D.2 F1 dirinci dari keluaran `preprocessing.py --check`.
- Untuk jalan ulang Colab berikutnya: unggah `../colab/TA_patch_20260909c.zip` (build_training_population.py,
  rank_population.py, README_KODE.md) ke MyDrive di samping TA_patch_20260909b.zip; tambalan bernama terakhir yang menang.


## Perubahan 9 September 2026 malam (tanggapan penilai ketiga, tanpa melatih ulang)
- rank_population.py: penanda peringkat dibedakan. `p2tl_linked` = meter yang tercatat sebagai kasus P2TL pada sumber data
  (41 dari 2.453 meter ter-ranking); kolom baru `dikecualikan_latih` = meter set uji (kasus dan kontrol) yang dikeluarkan
  dari data latih (96 meter; 55 di antaranya meter kontrol). Sebelumnya `p2tl_linked` memakai himpunan kedua, sehingga
  meter kontrol ikut tertulis "riwayat P2TL". Mode `python3 rank_population.py --hanya-penanda` menghitung ulang ketiga
  kolom penanda pada output yang sudah ada tanpa menskor ulang (tanpa TensorFlow); dijalankan pada `output/` dan pada
  `experiments/20260909_0220_v11_colab/output/` (population_ranking.csv/pkl, meter_register_nol.csv; versi lama di
  `../tmp/backup_20260909_rerun/penanda_lama/`). Skor, urutan, dan kolom lain tidak berubah.
- dashboard.py: kolom baru "Dikecualikan dari Latih" di Daftar Prioritas dan tabel register nol (semula dinamai "Meter Set Uji",
  diganti karena 22 dari 96 meter itu bukan anggota set uji: komposisinya 55 meter kontrol set uji, 19 meter kasus set uji,
  22 meter lain dari rekap P2TL); keterangan kolom di caption halaman. Gambar 4.3 naskah
  (`../gambar/dashboard/gambar_4_3_daftar_prioritas.png`) diambil ulang dengan `../gambar/dashboard/shot_dashboard.py` (Playwright).
- analisis_v11.py bagian 1c: aturan urutan daftar prioritas dihitung dua kali, tanpa filter (117 unit) dan dengan filter
  bawaan tampilan dashboard (minimal 7 hari dinilai dan hanya meter tertandai: 47 unit, 9 kasus dan 38 kontrol; skor
  rata-rata menemukan 4, 6, 7 kasus pada 6, 12, 23 teratas). `tabel_topk_peringkat.csv` dan kunci `peringkat_topk` di
  `ringkasan_v11.json` (arsip dan `experiments/analysis_v11/`) diperbarui dengan menjalankan
  `analisis_v11.py --skip-prospektif` pada `detection_results.pkl` arsip yang sama; tabel lain identik dengan arsip.
- `../data/AP2T/dataset_lp_p2tl_fasa.csv` dibuat ulang dengan buat_dataset_fasa.py (versi nilai mutlak faktor daya):
  69.645 record berubah dari 0 menjadi |PF|; fitur_indikator.py dari berkas itu menghasilkan `data/fitur_harian_hibrida.csv`
  yang identik dengan arsip Colab. Cadangan berkas lama: `../tmp/backup_20260909_rerun/data_AP2T/`.
- Untuk jalan ulang Colab berikutnya: unggah `../colab/TA_patch_20260909b.zip` (ketiga berkas di atas) ke MyDrive dan
  tambahkan pada sel patch notebook, seperti TA_patch_20260909.zip.
- Naskah: rujukan peraturan P2TL dihapus (Daftar Pustaka 20 entri, Gers dkk. 2000 sebagai [14]); Bab I sampai V dipadatkan
  (Tabel 4.5 menjadi Tabel D.2; gambar arus/tegangan per fasa dan halaman Per Pelanggan menjadi Gambar B.9 dan B.10;
  Daftar Prioritas menjadi Gambar 4.3). Alat penyuntingan naskah: `../tmp/alat_revisi_20260909/`.

## Perubahan 9 September 2026 (rerun ketiga, tanggapan reviewer)
- Faktor daya bertanda negatif (konvensi arah daya reaktif, seri EMK1, 21% record set uji) dipakai NILAI MUTLAKNYA, bukan disetel nol: preprocessing.load_dataset, build_training_population.slots_to_frame, buat_dataset_fasa.py. Seluruh angka Bab IV berasal dari jalan ulang Colab setelah perubahan ini.
- rank_population.py dan dashboard.py: urutan Daftar Prioritas memakai skor rata-rata (persentase hari tertandai hanya keterangan), sesuai ukuran yang dievaluasi pada tingkat meter.
- analisis_v11.py: pembanding deviasi dua kanal (kWh dan arus, kolom bl2/baseline2) pada semua protokol, dan tabel_topk_peringkat.csv (kasus yang ditemukan pada 6/12/23 unit teratas menurut empat aturan urutan).
- klasifikasi_hibrida.py: kolom auc_rata_fold (rata-rata AUC di dalam tiap fold) di samping AUC gabungan out-of-fold.

JALAN RESMI = Google Colab (GPU T4), 9 September 2026 pukul 02.20 UTC (08.14 sampai 09.20 WIB), lewat notebook
`MyDrive/TA/TA_CRISP-DM_Colab.ipynb` (salinan: `../colab/TA_CRISP-DM_Colab.ipynb`; berisi keluaran:
`../colab/TA_CRISP-DM_Colab_hasil_20260909.ipynb`) dengan tambalan `TA_patch_20260908.zip` lalu
`TA_patch_20260909.zip` (faktor daya nilai mutlak, urutan skor rata-rata, pembanding dua kanal, AUC per fold,
top-K). Arsipnya: `experiments/20260909_0220_v11_colab/` (diunduh dari `MyDrive/TA_hasil/`); isi `models/`,
`output/`, `experiments/analysis_v11/`, `experiments/analysis_hibrida/`, `data/` (peta register, scaler,
window populasi, processed_sequences, fitur hibrida), dan `../gambar/` sudah disamakan dengan arsip itu
(cadangan keadaan sebelumnya di `../tmp/backup_20260909_rerun/`).
Arsip jalan sebelumnya: `experiments/20260908_0225_v11_colab/` (8 Sep, faktor daya negatif masih disetel nol)
dan `experiments/20260906_0914_v11_colab/` (6 Sep, tanpa aturan register nol).
Naskah yang memakai angka ini: `../naskah/220401010272_TA_2026090706.docx` (disunting di tempat 9 Sep 2026;
cadangan di `../tmp/backup_20260909_rerun/naskah/`). Ringkasan angka arsip: `../colab/angka_arsip_0909.json`
(dibuat `../colab/ambil_angka_arsip.py <folder_arsip>`) dan `../colab/angka_tambahan_0909.json`
(`../colab/hitung_tambahan.py`). Tanggapan reviewer: `../naskah/Tanggapan_Reviewer_2026-09-09.docx`.

Panduan lengkap (cara menjalankan dan cara menjelaskan ke dosen):
`../bimbingan/Panduan_Kode_TA_2026-08-30.docx`

## Berkas yang dipakai naskah (22 listing Lampiran A)

Jalur utama, dijalankan `bash run_v11.sh` (45-60 menit, melatih ulang):

| Berkas | Tugas | Di naskah |
|---|---|---|
| config.py | semua pengaturan (fitur, grid 30 menit, LSTM 64-32-16, P95) | Tabel A.1, Subbab 3.3, 3.4, 3.5 |
| set_grid.py | menyetel interval 30 menit | Subbab 3.3 |
| build_training_population.py | window harian populasi AMR 2025 (119.622 window, 2.355 meter pada jalan resmi 10 Sep), aturan register kWh-only dan aturan register nol (`zero_kwh_rule`: 165 dari 2.586 meter dengan >50% slot kWh nol dikeluarkan; lihat CATATAN_RERUN_REGISTER.md) | Subbab 3.2, Tabel 3.1 |
| main.py --population | memanggil visualize -> preprocessing -> train -> detect -> evaluate | Bab III, Subbab 4.1-4.3 |
| visualize.py | gambar eksplorasi data; `koma()`/`sumbu_koma()` = label koma desimal untuk semua gambar | Gambar B.1, B.2, B.4, B.5 |
| preprocessing.py | cleaning, grid 30 menit, normalisasi per meter, window 48 langkah; `--check` = F1 | Subbab 3.3 |
| model.py | LSTM Autoencoder, kanal kWh+arus ditutup (masked) | Subbab 3.4, Gambar 3.2 |
| train.py | pelatihan, validasi 10% meter, early stopping, training_loss.png | Subbab 4.2, Gambar B.12 |
| detect.py | skor = profil harapan - aktual, ambang P95 = 0,715 (jalan resmi 10 Sep) | Subbab 3.5, 4.3 |
| evaluate.py | confusion matrix, Precision, Recall, F1, AUC, ROC | Tabel 4.1, Gambar 4.1 |
| evaluasi_sintetis.py | penekanan energi 10/30/50/70% (baris T1 yang dipakai) | Tabel 4.2 |
| analisis_v11.py | analisis kasus pre/post, kasus & kontrol yang pernah tertandai (bagian CI/prospektif tidak dipakai naskah) | Subbab 4.3, 4.5, Tabel 4.3, Lampiran C |
| artefak.py | pembantu pemuat model/ambang/set uji | - |
| rank_population.py | peringkat 2.453 meter populasi (2.355 latih + 98 meter di luar latih yang berwindow lengkap); diproses per kelompok 300 meter agar muat RAM Colab (hasil identik); 172 meter yang dikecualikan aturan register nol -> output/meter_register_nol.csv | Subbab 4.7, Gambar 4.2 |
| dashboard.py | Streamlit 4 halaman; `?anonim=1` = ID meter samaran (M-0001, ...) untuk tangkapan layar; kolom Slot kWh Nol (%) dan daftar meter yang dikecualikan | Subbab 3.8, 4.7, Gambar 4.2, B.7, B.8, B.10, Tabel D.1, D.2 |

Jalur hibrida, dijalankan `bash run_hibrida.sh` (3-5 menit, hasil selalu sama, seed 42):

| Berkas | Tugas | Di naskah |
|---|---|---|
| buat_dataset_fasa.py | dataset kasus-kontrol per fasa | Subbab 3.2 |
| fitur_indikator.py | 4 indikator harian + skor model | Subbab 3.6, Tabel 3.2 |
| klasifikasi_hibrida.py | regresi logistik x (tanpa / random oversampling / SMOTE), CV 5-fold per meter; nomor fold disimpan di oof_hibrida.csv | Subbab 4.6, Tabel 4.4, Gambar B.11 |
| gambar_kasus_fasa.py | arus/tegangan per fasa A073, A081, A093, A107 | Gambar B.9 |
| cek_variasi_fold.py | pemeriksaan tambahan: validasi silang per meter diulang 30 seed (SMOTE), median dan rentang AUC; hasil experiments/analysis_hibrida/variasi_fold_30seed.csv (sejak 8 Sep dihitung di Colab sebagai langkah [5/5] run_hibrida.sh dan sel 5.5 notebook; jalan resmi 10 Sep: hibrida median 0,62, rentang 0,48-0,67) | Subbab 4.6, Lampiran A.20 |
| suku_profil_harapan.py | pemeriksaan tambahan: AUC suku deviasi dan suku profil harapan dari skor tersimpan (tanpa pelatihan ulang) | Subbab 4.8, Lampiran A.22 |

Asal data (dijalankan sekali di kantor, tidak diulang): `../data/AP2T/buat_dataset_lp_p2tl.py`
dan `export_queries.sql` -> dataset_lp_p2tl.csv, lp_anomali.csv, lp_kontrol.csv, export_amr/lp_2025MM.csv.

## Angka jalan Colab 8 September 2026 (arsip lama 20260908_0225, TIDAK dipakai naskah; angka resmi ada di bagian atas)

Peta register: 1.0.1.29 pada 2.202 meter, 1.0.2.29 pada 386 meter (dari 2.588 meter populasi);
aturan register nol mengeluarkan 165 meter (90 pada 1.0.1.29, 75 pada 1.0.2.29), 66 meter tanpa
hari lengkap gugur -> 119.639 window dari 2.357 meter (validasi 236 meter, 11.821 window).
Pelatihan berhenti epoch 38, bobot terbaik epoch 28 (train loss 0,44, val loss 0,58), 22 menit T4.
Ambang P95 0,758; uji kasus-kontrol AUC 0,587 (CI klaster 0,52-0,66), precision 0,250,
recall 0,072, F1 0,112 (CM TN 4.147 / FP 144 / FN 620 / TP 48); 192 window tertandai;
10 dari 32 kasus dan 36 dari 85 meter kontrol pernah tertandai; AUC tingkat meter 0,60;
baseline deviasi-median AUC 0,599; sintetis AUC 0,615-0,908; hibrida SMOTE AUC 0,612
(precision 0,187, recall 0,473), indikator saja 0,615, tanpa penyeimbangan 0,577;
peringkat 2.453 meter, 1.406 tertandai, 172 meter dikecualikan aturan register nol.
Lingkungan: Python 3.13.15, TensorFlow 2.20.0, pandas 2.2.3, scikit-learn 1.6.1,
imbalanced-learn 0.14.2, numpy 2.1.3, GPU T4.

Angka jalan 6 September (tanpa aturan register nol, untuk pembanding): ambang 0,825; AUC 0,585,
precision 0,220, recall 0,058, F1 0,092 (CM 4.153/138/629/39); hibrida SMOTE AUC 0,608;
peringkat 2.605 meter, 1.457 tertandai; 124.329 window dari 2.507 meter.

## Rerun kedua (aturan register nol) di Colab: cara menjalankan (SUDAH dilakukan 8 Sep 2026)

1. Buat zip tambalan dari folder TA: `python3 buat_zip_patch.py` -> `TA_patch_20260908.zip`
   (hanya berkas kode yang berubah, sekitar 0,1 MB; TA_full.zip 1,3 GB di Drive TIDAK perlu diunggah ulang).
2. Unggah ke Google Drive: `MyDrive/TA_patch_20260908.zip` dan notebook baru `MyDrive/TA/TA_CRISP-DM_Colab.ipynb`
   (timpa yang lama; salinan di `../TA_CRISP-DM_Colab.ipynb`).
3. Buka notebook di Colab, Runtime -> Change runtime type -> T4 GPU, lalu Runtime -> Run all.
   Sel 0 mencetak "Tambalan diterapkan" dan "kode tambalan aktif"; kalau tidak, zip tambalan belum ada di Drive.
   Lama jalan sekitar satu jam. Sel terakhir mengarsipkan ke `MyDrive/TA_hasil/<tanggal>_v11_colab/`.
4. Salin arsip itu ke `experiments/<tanggal>_v11_colab/`, lalu samakan `models/`, `output/`,
   `experiments/analysis_v11/`, `experiments/analysis_hibrida/`, `data/population_register_map.csv`,
   dan `data/fitur_harian_hibrida.csv` dengan isi arsip (seperti yang dilakukan 6 September).
5. Yang dicek di arsip baru: `lingkungan.txt` memuat `ZERO_KWH_MAX_FRACTION=0.5`;
   `population_register_map.csv` punya kolom `status_populasi`; `output/meter_register_nol.csv` ada;
   di `output/population_ranking.csv` tidak ada meter dengan `fraksi_slot_kwh_nol` > 0,5.
6. Semua angka Bab IV, Tabel 3.1/3.2, Lampiran C, dan gambar diperbarui dari arsip baru (sekali jalan, tidak dicampur).

## Berkas lama yang tidak dipakai naskah

Catatan 30 Agustus 2026: 12 skrip lama, model .h5 Juni, hasil SHAP dan pembanding pernah dipindahkan
ke experiments/_lama/; folder itu dan arsip experiments/ lain (grid 60, populasi lama, eksplorasi fitur,
backup kode v10, backup hibrida v1) sudah tidak ada lagi di folder ini.

13 September 2026: `Penjelasan_Sistem_TA.ipynb` (notebook penjelasan lama, keluaran arsip 8 September,
ambang 0,758, AUC 0,587) dipindahkan dari code/ ke experiments/_lama/ (lihat README_lama.md di sana)
atas rekomendasi audit akhir (M-12), agar tidak terbuka keliru saat sidang. `buat_bundel_colab.py`
(6 September) hanya melayani notebook lama itu dan tidak dipakai naskah.

## Sebelum sidang: jalankan tahap hibrida di Mac ini

imbalanced-learn belum terpasang di venv, jadi tabel hibrida yang ada sekarang berasal dari
jalan di mesin lain. Pasang lalu jalankan, dan cocokkan tabel_hibrida.csv dengan Tabel 4.4:

    ./venv/bin/pip install imbalanced-learn
    bash run_hibrida.sh
    grep Hibrida experiments/analysis_hibrida/tabel_hibrida.csv

## Perintah yang aman untuk demo (tidak melatih ulang)

    cd ~/Documents/Privates/TUGAS-AKHIR/code
    ./venv/bin/python preprocessing.py --check ../data/AP2T/dataset_lp_p2tl.csv
    ./venv/bin/python -c "import pickle; print(pickle.load(open('output/evaluation_metrics.pkl','rb')))"
    ./venv/bin/python evaluasi_sintetis.py
    bash run_hibrida.sh
    ./venv/bin/streamlit run dashboard.py      # http://localhost:8501  ?page=overview|timeseries|pelanggan|prioritas

Jangan jalankan `run_v11.sh` atau `Run all` notebook Colab menjelang sidang: pelatihan LSTM
memakai bobot awal acak, angka naskah berasal dari arsip experiments/20260910_1709_v11_colab/
(arsip sebelumnya: experiments/20260909_0220_v11_colab/, 20260908_0225_v11_colab/, dan 20260906_0914_v11_colab/ tanpa aturan register nol;
experiments/20260829_1002_v11_groupsplit_register/ aturan register lama).
