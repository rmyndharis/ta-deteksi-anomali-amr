# Perubahan seleksi register energi dan cara menjalankan ulang

Tanggal: 31 Agustus 2026
Berkas yang diubah: `build_training_population.py` (backup: `build_training_population.py.bak-satuan-20260831`)

## 1. Masalah yang diperbaiki

Sebelumnya, bila register energi kWh sebuah meter kosong, pipeline **mengganti**nya
dengan register daya `1.0.2.25` (kW, daya aktif rata-rata per interval). Ini bukan
penanganan data kosong yang sah dalam CRISP-DM: yang dilakukan bukan menghapus atau
mengimputasi variabel yang sama, melainkan **menukar variabel dengan variabel lain
yang besarannya berbeda**. Pembenaran lama ("faktor konversi konstan diserap
normalisasi robust per meter") mengandaikan kesetaraan yang **tidak dapat
diverifikasi** dari isi ekspor: uji silang terhadap V x I x PF menghasilkan rasio
berserak 0,10-0,62, dan `pfaverage_25` pada seri EM1I bermedian 0,057 yang tidak
berperilaku seperti faktor daya.

Seluruh bagian lain pipeline sudah memakai **penghapusan** sebagai strategi data
kosong (hari yang tidak lengkap 48 pembacaan dibuang, slot berkadensi campuran
di-NaN-kan). Penggantian register adalah satu-satunya tempat yang menyimpang dari
aturan itu. Perubahan ini membuatnya konsisten.

## 2. Perubahan kode

Konstanta baru:

    ENERGY_REGISTER_KWH_ONLY = True

Bila aktif, kandidat register energi dibatasi pada **register kWh saja**
(`1.0.1.29` dan `1.0.2.29`). Register daya `x.25` tidak pernah dipakai sebagai
pengganti. Meter yang register kWh-nya tidak pernah lengkap 48 pembacaan dalam
satu hari **gugur sendiri** pada aturan kelengkapan hari yang sudah ada, bukan
ditambal. Setel `False` untuk kembali ke perilaku lama.

Aturan lama tetap ada di cabang `else` supaya perbandingan masih bisa dilakukan.

## 3. Perkiraan dampak (dihitung dari data mentah, bukan tebakan)

Meter terdampak: 134 memakai register daya, 113 di antaranya masuk data latih
atau validasi (102 latih + 11 validasi). Seri EM1I, H310, HXF3.

Keterisian register pada 113 meter itu (224.310 baris, Jun+Sep+Des 2025):

| Register | Satuan | Terisi non-nol |
|---|---|---|
| 1.0.1.29 | kWh | 0,1% |
| 1.0.1.25 | kW  | 0,1% |
| 1.0.2.29 | kWh | 34,5% |
| 1.0.2.25 | kW  | 84,7% |

Bila dipaksa memakai register kWh, dari 113 meter:

* **21 meter bertahan**, menyumbang sekitar **91 hari lengkap** (window)
* **92 meter gugur** karena register kWh-nya tidak pernah lengkap satu hari pun

Perkiraan pergeseran angka (angka pasti keluar dari log rerun):

| | Sebelum | Perkiraan sesudah |
|---|---|---|
| Meter populasi | 2.507 | sekitar 2.415 |
| Window (latih + validasi) | 124.329 | sekitar 122.900 (turun ~1,2%) |

**117 meter data uji tidak ada yang terdampak** - semuanya seri EMK6, EMK1, LZMD,
W310, LZMG, W318, ASL7, dan tidak satu pun memakai register daya. Jadi susunan
data uji dan seluruh analisis kasus P2TL tidak berubah.

## 4. Cara menjalankan ulang

    cd ~/Documents/Privates/UNSIA/TA/code
    ./venv/bin/pip install imbalanced-learn      # belum terpasang, perlu untuk tahap hibrida
    bash run_v11.sh                              # 25-35 menit, mengarsipkan sendiri
    bash run_hibrida.sh                          # 3-5 menit, harus setelah run_v11.sh

`run_v11.sh` menyalin hasilnya ke `experiments/<tanggal>_v11_groupsplit_register/`,
jadi arsip lama `experiments/20260829_1002_v11_groupsplit_register/` **tidak tertimpa**.

## 5. Yang harus dicek setelah rerun

    grep -E "Meter tanpa register kWh|Window populasi final" logs/v11_02_build.log
    cut -d, -f2 data/population_register_map.csv | sort | uniq -c
    ./venv/bin/python -c "import pickle;print(pickle.load(open('output/evaluation_metrics.pkl','rb')))"
    grep Hibrida experiments/analysis_hibrida/tabel_hibrida.csv

Yang diharapkan pada `population_register_map.csv`: hanya muncul `1.0.1.29` dan
`1.0.2.29`, **tidak ada lagi** `1.0.1.25` maupun `1.0.2.25`.

## 6. HASIL RERUN (Google Colab T4, 6 September 2026)

Peta register: `1.0.1.29` 2.202 meter, `1.0.2.29` 386 meter (EM1I 266, H310 104, HXF3 16);
tidak ada lagi `x.25`. 36 meter tanpa register kWh terisi sama sekali. Berbeda dari
perkiraan Bagian 3, jumlah window/meter populasi TIDAK berubah (371.492 hari lengkap ->
124.329 window dari 2.507 meter): register kWh pada meter yang dulu memakai x.25 ternyata
terisi (termasuk nilai nol) pada slot yang sama, sehingga tidak ada hari yang gugur; yang
berubah adalah nilai kanal kWh-nya, bukan cakupannya. Arsip: experiments/20260906_0914_v11_colab/.

## 7. PERHATIAN sebelum memutuskan rerun (catatan lama)

Pelatihan LSTM memakai bobot awal acak, jadi angka Bab IV akan bergeser
**walaupun tanpa perubahan ini**. Artinya setelah rerun, angka Bab IV harus
diperbarui **seluruhnya dari arsip baru**, tidak boleh dicampur sebagian dari
arsip lama. Yang ikut berubah: Tabel 3.1, Tabel 4.1, Tabel 4.2, Tabel 4.3,
Tabel 4.4, Tabel 4.5, Gambar 4.1-4.6, dan angka peringkat populasi.

Kalau rerun dilakukan mepet sidang, risikonya lebih besar daripada manfaatnya.
Pilihan yang aman: rerun sekarang selagi masih ada waktu, lalu perbarui Bab IV
sekali jalan dari arsip baru.

## 8. TAMBALAN KEDUA: aturan register nol (8 September 2026, rerun Colab kedua)

Temuan pada arsip 6 September: puncak Daftar Prioritas didominasi meter yang register
kWh-nya hampir selalu nol (17 dari 20 teratas memakai `1.0.2.29`; peringkat 2, 7, dan 20
punya 93 sampai 99,6% slot kWh bernilai nol). "Energi nol padahal arus mengalir" pada meter
seperti ini adalah register yang tidak mencatat, bukan pola konsumsi, sehingga skornya
bukan sinyal pelanggaran. Pemeriksaan `tmp/cek_kwh_nol.py` (2.583 meter populasi):
median fraksi slot nol 0,001; 180 meter di atas 0,5 (97 pada `1.0.1.29`, 83 pada `1.0.2.29`);
69 meter di atas 0,9.

Perubahan kode (`build_training_population.py`):

    ZERO_KWH_MAX_FRACTION = 0.5     # None = nonaktif
    zero_kwh_rule(df, reg)          # dipanggil di dalam slots_to_frame

Meter yang lebih dari separuh slot kWh validnya bernilai nol **dikeluarkan** dari populasi
latih dan dari peringkat (`rank_population.py` menyimpannya ke `output/meter_register_nol.csv`).
Peta register mendapat kolom `n_slot_kwh_valid`, `fraksi_slot_kwh_nol`, `status_populasi`
(`dipakai` / `dikecualikan_register_nol`); `population_ranking.csv` mendapat kolom
`fraksi_slot_kwh_nol`. Dashboard: kolom "Slot kWh Nol (%)", daftar meter yang dikecualikan,
dan mode ID samaran (`?anonim=1`). Hari lengkap yang seluruh slotnya nol pada meter yang
tersisa TIDAK dibuang (hanya 0,36% hari lengkap; aturan dibuat satu saja supaya mudah dijelaskan).

Perubahan lain yang ikut dalam tambalan: label gambar berbahasa Indonesia dengan koma desimal
(`visualize.py` menyediakan `koma()` dan `sumbu_koma()`, dipakai `train.py`, `detect.py`,
`evaluate.py`; "Skor deteksi" menggantikan "Reconstruction Error"), kolom `fold` pada
`oof_hibrida.csv` (`klasifikasi_hibrida.py`), dan `cek_variasi_fold.py` menjadi langkah [5/5]
`run_hibrida.sh` serta sel 5.5 notebook.

Cara menjalankan ulang: lihat README_KODE.md bagian "Rerun kedua". Setelah rerun, SEMUA angka
Bab IV, Tabel 3.1/3.2, Lampiran C, dan gambar 4.1-4.6 / B.1-B.9 diambil dari arsip baru.

## 9. HASIL RERUN KEDUA (Google Colab T4, 8 September 2026, arsip experiments/20260908_0225_v11_colab/)

Aturan register nol mengeluarkan 165 dari 2.588 meter populasi (90 pada 1.0.1.29, 75 pada
1.0.2.29; seri terbanyak EM1I 68, H310 28, W318 21, EMK1 15); 66 meter tanpa hari lengkap gugur;
358.287 hari lengkap -> 119.639 window dari 2.357 meter (sebelumnya 124.329 dari 2.507).
Pada peringkat (semua meter master) yang dikecualikan 172 meter (3 berriwayat P2TL) ->
output/meter_register_nol.csv. Peringkat: 2.453 meter, 1.406 tertandai; 20 teratas kini fraksi
slot kWh nol maksimum 39% (median 0,3%), 14 memakai 1.0.2.29 dan sebagian besar < 7 hari dinilai
(tersembunyi pada tampilan bawaan dashboard, yang mulai dari peringkat 6).
Model: 38 epoch, terbaik 28, ambang P95 0,758; AUC 0,587, precision 0,250, recall 0,072,
F1 0,112; 10/32 kasus tertandai; hibrida SMOTE AUC 0,612 (recall 0,473); variasi 30 fold
hibrida 0,49-0,67 (median 0,63). Hari berenergi nol yang tersisa pada data latih: 468 window
(0,39%) pada 78 meter.
