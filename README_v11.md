# Menjalankan pipeline utama (v11, dirapikan 30 Agustus 2026; tambalan aturan register nol 8 September 2026)

Peta lengkap berkas yang dipakai naskah: README_KODE.md.
Panduan menjalankan dan menjelaskan ke dosen: ../bimbingan/Panduan_Kode_TA_2026-08-30.docx.

Isi pipeline:
- build_training_population.py: window harian populasi AMR 2025, pemetaan register
  energi per meter, min_count, k_n, aturan register nol (meter dengan lebih dari
  separuh slot kWh bernilai nol dikeluarkan) -> data/population_windows.pkl.
- main.py --population: praproses, pelatihan (validasi 10% meter, dipisah per meter),
  deteksi (ambang P95), evaluasi -> models/, output/.
- evaluasi_sintetis.py: penekanan energi 10/30/50/70% pada window kontrol -> Tabel 4.2.
- analisis_v11.py: analisis kasus pre/post dan hitungan kasus/meter yang pernah tertandai
  -> Tabel 4.3, Lampiran C (bagian interval kepercayaan dan normalisasi prospektif
  di dalamnya tidak dipakai naskah).
- rank_population.py: peringkat meter populasi -> output/population_ranking.csv;
  meter yang dikecualikan aturan register nol -> output/meter_register_nol.csv.
- dashboard.py: halaman Overview, Time-Series, Per Pelanggan, Daftar Prioritas
  (?anonim=1 menyamarkan ID meter untuk tangkapan layar).

Cara menjalankan (dari folder code/, venv sudah ada):

    bash run_v11.sh

Urutan: set_grid 30 -> build populasi -> main.py --population (latih ulang, sekitar
14 menit) -> evaluasi_sintetis -> analisis_v11 -> rank_population. Log di logs/v11_*.log,
arsip lengkap di experiments/<tanggal>_v11_groupsplit_register/.

Perhatian: pelatihan memakai bobot awal acak, sehingga angka evaluasi bisa sedikit
berbeda dari naskah. Jalan resmi dilakukan di Google Colab lewat ../TA_CRISP-DM_Colab.ipynb
(arsip resmi 8 Sep 2026 dengan aturan register nol: experiments/20260908_0225_v11_colab/;
arsip 6 Sep 2026 tanpa aturan itu: experiments/20260906_0914_v11_colab/).

Dashboard:

    ./venv/bin/streamlit run dashboard.py

Halaman: ?page=overview | timeseries | pelanggan | prioritas

Berkas lama yang tidak dipakai naskah dipindahkan ke experiments/_lama/ (lihat README di sana).
