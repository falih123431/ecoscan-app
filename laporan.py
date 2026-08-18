"""
laporan.py - Modul pembuat laporan PDF EcoScan
Membuat berkas PDF berisi data setoran sampah pengguna.
Membutuhkan pustaka fpdf2:  !pip install fpdf2 -q
"""

from fpdf import FPDF
from datetime import datetime


# ============================================================
# PEMBERSIH TEKS
# ------------------------------------------------------------
# Font bawaan PDF hanya mendukung karakter Latin-1, sehingga
# tanda baca khusus dan emoji perlu diganti terlebih dahulu.
# ============================================================
def _t(teks):
    ganti = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u00b7": "-",
    }
    teks = str(teks)
    for a, b in ganti.items():
        teks = teks.replace(a, b)
    return teks.encode("latin-1", "replace").decode("latin-1")


def _hitung_per_jenis(riwayat):
    """Menghitung jumlah setoran dan total berat untuk setiap jenis sampah."""
    hasil = {}
    for r in riwayat:
        k = r["kategori"]
        if k not in hasil:
            hasil[k] = {"jumlah": 0, "berat": 0.0}
        hasil[k]["jumlah"] += 1
        hasil[k]["berat"] += float(r["berat_kg"] or 0)
    return hasil


def _buat_pdf(html, orientasi="P"):
    """Menghasilkan berkas PDF dari potongan HTML."""
    pdf = FPDF(orientation=orientasi, unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.write_html(html)
    return bytes(pdf.output())


# ============================================================
# LAPORAN SETORAN PENGGUNA
# ============================================================
def laporan_setoran_user(user, riwayat):
    """
    Membuat PDF laporan setoran milik satu pengguna.
    Memuat identitas beserta tempat tinggal, dan seluruh waktu setornya.
    """
    dicetak = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    per_jenis = _hitung_per_jenis(riwayat)
    total_berat = sum(float(r["berat_kg"] or 0) for r in riwayat)

    baris = ""
    for i, r in enumerate(riwayat, start=1):
        baris += (
            "<tr>"
            f"<td align='center'>{i}</td>"
            f"<td>{_t(r['tanggal'])}</td>"
            f"<td>{_t(r.get('alamat_setor') or '-')}</td>"
            f"<td>{_t(r['kategori'])}</td>"
            f"<td align='center'>{float(r['berat_kg'] or 0):.1f}</td>"
            f"<td align='center'>{float(r['keyakinan']):.2f}%</td>"
            "</tr>"
        )
    if not baris:
        baris = "<tr><td colspan='6' align='center'>Belum ada setoran tercatat</td></tr>"

    baris_jenis = ""
    for jenis, d in sorted(per_jenis.items()):
        baris_jenis += (
            "<tr>"
            f"<td>{_t(jenis)}</td>"
            f"<td align='center'>{d['jumlah']} kali</td>"
            f"<td align='center'>{d['berat']:.1f} kg</td>"
            "</tr>"
        )
    if not baris_jenis:
        baris_jenis = "<tr><td colspan='3' align='center'>-</td></tr>"

    html = f"""
<h1 align="center">LAPORAN SETORAN SAMPAH</h1>
<p align="center"><b>EcoScan</b> - Sistem Klasifikasi Sampah Organik dan Non-Organik</p>
<hr>

<h3>Data Penyetor</h3>
<table width="100%">
  <tr><td width="35%">Nama Lengkap</td><td width="65%">: {_t(user['nama_lengkap'])}</td></tr>
  <tr><td width="35%">Username</td><td width="65%">: {_t(user['username'])}</td></tr>
  <tr><td width="35%">Nomor HP</td><td width="65%">: {_t(user['no_hp'])}</td></tr>
  <tr><td width="35%">Terdaftar Sejak</td><td width="65%">: {_t(user['tanggal_daftar'])}</td></tr>
</table>

<h3>Rincian Setoran</h3>
<table border="1" width="100%">
  <thead>
    <tr bgcolor="#DDDDDD">
      <th width="6%" align="center">No</th>
      <th width="20%" align="center">Waktu Setor</th>
      <th width="30%" align="center">Alamat Lokasi Sampah</th>
      <th width="20%" align="center">Jenis Sampah</th>
      <th width="12%" align="center">Berat (kg)</th>
      <th width="12%" align="center">Keyakinan</th>
    </tr>
  </thead>
  {baris}
</table>

<h3>Ringkasan</h3>
<table border="1" width="100%">
  <thead>
    <tr bgcolor="#DDDDDD">
      <th width="45%" align="center">Jenis Sampah</th>
      <th width="27%" align="center">Jumlah Setoran</th>
      <th width="28%" align="center">Total Berat</th>
    </tr>
  </thead>
  {baris_jenis}
  <tr>
    <td width="45%"><b>TOTAL</b></td>
    <td width="27%" align="center"><b>{len(riwayat)} kali</b></td>
    <td width="28%" align="center"><b>{total_berat:.1f} kg</b></td>
  </tr>
</table>

<br>
<p><i>Laporan ini dihasilkan secara otomatis oleh sistem EcoScan
pada {_t(dicetak)}. Jenis sampah ditentukan menggunakan model
Convolutional Neural Network dengan arsitektur MobileNetV2,
sedangkan berat dan alamat lokasi sampah diisi oleh penyetor
pada saat menyimpan setoran.</i></p>
"""
    return _buat_pdf(html, orientasi="L")


# ============================================================
# LAPORAN SELURUH SETORAN (ADMIN)
# ============================================================
def laporan_setoran_admin(semua_riwayat, rekap):
    """Membuat PDF rekap seluruh setoran dari semua pengguna."""
    dicetak = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    baris = ""
    for i, r in enumerate(semua_riwayat, start=1):
        baris += (
            "<tr>"
            f"<td align='center'>{i}</td>"
            f"<td>{_t(r['tanggal'])}</td>"
            f"<td>{_t(r['nama_lengkap'])}</td>"
            f"<td>{_t(r.get('alamat_setor') or '-')}</td>"
            f"<td>{_t(r['kategori'])}</td>"
            f"<td align='center'>{float(r['berat_kg'] or 0):.1f}</td>"
            "</tr>"
        )
    if not baris:
        baris = "<tr><td colspan='6' align='center'>Belum ada setoran tercatat</td></tr>"

    html = f"""
<h1 align="center">REKAP SELURUH SETORAN SAMPAH</h1>
<p align="center"><b>EcoScan</b> - Sistem Klasifikasi Sampah Organik dan Non-Organik</p>
<hr>

<h3>Ringkasan Sistem</h3>
<table width="100%">
  <tr><td width="45%">Jumlah Pengguna Terdaftar</td><td width="55%">: {rekap['jumlah_user']}</td></tr>
  <tr><td width="45%">Total Setoran</td><td width="55%">: {rekap['total'] or 0} kali</td></tr>
  <tr><td width="45%">Setoran Sampah Organik</td><td width="55%">: {rekap['organik'] or 0} kali</td></tr>
  <tr><td width="45%">Setoran Sampah Non-Organik</td><td width="55%">: {rekap['non_organik'] or 0} kali</td></tr>
  <tr><td width="45%">Total Berat Tercatat</td><td width="55%">: {float(rekap['total_berat'] or 0):.1f} kg</td></tr>
</table>

<h3>Daftar Setoran</h3>
<table border="1" width="100%">
  <thead>
    <tr bgcolor="#DDDDDD">
      <th width="5%" align="center">No</th>
      <th width="18%" align="center">Waktu Setor</th>
      <th width="17%" align="center">Penyetor</th>
      <th width="30%" align="center">Alamat Lokasi Sampah</th>
      <th width="19%" align="center">Jenis Sampah</th>
      <th width="11%" align="center">Berat (kg)</th>
    </tr>
  </thead>
  {baris}
</table>

<br>
<p><i>Rekap ini dihasilkan secara otomatis oleh sistem EcoScan pada {_t(dicetak)}.
Alamat yang tercantum merupakan lokasi sampah yang diisi pengguna pada saat menyimpan setoran.</i></p>
"""
    return _buat_pdf(html, orientasi="L")
