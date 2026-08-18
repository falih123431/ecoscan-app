"""
app.py - EcoScan
Sistem klasifikasi sampah organik dan non-organik berbasis CNN (MobileNetV2)
dilengkapi akun pengguna, riwayat klasifikasi, dan halaman admin.

Falih Setyo Ghani - Universitas Gunadarma
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import db
import laporan

st.set_page_config(
    page_title="EcoScan - Klasifikasi Sampah",
    page_icon="\u267b\ufe0f",
    layout="centered"
)

@st.cache_resource
def _siapkan_db():
    """Menyiapkan tabel sekali saja per sesi, bukan setiap kali halaman dimuat ulang."""
    db.init_db()
    return True


_siapkan_db()

# ============================================================
# THEME - CSS (eco / nature, lively)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Quicksand:wght@500;600;700&display=swap');

.stApp {
    background: linear-gradient(160deg, #eafbea 0%, #f4fbf0 35%, #eef8e8 100%);
    color: #1f3a2e;
}

header[data-testid="stHeader"] {
    background: linear-gradient(160deg, #eafbea 0%, #f4fbf0 100%) !important;
}

header[data-testid="stHeader"] * {
    color: #1b4332 !important;
}

* {
    font-family: 'Quicksand', sans-serif;
}

h1, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    color: #1b4332 !important;
}

p, span, div, label, li {
    color: #1f3a2e;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #d8f3dc 0%, #eafbea 100%);
    border-right: 1px solid #b7e4c7;
}

@keyframes grow {
    0% { transform: scale(0.98); }
    50% { transform: scale(1.01); }
    100% { transform: scale(0.98); }
}

.glass-card {
    background: #ffffff;
    border: 1px solid #d8ecd4;
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(45, 106, 79, 0.08);
    animation: grow 6s ease-in-out infinite;
}

.glass-card-organik {
    background: linear-gradient(135deg, #f0fbf1 0%, #e3f7e6 100%);
    border-color: #95d5b2;
    box-shadow: 0 8px 28px rgba(64, 145, 108, 0.18);
}

.glass-card-nonorganik {
    background: linear-gradient(135deg, #fff8ec 0%, #fdf1dc 100%);
    border-color: #f4c98b;
    box-shadow: 0 8px 28px rgba(191, 138, 61, 0.18);
}

.result-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #1b4332;
}

.result-conf {
    font-family: 'Poppins', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #2d6a4f;
}

.scan-badge {
    display: inline-block;
    background: #d8f3dc;
    color: #2d6a4f;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 1rem;
}

div[data-testid="stMetricValue"] {
    color: #2d6a4f;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #74c69d, #40916c);
}

.stFileUploader {
    border-radius: 18px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #f0fbf1 0%, #e3f7e6 100%) !important;
    border: 2px dashed #95d5b2 !important;
    border-radius: 18px !important;
}

div[data-testid="stFileUploaderDropzone"] section {
    background: transparent !important;
}

.main-banner {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(135deg, #d8f3dc 0%, #f0fbf1 100%);
    border: 1px solid #b7e4c7;
    border-radius: 22px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 6px 20px rgba(45, 106, 79, 0.08);
}

.main-banner-icon {
    font-size: 2.6rem;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(-4deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

.main-banner-text h3 {
    margin: 0 !important;
    font-size: 1.3rem !important;
}

.main-banner-text p {
    margin: 0;
    color: #4a7a5e;
    font-size: 0.92rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #40916c, #2d6a4f);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: transform 0.15s ease;
}

div.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #2d6a4f, #1b4332);
    color: white;
}

.footer-note {
    text-align: center;
    color: #84a98c;
    font-size: 0.8rem;
    letter-spacing: 1px;
    margin-top: 2rem;
}

.login-box {
    background: #ffffff;
    border: 1px solid #d8ecd4;
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(45, 106, 79, 0.08);
}

.user-chip {
    display: inline-block;
    background: #d8f3dc;
    color: #2d6a4f;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_waste_classifier.h5"
    )


# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_image(pil_image):
    """Preprocessing HARUS sama persis dengan tahap training."""
    model = load_model()
    img = pil_image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)          # skala 0-255 -> -1..1
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    return float(prediction[0][0])


# ============================================================
# PEMBUAT PDF (hasilnya disimpan sementara)
# ------------------------------------------------------------
# Tanpa penyimpanan sementara, berkas PDF akan disusun ulang
# setiap kali ada interaksi pada halaman sehingga terasa lambat.
# ============================================================
@st.cache_data(show_spinner=False)
def pdf_user(_kunci, _user, _riwayat):
    return laporan.laporan_setoran_user(_user, _riwayat)


@st.cache_data(show_spinner=False)
def pdf_admin(_kunci, _semua, _rekap):
    return laporan.laporan_setoran_admin(_semua, _rekap)


def kunci_riwayat(daftar):
    """Penanda sederhana agar PDF dibuat ulang hanya bila datanya berubah."""
    if not daftar:
        return "kosong"
    return f"{len(daftar)}-{daftar[0]['tanggal']}-{daftar[-1]['tanggal']}"


# ============================================================
# ISI INFORMASI TINDAK LANJUT (dipakai layar dan PDF)
# ============================================================
INFO = {
    "Sampah Organik": {
        "deskripsi": (
            "Sampah organik adalah sampah yang berasal dari makhluk hidup dan "
            "dapat terurai secara alami oleh mikroorganisme.\n\n"
            "**Contoh:** kulit pisang, sisa buah, sayuran, sisa makanan, daun kering."
        ),
        "dampak": """
| Jenis | Estimasi Waktu Terurai |
|---|---|
| Kulit buah | 1-2 minggu |
| Sisa sayuran | 2-4 minggu |
| Daun kering | 1-2 bulan |

Jika dikelola dengan benar, sampah organik justru bermanfaat dan tidak mencemari lingkungan.
        """,
        "pengelolaan": """
1. Pisahkan dari sampah non-organik sejak awal
2. Kumpulkan di wadah/tempat sampah organik (biasanya berwarna hijau)
3. Olah menjadi kompos menggunakan komposter sederhana atau lubang biopori
4. Kompos yang jadi bisa dipakai untuk menyuburkan tanaman
        """,
        "rekomendasi": (
            "Mulai kumpulkan sampah organik dari rumah untuk dijadikan kompos \u2014 "
            "selain mengurangi timbunan di TPA, hasil kompos juga bisa digunakan "
            "sendiri untuk berkebun."
        ),
    },
    "Sampah Non-Organik": {
        "deskripsi": (
            "Sampah non-organik merupakan sampah yang sulit terurai dan umumnya "
            "dapat didaur ulang.\n\n"
            "**Contoh:** botol plastik, gelas plastik, kaleng, botol kaca, kemasan plastik."
        ),
        "dampak": """
| Jenis | Estimasi Waktu Terurai |
|---|---|
| Plastik | 50-100+ tahun |
| Kaleng aluminium | 80-100 tahun |
| Botol kaca | 1 juta+ tahun |

Sampah non-organik yang tidak dikelola dengan baik berkontribusi pada pencemaran tanah dan air dalam jangka panjang.
        """,
        "pengelolaan": """
1. Pisahkan berdasarkan jenis (plastik, kertas, logam, kaca) jika memungkinkan
2. Bersihkan dari sisa makanan/cairan sebelum dibuang
3. Kumpulkan di tempat sampah non-organik (biasanya berwarna biru/kuning)
4. Salurkan ke bank sampah atau pengepul untuk proses daur ulang
        """,
        "rekomendasi": (
            "Kurangi penggunaan barang sekali pakai, dan biasakan memilah sampah "
            "non-organik agar lebih mudah didaur ulang oleh bank sampah atau "
            "pengepul di sekitar tempat tinggal."
        ),
    },
}


# ============================================================
# SESSION STATE
# ============================================================
for key, default in [
    ("user", None),
    ("analyzed_image", False),
    ("probability", None),
    ("last_file_key", None),
    ("tersimpan", False),
    ("token", None),
    ("minta_hapus_semua", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------
# PEMULIHAN SESI
# Bila koneksi peramban sempat terputus atau halaman dimuat ulang,
# status masuk dipulihkan kembali menggunakan token pada alamat URL.
# ------------------------------------------------------------
if st.session_state.user is None:
    token_url = st.query_params.get("s")
    if token_url:
        data_pulih = db.ambil_user_dari_token(token_url)
        if data_pulih:
            st.session_state.user = data_pulih
            st.session_state.token = token_url


# ============================================================
# HALAMAN MASUK / DAFTAR
# ============================================================
def halaman_masuk():
    st.markdown("""
    <div class="main-banner">
        <div class="main-banner-icon">\U0001f5d1\ufe0f\u267b\ufe0f</div>
        <div class="main-banner-text">
            <h3>Selamat Datang di EcoScan</h3>
            <p>Masuk atau daftar untuk mulai mengenali jenis sampahmu.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_masuk, tab_daftar = st.tabs(["Masuk", "Daftar Akun"])

    with tab_masuk:
        u = st.text_input("Username", key="in_user")
        p = st.text_input("Kata sandi", type="password", key="in_pass")
        if st.button("\U0001f513 Masuk", use_container_width=True):
            if not u or not p:
                st.warning("Username dan kata sandi wajib diisi.")
            else:
                data, pesan = db.login_user(u, p)
                if data:
                    st.session_state.user = data
                    st.session_state.token = db.buat_sesi(data["id"])
                    st.query_params["s"] = st.session_state.token
                    st.rerun()
                else:
                    st.error(pesan)

    with tab_daftar:
        nama = st.text_input("Nama lengkap", key="rg_nama")
        hp = st.text_input("Nomor HP", key="rg_hp")
        u2 = st.text_input("Username", key="rg_user")
        p2 = st.text_input("Kata sandi", type="password", key="rg_pass")
        p3 = st.text_input("Ulangi kata sandi", type="password", key="rg_pass2")
        if st.button("\u2705 Daftar", use_container_width=True):
            if p2 != p3:
                st.error("Kata sandi tidak sama.")
            else:
                ok, pesan = db.daftar_user(u2, p2, nama, hp)
                if ok:
                    st.success(pesan)
                else:
                    st.error(pesan)


# ============================================================
# SIDEBAR
# ============================================================
def sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown("### \U0001f331 EcoScan")
        st.write("Sistem klasifikasi sampah organik dan non-organik menggunakan CNN (Convolutional Neural Network).")
        st.markdown("---")
        st.markdown(f"**{user['nama_lengkap']}**")
        st.markdown(f'<span class="user-chip">@{user["username"]} \u00b7 {user["role"]}</span>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown("---")
        if user["role"] == "user":
            st.markdown("**Cara Pakai**")
            st.write("1. Upload gambar sampah\n2. Klik tombol **Analisis**\n3. Simpan ke riwayat")
            st.markdown("---")
        if st.button("\U0001f6aa Keluar", use_container_width=True):
            db.hapus_sesi(st.session_state.token)
            st.query_params.clear()
            for k in ["user", "probability", "last_file_key", "token"]:
                st.session_state[k] = None
            for k in ["analyzed_image", "tersimpan"]:
                st.session_state[k] = False
            st.rerun()
        st.markdown("---")
        st.caption("\U0001f33f Bersama menjaga lingkungan, satu klasifikasi setiap saat")


# ============================================================
# HALAMAN PENGGUNA
# ============================================================
def halaman_pengguna():
    user = st.session_state.user

    st.markdown("""
    <div class="main-banner">
        <div class="main-banner-icon">\U0001f5d1\ufe0f\u267b\ufe0f</div>
        <div class="main-banner-text">
            <h3>Deteksi Jenis Sampahmu</h3>
            <p>Unggah gambar, dan biarkan AI menentukan apakah sampahmu organik atau non-organik.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_klasifikasi, tab_riwayat = st.tabs(["\U0001f50d Klasifikasi", "\U0001f4ca Riwayat Setoran"])

    # ---------------- TAB KLASIFIKASI ----------------
    with tab_klasifikasi:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**\U0001f4e4 Upload Gambar**")
        uploaded_file = st.file_uploader(
            "Upload gambar sampah",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        image = None
        nama_file = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            nama_file = uploaded_file.name
            st.image(image, caption="Preview gambar", use_container_width=True)

            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_file_key != file_key:
                st.session_state.analyzed_image = False
                st.session_state.probability = None
                st.session_state.tersimpan = False
                st.session_state.last_file_key = file_key

        analyze_clicked = st.button("\U0001f50d Analisis Sekarang", disabled=(image is None))
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_clicked and image is not None:
            with st.spinner("\U0001f50d Menganalisis gambar..."):
                st.session_state.probability = predict_image(image)
            st.session_state.analyzed_image = True
            st.session_state.tersimpan = False

        if image is not None and st.session_state.analyzed_image and st.session_state.probability is not None:
            probability = st.session_state.probability

            if probability < 0.5:
                label = "Sampah Organik"
                icon = "\U0001f7e2"
                confidence = (1 - probability) * 100
                card_class = "glass-card-organik"
            else:
                label = "Sampah Non-Organik"
                icon = "\U0001f535"
                confidence = probability * 100
                card_class = "glass-card-nonorganik"

            st.markdown(f"""
            <div class="glass-card {card_class}">
                <span class="scan-badge">Hasil Analisis \u00b7 {datetime.now().strftime('%H:%M:%S')}</span>
                <div class="result-label">{icon} {label}</div>
                <div class="result-conf">Tingkat Keyakinan: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

            if confidence < 70:
                st.warning(
                    "Sistem kurang yakin dengan hasil ini. Coba gunakan gambar dengan "
                    "objek sampah yang lebih jelas, pencahayaan cukup, dan latar sederhana."
                )

            # ---------- SIMPAN KE RIWAYAT ----------
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f4be Simpan ke Riwayat")
            if st.session_state.tersimpan:
                st.success("Hasil klasifikasi ini sudah tersimpan di riwayat.")
            else:
                berat = st.number_input(
                    "Perkiraan berat sampah (kg)",
                    min_value=0.0, max_value=1000.0, value=1.0, step=0.5
                )
                alamat_setor = st.text_area(
                    "Alamat lokasi sampah",
                    placeholder="Contoh: Jl. Melati No. 12, RT 03/RW 05, Depok",
                    key="alamat_setor_input",
                )
                if st.button("Simpan hasil ini", use_container_width=True):
                    if not alamat_setor.strip():
                        st.warning("Alamat lokasi sampah wajib diisi.")
                    else:
                        db.simpan_riwayat(user["id"], label, confidence, berat,
                                          alamat_setor.strip(), nama_file)
                        st.session_state.tersimpan = True
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            info = INFO[label]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f4d6 Deskripsi")
            st.write(info["deskripsi"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f30d Dampak Lingkungan & Waktu Terurai")
            st.write(info["dampak"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f6e0\ufe0f Cara Pengelolaan")
            st.write(info["pengelolaan"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f4a1 Rekomendasi")
            st.write(info["rekomendasi"])
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB RIWAYAT SETORAN ----------------
    with tab_riwayat:
        rekap = db.rekap_user(user["id"])

        # --- Ringkasan keseluruhan ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f4ca Ringkasan Setoran Saya")
        c1, c2 = st.columns(2)
        c1.metric("Total setoran", f"{rekap['total'] or 0} kali")
        c2.metric("Total berat", f"{rekap['total_berat']:.1f} kg")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Daftar setoran satu per satu ---
        riwayat = db.ambil_riwayat_user(user["id"])
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f4dc Riwayat Setoran")
        if riwayat:
            tampil = [{
                "No": i + 1,
                "Tanggal Setor": r["tanggal"],
                "Jenis Sampah": r["kategori"],
                "Berat (kg)": r["berat_kg"],
                "Alamat Lokasi": r["alamat_setor"] or "-",
                "Keyakinan (%)": round(r["keyakinan"], 2),
                "Berkas Gambar": r["nama_file"],
            } for i, r in enumerate(riwayat)]
            st.dataframe(tampil, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada setoran. Lakukan klasifikasi terlebih dahulu, "
                    "lalu simpan hasilnya ke riwayat.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Unduh laporan PDF ---
        if riwayat:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f4c4 Unduh Laporan Setoran")
            st.write("Laporan memuat data penyetor beserta tempat tinggal, "
                     "waktu setiap setoran, jenis sampah, dan beratnya.")
            try:
                berkas_pdf = pdf_user(kunci_riwayat(riwayat), user, riwayat)
                st.download_button(
                    label="\u2b07\ufe0f Unduh PDF",
                    data=berkas_pdf,
                    file_name=f"Laporan_Setoran_{user['username']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Gagal membuat PDF: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HALAMAN ADMIN
# ============================================================
def halaman_admin():
    st.markdown("""
    <div class="main-banner">
        <div class="main-banner-icon">\U0001f6e1\ufe0f</div>
        <div class="main-banner-text">
            <h3>Halaman Administrator</h3>
            <p>Pantau aktivitas klasifikasi dan kelola akun pengguna EcoScan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_pantau, tab_akun = st.tabs(["\U0001f4ca Pemantauan", "\U0001f465 Kelola Akun"])

    # ---------------- PEMANTAUAN ----------------
    with tab_pantau:
        rekap = db.rekap_global()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f4c8 Statistik Keseluruhan")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pengguna", rekap["jumlah_user"])
        c2.metric("Total Klasifikasi", rekap["total"] or 0)
        c3.metric("Organik", rekap["organik"] or 0)
        c4.metric("Non-Organik", rekap["non_organik"] or 0)
        st.metric("Total berat tercatat (kg)", f"{rekap['total_berat']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

        pj = db.rekap_per_jenis_global()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f9fe Rincian per Jenis Sampah")
        if pj:
            st.dataframe([{
                "Jenis Sampah": j["kategori"],
                "Jumlah Setoran": f"{j['jumlah']} kali",
                "Total Berat (kg)": f"{j['total_berat']:.1f}",
            } for j in pj], use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada data setoran.")
        st.markdown('</div>', unsafe_allow_html=True)

        semua = db.ambil_semua_riwayat()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f4dc Seluruh Riwayat Klasifikasi")
        if semua:
            tampil = [{
                "Tanggal": r["tanggal"],
                "Pengguna": r["nama_lengkap"],
                "Username": r["username"],
                "Alamat Lokasi Sampah": r["alamat_setor"] or "-",
                "Kategori": r["kategori"],
                "Keyakinan (%)": round(r["keyakinan"], 2),
                "Berat (kg)": r["berat_kg"],
            } for r in semua]
            st.dataframe(tampil, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat klasifikasi dari pengguna.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- HAPUS RIWAYAT ----------
        if semua:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f5d1\ufe0f Hapus Riwayat Setoran")

            pilihan = {
                f"{r['tanggal']} - {r['nama_lengkap']} - {r['kategori']} "
                f"({float(r['berat_kg'] or 0):.1f} kg)": r["id"]
                for r in semua
            }
            terpilih = st.selectbox(
                "Pilih riwayat yang akan dihapus",
                list(pilihan.keys()),
                key="pilih_hapus_riwayat",
            )

            konfirmasi = st.checkbox(
                "Saya yakin ingin menghapus riwayat ini",
                key="konfirmasi_hapus_riwayat",
            )
            if st.button("Hapus riwayat terpilih", use_container_width=True):
                if not konfirmasi:
                    st.warning("Centang kotak konfirmasi terlebih dahulu.")
                else:
                    db.hapus_riwayat(pilihan[terpilih])
                    st.success("Riwayat berhasil dihapus.")
                    st.rerun()

            with st.expander("\u26a0\ufe0f Hapus SELURUH riwayat setoran"):
                st.write(
                    "Tindakan ini menghapus seluruh riwayat setoran dari semua "
                    "pengguna dan tidak dapat dibatalkan. Data akun pengguna "
                    "tidak ikut terhapus."
                )
                if not st.session_state.get("minta_hapus_semua", False):
                    if st.button("Hapus seluruh riwayat", use_container_width=True):
                        st.session_state.minta_hapus_semua = True
                        st.rerun()
                else:
                    st.warning(
                        f"Yakin menghapus seluruh {len(semua)} riwayat setoran? "
                        "Tindakan ini tidak dapat dibatalkan."
                    )
                    ya, tidak = st.columns(2)
                    if ya.button("Ya, hapus semua", use_container_width=True):
                        db.hapus_semua_riwayat()
                        st.session_state.minta_hapus_semua = False
                        st.success("Seluruh riwayat setoran telah dihapus.")
                        st.rerun()
                    if tidak.button("Batal", use_container_width=True):
                        st.session_state.minta_hapus_semua = False
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if semua:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("\U0001f4c4 Unduh Rekap Seluruh Setoran")
            try:
                berkas_pdf = pdf_admin(kunci_riwayat(semua), semua, rekap)
                st.download_button(
                    label="\u2b07\ufe0f Unduh PDF",
                    data=berkas_pdf,
                    file_name="Rekap_Seluruh_Setoran_EcoScan.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Gagal membuat PDF: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- KELOLA AKUN ----------------
    with tab_akun:
        daftar = db.ambil_semua_user()
        pengguna = [u for u in daftar if u["role"] == "user"]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("\U0001f465 Daftar Pengguna Terdaftar")
        if not pengguna:
            st.info("Belum ada pengguna terdaftar.")
        else:
            for u in pengguna:
                with st.container(border=True):
                    k1, k2 = st.columns([3, 2])
                    with k1:
                        st.markdown(f"**{u['nama_lengkap']}** (@{u['username']})")
                        st.caption(f"\U0001f4de {u['no_hp']} \u00b7 Daftar: {u['tanggal_daftar']}")
                    with k2:
                        status = "Aktif" if u["aktif"] == 1 else "Nonaktif"
                        st.write(f"Status: **{status}**")
                        b1, b2 = st.columns(2)
                        if u["aktif"] == 1:
                            if b1.button("Nonaktifkan", key=f"off{u['id']}"):
                                db.ubah_status_user(u["id"], False)
                                st.rerun()
                        else:
                            if b1.button("Aktifkan", key=f"on{u['id']}"):
                                db.ubah_status_user(u["id"], True)
                                st.rerun()
                        if b2.button("Hapus", key=f"del{u['id']}"):
                            db.hapus_user(u["id"])
                            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PENENTU HALAMAN
# ============================================================
if st.session_state.user is None:
    halaman_masuk()
else:
    sidebar()
    if st.session_state.user["role"] == "admin":
        halaman_admin()
    else:
        halaman_pengguna()

st.markdown(
    '<p class="footer-note">\U0001f33f Bersama menjaga lingkungan, satu klasifikasi setiap saat \U0001f33f'
    '<br>Falih Setyo Ghani \u00b7 Universitas Gunadarma \u00b7 2026</p>',
    unsafe_allow_html=True
)
