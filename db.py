"""
db.py - Modul database EcoScan
Semua urusan database ditaruh di sini supaya app.py tidak terlalu panjang.
"""

import sqlite3
import hashlib
import os
import secrets
from datetime import datetime

# ============================================================
# LOKASI DATABASE
# ------------------------------------------------------------
# PENTING: disimpan di Google Drive, BUKAN di folder Colab.
# Kalau disimpan di Colab, seluruh data hilang saat sesi berakhir.
# ============================================================
DB_PATH = "/data/ecoscan.db" if os.path.isdir("/data") else "ecoscan.db"

def koneksi():
    """Membuka koneksi ke database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # supaya hasil query bisa diakses pakai nama kolom
    return conn


# ============================================================
# KEAMANAN KATA SANDI
# ------------------------------------------------------------
# Kata sandi TIDAK disimpan apa adanya, tetapi diacak (hash)
# menggunakan PBKDF2 dengan salt acak untuk setiap pengguna.
# ============================================================
def buat_hash(password, salt=None):
    """Mengubah kata sandi menjadi hash. Mengembalikan (hash, salt)."""
    if salt is None:
        salt = os.urandom(16).hex()
    hash_hasil = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()
    return hash_hasil, salt


def cek_password(password, hash_tersimpan, salt):
    """Memeriksa apakah kata sandi yang dimasukkan cocok."""
    hash_baru, _ = buat_hash(password, salt)
    return hash_baru == hash_tersimpan


# ============================================================
# PEMBUATAN TABEL
# ============================================================
def init_db():
    """Membuat tabel bila belum ada, lalu menyiapkan akun admin bawaan."""
    conn = koneksi()
    c = conn.cursor()

    # --- Tabel users ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username       TEXT    NOT NULL UNIQUE,
            password_hash  TEXT    NOT NULL,
            salt           TEXT    NOT NULL,
            nama_lengkap   TEXT    NOT NULL,
            no_hp          TEXT,
            role           TEXT    NOT NULL DEFAULT 'user',
            aktif          INTEGER NOT NULL DEFAULT 1,
            tanggal_daftar TEXT    NOT NULL
        )
    """)

    # --- Tabel riwayat klasifikasi ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            kategori  TEXT    NOT NULL,
            keyakinan REAL    NOT NULL,
            berat_kg  REAL    DEFAULT 0,
            alamat_setor TEXT,
            nama_file TEXT,
            tanggal   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # --- Tabel sesi login ---
    # Menyimpan token agar pengguna tidak keluar sendiri ketika
    # koneksi peramban terputus sesaat atau halaman dimuat ulang.
    c.execute("""
        CREATE TABLE IF NOT EXISTS sesi (
            token   TEXT    PRIMARY KEY,
            user_id INTEGER NOT NULL,
            dibuat  TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()

    # --- Migrasi: tambahkan kolom alamat_setor pada database lama ---
    c.execute("PRAGMA table_info(riwayat)")
    kolom_riwayat = [baris[1] for baris in c.fetchall()]
    if "alamat_setor" not in kolom_riwayat:
        c.execute("ALTER TABLE riwayat ADD COLUMN alamat_setor TEXT")
        conn.commit()

    # --- Akun admin bawaan (hanya dibuat sekali) ---
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if c.fetchone()[0] == 0:
        h, s = buat_hash("admin123")
        c.execute("""
            INSERT INTO users
                (username, password_hash, salt, nama_lengkap, no_hp, role, tanggal_daftar)
            VALUES (?, ?, ?, ?, ?, 'admin', ?)
        """, ("admin", h, s, "Administrator", "-",
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

    conn.close()


# ============================================================
# FUNGSI AKUN
# ============================================================
def daftar_user(username, password, nama_lengkap, no_hp):
    """Mendaftarkan pengguna baru. Mengembalikan (berhasil, pesan)."""
    if len(username.strip()) < 4:
        return False, "Username minimal 4 karakter."
    if len(password) < 6:
        return False, "Kata sandi minimal 6 karakter."
    if not nama_lengkap.strip():
        return False, "Nama lengkap wajib diisi."

    conn = koneksi()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
    if c.fetchone():
        conn.close()
        return False, "Username sudah digunakan, silakan pilih yang lain."

    h, s = buat_hash(password)
    c.execute("""
        INSERT INTO users
            (username, password_hash, salt, nama_lengkap, no_hp, role, tanggal_daftar)
        VALUES (?, ?, ?, ?, ?, 'user', ?)
    """, (username.strip(), h, s, nama_lengkap.strip(), no_hp.strip(),
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return True, "Pendaftaran berhasil. Silakan masuk."


def login_user(username, password):
    """Memeriksa kredensial. Mengembalikan (data_user_atau_None, pesan)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    baris = c.fetchone()
    conn.close()

    if baris is None:
        return None, "Username tidak ditemukan."
    if not cek_password(password, baris["password_hash"], baris["salt"]):
        return None, "Kata sandi salah."
    if baris["aktif"] == 0:
        return None, "Akun ini sedang dinonaktifkan. Hubungi administrator."

    return dict(baris), "Berhasil masuk."


def ambil_semua_user():
    """Mengambil daftar seluruh pengguna (untuk halaman admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT id, username, nama_lengkap, no_hp, role, aktif, tanggal_daftar
        FROM users ORDER BY tanggal_daftar DESC
    """)
    hasil = [dict(r) for r in c.fetchall()]
    conn.close()
    return hasil


def ubah_status_user(user_id, aktif):
    """Mengaktifkan (1) atau menonaktifkan (0) sebuah akun."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("UPDATE users SET aktif = ? WHERE id = ? AND role = 'user'",
              (1 if aktif else 0, user_id))
    conn.commit()
    conn.close()


def hapus_user(user_id):
    """Menghapus akun pengguna beserta seluruh riwayatnya."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("DELETE FROM riwayat WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM users WHERE id = ? AND role = 'user'", (user_id,))
    conn.commit()
    conn.close()


# ============================================================
# FUNGSI RIWAYAT KLASIFIKASI
# ============================================================
def simpan_riwayat(user_id, kategori, keyakinan, berat_kg, alamat_setor, nama_file):
    """Menyimpan satu hasil klasifikasi beserta alamat lokasi setoran."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        INSERT INTO riwayat (user_id, kategori, keyakinan, berat_kg, alamat_setor, nama_file, tanggal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, kategori, keyakinan, berat_kg, alamat_setor, nama_file,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def ambil_riwayat_user(user_id):
    """Mengambil riwayat milik satu pengguna."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("SELECT * FROM riwayat WHERE user_id = ? ORDER BY tanggal DESC", (user_id,))
    hasil = [dict(r) for r in c.fetchall()]
    conn.close()
    return hasil


def ambil_semua_riwayat():
    """Mengambil seluruh riwayat beserta nama pemiliknya (untuk halaman admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT r.*, u.username, u.nama_lengkap, u.no_hp
        FROM riwayat r JOIN users u ON r.user_id = u.id
        ORDER BY r.tanggal DESC
    """)
    hasil = [dict(r) for r in c.fetchall()]
    conn.close()
    return hasil


def hapus_riwayat(riwayat_id):
    """Menghapus satu baris riwayat setoran (dipakai admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("DELETE FROM riwayat WHERE id = ?", (riwayat_id,))
    conn.commit()
    conn.close()


def hapus_semua_riwayat():
    """Menghapus seluruh riwayat setoran dari semua pengguna (dipakai admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("DELETE FROM riwayat")
    conn.commit()
    conn.close()


def rekap_user(user_id):
    """Menghitung rekap klasifikasi milik satu pengguna."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*)                                                        AS total,
            SUM(CASE WHEN kategori LIKE '%Organik' AND kategori NOT LIKE '%Non%' THEN 1 ELSE 0 END) AS organik,
            SUM(CASE WHEN kategori LIKE '%Non-Organik%' THEN 1 ELSE 0 END)  AS non_organik,
            IFNULL(SUM(berat_kg), 0)                                        AS total_berat
        FROM riwayat WHERE user_id = ?
    """, (user_id,))
    hasil = dict(c.fetchone())
    conn.close()
    return hasil


def rekap_global():
    """Menghitung rekap keseluruhan sistem (untuk halaman admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
    jumlah_user = c.fetchone()[0]
    c.execute("""
        SELECT
            COUNT(*)                                                        AS total,
            SUM(CASE WHEN kategori LIKE '%Organik' AND kategori NOT LIKE '%Non%' THEN 1 ELSE 0 END) AS organik,
            SUM(CASE WHEN kategori LIKE '%Non-Organik%' THEN 1 ELSE 0 END)  AS non_organik,
            IFNULL(SUM(berat_kg), 0)                                        AS total_berat
        FROM riwayat
    """)
    hasil = dict(c.fetchone())
    hasil["jumlah_user"] = jumlah_user
    conn.close()
    return hasil

def rekap_per_jenis(user_id):
    """Rekap per jenis sampah: jumlah setoran dan total beratnya."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT kategori,
               COUNT(*)                 AS jumlah,
               IFNULL(SUM(berat_kg), 0) AS total_berat
        FROM riwayat
        WHERE user_id = ?
        GROUP BY kategori
        ORDER BY kategori
    """, (user_id,))
    hasil = [dict(r) for r in c.fetchall()]
    conn.close()
    return hasil


def rekap_per_jenis_global():
    """Rekap per jenis sampah untuk seluruh pengguna (halaman admin)."""
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT kategori,
               COUNT(*)                 AS jumlah,
               IFNULL(SUM(berat_kg), 0) AS total_berat
        FROM riwayat
        GROUP BY kategori
        ORDER BY kategori
    """)
    hasil = [dict(r) for r in c.fetchall()]
    conn.close()
    return hasil


# ============================================================
# FUNGSI SESI LOGIN
# ============================================================
def buat_sesi(user_id):
    """Membuat token sesi baru untuk pengguna yang berhasil masuk."""
    token = secrets.token_urlsafe(24)
    conn = koneksi()
    c = conn.cursor()
    c.execute("INSERT INTO sesi (token, user_id, dibuat) VALUES (?, ?, ?)",
              (token, user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return token


def ambil_user_dari_token(token):
    """Mengembalikan data pengguna berdasarkan token sesi, atau None."""
    if not token:
        return None
    conn = koneksi()
    c = conn.cursor()
    c.execute("""
        SELECT u.* FROM sesi s JOIN users u ON s.user_id = u.id
        WHERE s.token = ? AND u.aktif = 1
    """, (token,))
    baris = c.fetchone()
    conn.close()
    return dict(baris) if baris else None


def hapus_sesi(token):
    """Menghapus token sesi (dipakai saat pengguna keluar)."""
    if not token:
        return
    conn = koneksi()
    c = conn.cursor()
    c.execute("DELETE FROM sesi WHERE token = ?", (token,))
    conn.commit()
    conn.close()
