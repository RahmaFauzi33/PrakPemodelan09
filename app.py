import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LOAD DATASET TIME-SERIES
# ==========================================================
@st.cache_data
def load_timeseries():
    ts = pd.read_csv("used_cars_timeseries.csv")
    ts = ts.sort_values("age")
    ages = ts["age"].values
    prices = ts["price"].values
    return ts, ages, prices

ts, ages, prices = load_timeseries()


# ==========================================================
# MODEL ODE & METODE EULER
# ==========================================================
def model_penyusutan(V, t, k):
    return -k * V

def euler_method(func, y0, t_points, params):
    y = np.zeros(len(t_points))
    y[0] = y0
    h = t_points[1] - t_points[0]
    
    for i in range(len(t_points) - 1):
        slope = func(y[i], t_points[i], *params)
        y[i+1] = y[i] + h * slope
    
    return y


# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("📉 Simulasi Penyusutan Nilai Aset Menggunakan Metode Euler")
st.write("""
Aplikasi ini memvisualisasikan penyusutan nilai aset berdasarkan data harga mobil bekas 
yang telah diubah menjadi deret waktu (time-series). Model matematika yang digunakan:

### dV/dt = -kV

Gunakan slider di sidebar untuk mengubah parameter model.
""")

st.sidebar.header("🔧 Pengaturan Parameter Model")

# Slider Parameter
default_V0 = prices[0] if len(prices) > 0 else 50000

V0 = st.sidebar.number_input("Nilai Awal Aset (V0)", min_value=1000.0, value=float(default_V0))
k = st.sidebar.slider("Konstanta Penyusutan (k)", 0.001, 1.0, 0.12)
h = st.sidebar.slider("Step Size Euler (h)", 0.01, 2.0, 0.5)
duration = st.sidebar.slider("Durasi Simulasi (tahun)", 1, int(max(ages)), int(max(ages)))


# ==========================================================
# SIMULASI EULER & SOLUSI EKSAK
# ==========================================================
t_points = np.arange(0, duration + h, h)

# Euler
V_euler = euler_method(model_penyusutan, V0, t_points, (k,))

# Solusi Eksak
V_exact = V0 * np.exp(-k * t_points)


# ==========================================================
# HITUNG MSE TERHADAP DATA ASLI
# ==========================================================
sim_interp = np.interp(ages, t_points, V_euler)
mse = np.mean((sim_interp - prices)**2)


# ==========================================================
# PLOT GRAFIK
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data asli
ax.scatter(ages, prices, color='black', label="Data Asli (Time-Series)", zorder=5)

# Euler
ax.plot(t_points, V_euler, 'o--', label=f"Euler (h={h})", zorder=4)

# Solusi eksak (opsional)
ax.plot(t_points, V_exact, label="Solusi Eksak", linewidth=2, zorder=3)

ax.set_xlabel("Usia Mobil (tahun)")
ax.set_ylabel("Harga Aset (Rp)")
ax.set_title("Simulasi Penyusutan Nilai Aset (Euler vs Data Asli)")
ax.grid(True)
ax.legend()

st.pyplot(fig)


# ==========================================================
# METRIK ERROR
# ==========================================================
st.subheader("📊 Mean Squared Error (MSE)")
st.write(f"**MSE antara Data Asli dan Simulasi Euler:** `{mse:.2f}`")

st.info("""
Geser slider untuk mencari kombinasi parameter yang membuat hasil simulasi Euler paling sesuai dengan data.
""")
