import streamlit as st
import joblib
import numpy as np
import pandas as pd
from obspy import read

st.title("🌍 EthioQuake")
st.write("AI Earthquake Detector for Ethiopia")

model = joblib.load("/content/drive/MyDrive/ethioquake/model.pkl")

uploaded = st.file_uploader("Upload a .mseed seismic file", type=["mseed"])

if uploaded:
    with open("/tmp/temp.mseed", "wb") as f:
        f.write(uploaded.read())

    trace = read("/tmp/temp.mseed")
    trace.detrend("demean")
    trace.taper(max_percentage=0.05, type="cosine")
    trace.filter("bandpass", freqmin=1.0, freqmax=9.0)

    data = trace[0].data.astype(float)
    sr = trace[0].stats.sampling_rate
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/sr)

    features = pd.DataFrame([{
        "max_amp": np.max(np.abs(data)),
        "mean_amp": np.mean(np.abs(data)),
        "std_amp": np.std(data),
        "rms": np.sqrt(np.mean(data**2)),
        "dominant_freq": freqs[np.argmax(fft)],
        "duration": trace[0].stats.npts / sr
    }])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ EARTHQUAKE DETECTED")
    else:
        st.success("✅ No earthquake — normal noise")
