import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
from obspy import read

# ============================================================
# EthioQuake — AI Earthquake Detection for Ethiopia
# Models: Random Forest (baseline) + 1D CNN (deep learning)
# ============================================================

st.set_page_config(page_title="EthioQuake", page_icon="🌍", layout="wide")

st.title("🌍 EthioQuake")
st.subheader("AI-Powered Earthquake Detection for Ethiopia")
st.write(
    "Upload a seismic waveform in MiniSEED format. EthioQuake will extract "
    "signal features and run predictions using both a Random Forest classifier "
    "and a 1D Convolutional Neural Network."
)

# ---------- Sidebar ----------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    "EthioQuake uses real seismic data from the **IU.FURI** station in Ethiopia "
    "(EarthScope/IRIS). Two models are compared: a Random Forest baseline and a "
    "1D CNN trained on raw waveforms."
)
st.sidebar.markdown("**Model Performance**")
st.sidebar.write("- Random Forest: 82% test accuracy")
st.sidebar.write("- 1D CNN: 100% test accuracy (small test set)")

# ---------- Load Models ----------
@st.cache_resource
def load_rf_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

@st.cache_resource
def load_cnn_model():
    if os.path.exists("ethioquake_cnn.keras"):
        from tensorflow.keras.models import load_model
        return load_model("ethioquake_cnn.keras")
    return None

rf_model = load_rf_model()
cnn_model = load_cnn_model()

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "Upload Seismic Data (MiniSEED .mseed)", type=["mseed", "ms"]
)

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mseed") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("Seismic waveform processed successfully.")

    # Read waveform
    try:
        st_stream = read(tmp_path)
        trace = st_stream[0]
        data = trace.data.astype(np.float32)

        # Clean signal
        data = data - data.mean()
        if data.std() > 0:
            data = data / data.std()

        # ---------- Display Waveform Info ----------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Waveform Info")
            st.write(f"**Station:** {trace.stats.station}")
            st.write(f"**Channel:** {trace.stats.channel}")
            st.write(f"**Sampling rate:** {trace.stats.sampling_rate} Hz")
            st.write(f"**Samples:** {len(data)}")

        with col2:
            st.markdown("### 📈 Signal Statistics")
            st.write(f"**Max amplitude:** {data.max():.4f}")
            st.write(f"**Min amplitude:** {data.min():.4f}")
            st.write(f"**Mean:** {data.mean():.4f}")
            st.write(f"**Std:** {data.std():.4f}")

        # ---------- Random Forest Prediction ----------
        st.markdown("---")
        st.markdown("### 🌲 Random Forest Prediction")

        if rf_model is not None:
            # Extract hand-engineered features (same as training)
        features = np.array([[
    np.sqrt(np.mean(data ** 2)),                      # RMS
    np.var(data),                                     # Variance
    np.max(np.abs(data)),                             # Peak amplitude
    np.mean(np.abs(data)),                            # Mean absolute
    np.std(data),                                     # Standard deviation
    np.max(data) - np.min(data),                      # Peak-to-peak
]])          # Peak amplitude
            ]])
            try:
                rf_pred = rf_model.predict(features)[0]
                rf_label = "🚨 EARTHQUAKE" if rf_pred == 1 else "✅ NOISE"
                st.write(f"**Prediction:** {rf_label}")
            except Exception as e:
                st.warning(f"Random Forest prediction failed: {e}")
        else:
            st.warning("Random Forest model (model.pkl) not found in repository.")

        # ---------- CNN Prediction ----------
        st.markdown("---")
        st.markdown("### 🧠 1D CNN Prediction")

        if cnn_model is not None:
            # Slice into 10-second windows (400 samples @ 40Hz)
            window_size = 400
            if len(data) >= window_size:
                windows = []
                for i in range(0, len(data) - window_size, window_size):
                    windows.append(data[i:i + window_size])
                X_input = np.array(windows).reshape(-1, window_size, 1)

                preds = cnn_model.predict(X_input, verbose=0).flatten()
                avg_pred = float(np.mean(preds))
                cnn_label = "🚨 EARTHQUAKE" if avg_pred > 0.5 else "✅ NOISE"

                st.write(f"**Windows analyzed:** {len(windows)}")
                st.write(f"**Average confidence:** {avg_pred:.4f}")
                st.write(f"**Prediction:** {cnn_label}")
            else:
                st.warning(
                    f"Waveform too short for CNN ({len(data)} samples). "
                    f"Need at least {window_size} samples (10 seconds at 40 Hz)."
                )
        else:
            st.warning("CNN model (ethioquake_cnn.keras) not found in repository.")

        # ---------- Cleanup ----------
        os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Failed to read MiniSEED file: {e}")
else:
    st.info("👆 Upload a .mseed file to begin analysis.")

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "EthioQuake • Built by Hafiz Mohammed • Hawassa University • "
    "[GitHub](https://github.com/hm4407674-prog/ethioquake)"
            )
