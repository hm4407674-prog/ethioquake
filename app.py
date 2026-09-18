import streamlit as st
import joblib
import numpy as np
import pandas as pd
from obspy import read


# ============================================================
# EthioQuake — AI Earthquake Detection System
# ============================================================

st.set_page_config(
    page_title="EthioQuake | Earthquake Detector",
    page_icon="🌍",
    layout="centered",
)


# ------------------------------------------------------------
# Application Header
# ------------------------------------------------------------

st.title(" EthioQuake")

st.subheader("AI-Powered Earthquake Detection for Ethiopia")

st.write(
    "Upload a seismic waveform in MiniSEED format and EthioQuake "
    "will extract signal features for machine-learning-based analysis."
)


# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained machine-learning model."""
    return joblib.load("model.pkl")


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------

st.markdown("###  Upload Seismic Data")

uploaded_file = st.file_uploader(
    "Choose a MiniSEED (.mseed) seismic file",
    type=["mseed"],
    help="Upload a seismic waveform recorded by a compatible seismic station.",
)


# ------------------------------------------------------------
# Process Uploaded Waveform
# ------------------------------------------------------------

if uploaded_file is not None:

    try:
        # Save uploaded file temporarily
        temporary_file = "/tmp/temp.mseed"

        with open(temporary_file, "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Read seismic waveform
        stream = read(temporary_file)

        # Use the first available trace
        trace = stream[0]

        # ----------------------------------------------------
        # Signal Preprocessing
        # ----------------------------------------------------

        trace.detrend("demean")
        trace.taper(max_percentage=0.05, type="cosine")

        # Keep frequencies relevant to the detection pipeline
        trace.filter(
            "bandpass",
            freqmin=1.0,
            freqmax=9.0,
        )

        # Convert waveform to numerical array
        data = trace.data.astype(float)

        # Sampling rate
        sampling_rate = trace.stats.sampling_rate

        # ----------------------------------------------------
        # Frequency-Domain Analysis
        # ----------------------------------------------------

        fft_amplitude = np.abs(np.fft.rfft(data))

        frequencies = np.fft.rfftfreq(
            len(data),
            d=1 / sampling_rate,
        )

        dominant_frequency = frequencies[
            np.argmax(fft_amplitude)
        ]

        # ----------------------------------------------------
        # Feature Extraction
        # ----------------------------------------------------

        features = pd.DataFrame(
            [
                {
                    "max_amp": np.max(np.abs(data)),
                    "mean_amp": np.mean(np.abs(data)),
                    "std_amp": np.std(data),
                    "rms": np.sqrt(np.mean(data ** 2)),
                    "dominant_freq": dominant_frequency,
                    "duration": trace.stats.npts / sampling_rate,
                }
            ]
        )

        # ----------------------------------------------------
        # Display Extracted Features
        # ----------------------------------------------------

        st.success("Seismic waveform processed successfully.")

        st.markdown("###  Extracted Seismic Features")

        st.dataframe(
            features,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        prediction = model.predict(features)

        st.markdown("###  Model Analysis")

        st.write("Prediction:", prediction[0])

    except Exception as error:

        st.error(
            "An error occurred while processing the seismic file."
        )

        st.exception(error)
