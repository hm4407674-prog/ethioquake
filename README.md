# EthioQuake

AI-Powered Seismic Event Detection for Ethiopia

EthioQuake is a machine-learning system designed to distinguish earthquake signals from background seismic noise using real seismic waveform data recorded at the IU.FURI station in Ethiopia.

The project combines seismic signal processing, feature engineering, and machine learning to create an end-to-end earthquake detection pipeline—from raw waveform data to model predictions and an interactive web application.

«Project status: Active development»

---

# Project Objective

Earthquake monitoring systems must distinguish meaningful seismic events from continuous background noise.

EthioQuake explores how machine learning can assist this process by:

- Processing real seismic waveform data
- Cleaning and filtering seismic signals
- Extracting informative signal features
- Training a supervised machine-learning classifier
- Evaluating model performance on unseen data
- Providing predictions through an interactive dashboard

The project is intended as an applied AI and geoscience research project, with a focus on seismic monitoring in Ethiopia.

---

# Current Results

Metric| Result
Test Accuracy| 97%
Total Waveforms| 305
Earthquake Waveforms| 80
Background Noise Waveforms| 225
Classifier| Random Forest
Primary Station| IU.FURI, Ethiopia

«Note: Accuracy is measured on the project's current test set and should not be interpreted as real-world earthquake-warning performance. Further validation on larger and geographically diverse datasets is required.»

---

# System Architecture

Seismic Waveforms
        │
        ▼
┌─────────────────────┐
│   Data Acquisition  │
│   IRIS / EarthScope │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Signal Preprocessing│
│ Detrend             │
│ Taper               │
│ Bandpass 1–9 Hz     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Extraction  │
│ Max Amplitude       │
│ RMS                 │
│ Dominant Frequency  │
│ Standard Deviation │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Random Forest Model │
└──────────┬──────────┘
           │
           ▼
    Earthquake / Noise
           │
           ▼
┌─────────────────────┐
│ Streamlit Dashboard │
└─────────────────────┘

---

. Methodology

1. Data Acquisition

Real seismic waveforms are obtained from the IRIS / EarthScope seismic data infrastructure using recordings from the IU.FURI station in Ethiopia.

2. Signal Preprocessing

Raw waveforms are processed to improve signal quality and consistency:

- Detrending
- Tapering
- Bandpass filtering
- Frequency range: 1–9 Hz

3. Feature Engineering

Each waveform is converted into numerical features that can be used by the machine-learning model:

- Maximum amplitude
- Root Mean Square (RMS)
- Dominant frequency
- Standard deviation

4. Machine Learning

A Random Forest Classifier is trained to classify waveform segments into:

Earthquake
     │
     └── Real seismic event

Noise
     │
     └── Background seismic activity

5. Evaluation

The trained model is evaluated using previously unseen test data to measure its classification performance.

---

# Technology Stack

Programming & Data Science

- Python
- NumPy
- Pandas
- scikit-learn

Seismic Data Processing

- ObsPy
- IRIS / EarthScope seismic data services
- MiniSEED waveform data

Machine Learning

- Random Forest
- Feature engineering
- Train/test evaluation
- Classification metrics

Application

- Streamlit

Development & Version Control

- Google Colab
- Google Drive
- Git
- GitHub

---

# Project Structure

ethioquake/
│
├── data/
│   ├── quakes/
│   │   └── *.mseed
│   │
│   └── noise/
│       └── *.mseed
│
├── notebooks/
│   └── analysis_and_training.ipynb
│
├── app.py
├── model.pkl
├── dataset.csv
├── earthquakes.csv
├── requirements.txt
└── README.md

---

# Getting Started

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/ethioquake.git
cd ethioquake

2. Install dependencies

pip install -r requirements.txt

Or install the main packages directly:

pip install obspy scikit-learn pandas numpy streamlit

3. Run the application

streamlit run app.py

The Streamlit dashboard will then open in your browser.

---

# Future Development

EthioQuake is an ongoing research and development project.

Planned improvements include:

- [ ] Expand the seismic dataset
- [ ] Add data from additional Ethiopian seismic stations
- [ ] Improve feature engineering
- [ ] Compare multiple machine-learning models
- [ ] Perform cross-validation and temporal validation
- [ ] Evaluate precision, recall, F1-score, and confusion matrices
- [ ] Investigate deep-learning approaches
- [ ] Improve real-time waveform processing
- [ ] Develop a more comprehensive Streamlit dashboard
- [ ] Add geographic visualization of detected events
- [ ] Investigate automated seismic event monitoring

---

# Important Scientific Disclaimer

EthioQuake is currently a research and educational prototype.

A 97% test accuracy on the current dataset does not mean the system can reliably predict earthquakes or provide operational earthquake early warning.

Reliable operational deployment would require substantially larger datasets, independent validation, multiple seismic stations, robust temporal testing, careful handling of class imbalance, and evaluation under real-world seismic conditions.

---

# Why Ethiopia?

Ethiopia is located within the East African Rift system, an active geological region where seismic activity occurs.

EthioQuake explores how modern machine-learning techniques can be applied to locally relevant seismic data and potentially contribute to future research in:

- Earthquake monitoring
- Seismology
- Artificial intelligence
- Signal processing
- Geospatial analysis
- Disaster-risk research

---

# Author

Computer Science Student
Hawassa University, Ethiopia

Interested in:

Artificial Intelligence · Machine Learning · Software Engineering · Ai Research with real problem. 

---

# License

This project is intended for educational and research purposes.

A formal open-source license can be added as the project develops.

---

 # Project Vision

«Building practical AI systems for real-world problems in Ethiopia.»

EthioQuake is an exploration of how computer science, machine learning, and seismic science can be combined to develop locally relevant technology and research.

---

Built with Python and machine learning.
