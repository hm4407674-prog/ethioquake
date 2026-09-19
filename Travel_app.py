 import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 Travel Planner Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Plan your trip with location, weather, budget, and interactive map information."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# API CONFIGURATION
# ============================================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 10


# ============================================================
# CITY SEARCH
# ============================================================

@st.cache_data(ttl=3600)
def find_city(city_name: str):
    """Find the first matching city using Open-Meteo geocoding."""

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "name": city_name,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("results"):
            return None

        place = data["results"][0]

        return {
            "name": place.get("name", city_name),
            "country": place.get("country", "Unknown"),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        }

    except requests.RequestException:
        return None


# ============================================================
# WEATHER SERVICE
# ============================================================

@st.cache_data(ttl=1800)
def get_weather(latitude: float, longitude: float):
    """Retrieve current weather and a five-day forecast."""

    try:
        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 5,
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("✈️ Trip Settings")

    city_input = st.text_input(
        "🔍 Destination",
        value="Tokyo",
        placeholder="Enter a city...",
    )

    budget = st.number_input(
        "💰 Total Budget ($)",
        min_value=0,
        max_value=100000,
        value=2000,
        step=100,
    )

    st.divider()

    st.caption(
        "Weather data provided by Open-Meteo."
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

if city_input.strip():

    with st.spinner("🔎 Searching for destination..."):

        location = find_city(city_input.strip())

    if location is None:

        st.error(
            "❌ City not found or the location service is temporarily unavailable."
        )

        st.info(
            "Try a different spelling or enter a major city."
        )

    else:

        with st.spinner("🌤️ Loading weather information..."):

            weather = get_weather(
                location["latitude"],
                location["longitude"],
            )

        if weather is None:

            st.error(
                "❌ Unable to retrieve weather information right now."
            )

        else:

            # ------------------------------------------------
            # LOCATION
            # ------------------------------------------------

            st.subheader(
                f"📍 {location['name']}, {location['country']}"
            )

            st.caption(
                f"Coordinates: "
                f"{location['latitude']:.4f}, "
                f"{location['longitude']:.4f}"
            )

            # ------------------------------------------------
            # KEY METRICS
            # ------------------------------------------------

            current = weather.get("current", {})

            temperature = current.get(
                "temperature_2m",
                "N/A",
            )

            wind_speed = current.get(
                "wind_speed_10m",
                "N/A",
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🌡️ Current Temperature",
                    f"{temperature} °C",
                )

            with col2:
                st.metric(
                    "💨 Wind Speed",
                    f"{wind_speed} km/h",
                )

            with col3:
                st.metric(
                    "💰 Trip Budget",
                    f"${budget:,.0f}",
                )

            # ------------------------------------------------
            # WEATHER FORECAST
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📅 5-Day Forecast</div>',
                unsafe_allow_html=True,
            )

            daily = weather.get("daily", {})

            forecast = pd.DataFrame(
                {
                    "Date": daily.get("time", []),
                    "High °C": daily.get(
                        "temperature_2m_max",
                        [],
                    ),
                    "Low °C": daily.get(
                        "temperature_2m_min",
                        [],
                    ),
                }
            )

            if not forecast.empty:

                forecast["Date"] = pd.to_datetime(
                    forecast["Date"]
                ).dt.strftime("%a, %b %d")

                forecast["High °C"] = forecast[
                    "High °C"
                ].round(1)

                forecast["Low °C"] = forecast[
                    "Low °C"
                ].round(1)

                st.dataframe(
                    forecast,
                    use_container_width=True,
                    hide_index=True,
                )

            # ------------------------------------------------
            # INTERACTIVE MAP
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">🗺️ Destination Map</div>',
                unsafe_allow_html=True,
            )

            map_view = folium.Map(
                location=[
                    location["latitude"],
                    location["longitude"],
                ],
                zoom_start=11,
                control_scale=True,
            )

            folium.Marker(
                location=[
                    location["latitude"],
                    location["longitude"],
                ],
                popup=(
                    f"<b>{location['name']}</b><br>"
                    f"{location['country']}"
                ),
                tooltip="📍 Destination",
            ).add_to(map_view)

            st_folium(
                map_view,
                height=450,
                use_container_width=True,
            )

            # ------------------------------------------------
            # TRIP SUMMARY
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📋 Trip Summary</div>',
                unsafe_allow_html=True,
            )

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.write(
                    f"**Destination:** "
                    f"{location['name']}, "
                    f"{location['country']}"
                )

                st.write(
                    f"**Budget:** ${budget:,.0f}"
                )

            with summary_col2:

                st.write(
                    f"**Current Temperature:** "
                    f"{temperature} °C"
                )

                st.write(
                    f"**Wind Speed:** "
                    f"{wind_speed} km/h"
                )

else:

    st.info(
        "🌍 Enter a destination in the sidebar to start planning your trip."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 Travel Planner • Location and weather data powered by Open-Meteo"
  )
