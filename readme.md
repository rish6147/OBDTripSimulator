🚗 OBD Trip Simulator

A real-time OBD-II data generator and visualization tool built using Streamlit. This project simulates vehicle sensor data (speed, RPM, heart rate, altitude, traffic, etc.) and visualizes trip information on an interactive map.

It is designed to help researchers, students, and developers test vehicle data analytics pipelines, train AI/ML models, and evaluate smart transportation algorithms without requiring real-world OBD devices.

🧩 Key Features

🕹️ Realistic Trip Simulation – Generates synthetic vehicle trips with GPS coordinates, speed, acceleration, altitude, and traffic conditions.

📊 Driver Health & Vehicle Metrics – Heart rate, breathing rate, and other biometrics are simulated for advanced studies in driver monitoring.

🗺️ Map Visualization – Interactive Folium maps display routes with traffic-level indicators.

📤 Dataset Export – CSV export for machine learning, research, or IoT system testing.

⚙️ Customizable Parameters – Random trips, city-to-city, or manual coordinates with adjustable start time.

🎯 Usage / Gap This Project Fills

Bridges the gap between real-world vehicle data scarcity and the need for large-scale datasets for AI/ML research.

Enables students & researchers to test route-based algorithms, traffic prediction, and driver health models without expensive OBD hardware.

Supports simulation of different traffic conditions, altitudes, and driver states, which is often missing in public datasets.

Can be used for IoT-based smart city simulations, autonomous vehicle route testing, and educational purposes.

🧠 How It Works

Randomized data points are generated for GPS, speed, altitude, traffic, and driver health metrics.

Route is either random, city-to-city, or manually defined.

Each point is time-stamped and smoothed for realism.

The Streamlit dashboard allows visualization, previewing, and exporting data.

📦 Installation & Setup
Clone the repository
git clone https://github.com/rish6147/OBDTripSimulator.git
cd OBDTripSimulator

Install dependencies
pip install -r requirements.txt

Run the Streamlit app
streamlit run main.py