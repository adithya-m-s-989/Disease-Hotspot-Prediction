# Disease Hotspot Prediction with Graph Neural Networks

This project uses spatio-temporal Graph Neural Networks to predict respiratory disease hotspots across U.S. counties using EPA air pollution data. An interactive Streamlit dashboard visualizes hotspot intensity, pollutant trends, and high-risk regions, supporting public health research and environmental monitoring.

# 🌡️ US Disease Hotspot & Pollutant Dashboard

Dashboard Preview<img width="945" alt="dashboard" src="https://github.com/user-attachments/assets/04969b53-0f3c-4d50-9ff7-97fbbf25d542" />


An interactive web dashboard to visualize and analyze disease hotspots across the United States and their correlation with air pollutant levels. This tool combines **Graph Neural Network (GNN)** predictions with EPA pollution data to provide insights for public health research and environmental monitoring.

📍 **Live App:** [https://diseasehotspotprediction.streamlit.app/](https://diseasehotspotprediction.streamlit.app/)

---

## 🚀 Features

* 🗺️ **Interactive Hotspot Map** – Visualize county-level GNN-based hotspot intensity
* 📈 **Trend Forecasting** – Analyze past hotspot trends and forecast future ones
* 🏆 **Top Counties & States** – Discover which regions are most affected
* 🏭 **Pollutant Analysis** – Explore pollutant distributions and monthly variations
* 💾 **Data Explorer** – View and export raw merged datasets

---

## 📁 Project Structure

```
📦 disease-hotspot-dashboard/
├── 01_datacleaning_eda.ipynb                  # Initial EDA and data cleaning
├── Feature_Engineering.ipynb                  # Feature transformation for modeling
├── Semi_Supervised_Learning_Model.ipynb       # GNN-based semi-supervised learning implementation
├── Supervised_Learning_Model.ipynb            # Baseline supervised models
├── all_pollutants_merged_inner.csv            # Final pollutant dataset (EPA + harmonized)
├── all_years_gnn_predictions_semi_supervised.csv # GNN output for county-wise hotspot scores
├── main.py                                    # Streamlit app combining disease & pollution insights
├── requirements.txt                           # Required Python dependencies for local setup
├── us_counties.geojson                        # County-level GeoJSON for mapping
├── README.md                                  # Project overview and usage instructions
```

---

## 📊 Data Sources

* **Hotspot Predictions**: Semi-supervised GNN output identifying likely disease hotspot regions
* **EPA Air Quality Data**: Historical pollution metrics including PM2.5, NO₂, SO₂, CO, and O₃ by county and month

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/disease-hotspot-dashboard.git
cd disease-hotspot-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App Locally

```bash
streamlit run main.py
```

---

## 📦 Requirements

Core dependencies:

* `streamlit`
* `pandas`
* `plotly`
* `numpy`

Install them via the provided `requirements.txt`.

---

## 💡 Future Enhancements

* 🔄 Real-time streaming via Kafka and Spark
* 📍 Enable FIPS-based filtering
* 🌐 Integrate satellite and climate datasets
* 🧪 Upload your own model for hotspot comparison
* 📱 Improve mobile responsiveness

---

## 🤝 Contributors

* **Adithya M Sivakumar**
  MSBA Candidate @ UC Davis
  
* **Aishwarya Banumathy**
  MSBA Candidate @ UC Davis

* **Sanjay Puri**
  MSBA Candidate @ UC Davis

* **Deheng Peng**
 MSBA Candidate @ UC Davis

* **Sam Mathew**
 MSBA Candidate @ UC Davis
