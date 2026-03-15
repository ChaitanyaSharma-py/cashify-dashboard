# 📱 Cashify Consumer Intelligence — MIS Dashboard

An interactive Decision Support System (DSS) built with Streamlit, analysing **3,444 survey responses** across two recommerce journeys — **Buyback (sell-side)** and **Refurbished (buy-side)**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📊 Dashboard Modules

| Tab | Description |
|-----|-------------|
| 🏠 **Overview** | KPI summary, respondent profile, demographic pivot table |
| 📣 **Brand Awareness** | TOM → Spontaneous → Aided for all platforms; demographic cuts |
| 🔽 **Health Funnel** | Awareness → Familiarity → Ever Used → Intent; Cashify waterfall |
| ⭐ **NPS** | Net Promoter Scores with P/Pa/D breakdown; NPS by demographic |
| 📡 **Source of Awareness** | Platform × Channel heatmap (counts + %) |
| 🎯 **Consideration Set** | Shortlist analysis; strong consideration ranking |
| 💡 **Choice Drivers** | Weighted rank scores — Cashify vs competitors |
| 🚧 **Barriers** | Key deterrents visualised; treemap + demographic drill-down |
| 📊 **Category Insights** | Ecosystem-level drivers & fears; Q24 top-3 driver analysis |

All tabs support **real-time filtering** by: Journey · City Tier · Gender · Age · Income · Occupation

---

## 🗂️ Repository Structure

```
cashify-dashboard/
├── cashify_dashboard.py          # Main Streamlit app
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore
├── .streamlit/
│   └── config.toml               # Theme + server config
└── data/
    ├── Live_Brand_Study_-_CASHIFY_Buyback_-_Final_data.xlsx
    └── Live_Brand_Study_-_CASHIFY_-_Refurbished_data.xlsx
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cashify-dashboard.git
cd cashify-dashboard

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run cashify_dashboard.py
```

Open your browser at **http://localhost:8501**

---

## 🚀 Deploy to Streamlit Cloud

See the detailed step-by-step guide below, or visit [share.streamlit.io](https://share.streamlit.io).

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.35.0 | Dashboard framework |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| plotly | 5.22.0 | Interactive charts |
| openpyxl | 3.1.2 | Excel file reading |
| xlrd | 2.0.1 | Legacy Excel support |

---

## 📁 Data

Survey data collected as part of the **Cashify MIS Live Brand Study** (PGDM programme).  
- `Buyback`: n = 1,596 responses, 103 variables  
- `Refurbished`: n = 1,848 responses, 101 variables

---

*MIS Live Project | PGDM Section*
