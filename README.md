
# Credit Score Prediction App with SHAP & GPT Assistant

This is an interactive Streamlit app that predicts borrower credit scores (Good, Standard, Poor) using a tuned XGBoost model. It includes global and individual SHAP explainability, and a built-in GPT-powered assistant to help interpret results, if need be.

## 🚀 Features

- 📊 **Credit score classification** using XGBoost
- 🧠 **Explainable AI** with SHAP (beeswarm & waterfall plots)
- 🤖 **Chat assistant** powered by GPT (BYO OpenAI API key)
- 📂 **CSV upload + downloadable template**
- 📥 Prediction download (.csv)
- 🧱 Handles missing values in input data gracefully


Live demo: https://credit-score-gpt-app-xkdoanmqmu2kzc4wanypwn.streamlit.app/ 

---

##  How to Use

1. Upload borrower data using the CSV uploader
2. Or [download the sample template](#) and fill in your data
3. View credit score predictions and confidence levels
4. Explore feature importance with SHAP plots
5. Ask the built-in assistant to explain results

---

## 🤖 LLM Assistant Setup

To use the GPT assistant:

1. Create a free account at [platform.openai.com](https://platform.openai.com)
2. Go to your API Keys page
3. Click **"Create new secret key"**
4. Paste the key in the sidebar input

💡 We do not store or share your key — it's used only in your session.

---

##  Built With

- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- Streamlit

---
## 📥 Run Locally

```bash
git clone https://github.com/Charlie257-star/credit-score-gpt-app.git
cd credit-score-gpt-app
pip install -r requirements.txt
streamlit run app.py

##  Contact

Charles Otieno – www.linkedin.com/in/charles-otieno-903921227 
