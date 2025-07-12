import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Score Classifier", layout="wide")
st.title("🔍 Credit Score Classifier with SHAP Explainability")

# Load model and label encoder
model = joblib.load("credit_model_tuned.pkl")
label_encoder = joblib.load("label_encoder.pkl")
preprocessor = model.named_steps['preprocess']
clf = model.named_steps['clf']

# Upload borrower CSV
uploaded = st.file_uploader("📤 Upload borrower data (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # Drop unused columns if they exist
    drop_cols = ['Credit_Score', 'Customer_ID', 'ID']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Transform features
    X_transformed = preprocessor.transform(df)

    # Predict
    preds = clf.predict(X_transformed)
    probs = clf.predict_proba(X_transformed)

    df["Predicted_Score"] = label_encoder.inverse_transform(preds)
    df["Confidence"] = np.max(probs, axis=1)

    st.subheader("📈 Prediction Results")
    st.dataframe(df[["Predicted_Score", "Confidence"]].join(df.drop(columns=["Predicted_Score", "Confidence"])))

    # Download predictions
    st.download_button("📥 Download Results", df.to_csv(index=False), file_name="credit_predictions.csv")

    # SHAP Explainability Section
    st.subheader("🔍 SHAP Explainability")
    explainer = shap.TreeExplainer(clf)

    # ---- Global SHAP Beeswarm Plot ----
    with st.expander("📊 Global Feature Importance (Beeswarm SHAP)"):
        st.write("Top features influencing credit score predictions:")
        sample_size = st.slider("Sample size for SHAP summary", min_value=50, max_value=300, step=50, value=100)
        sample_idx = np.random.choice(X_transformed.shape[0], size=sample_size, replace=False)
        X_sample = X_transformed[sample_idx]

        try:
            shap_values = explainer.shap_values(X_sample)
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="dot")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render SHAP plot: {str(e)}")

    # ---- Individual SHAP Waterfall Plot ----
    st.subheader("🔎 Individual Prediction Explanation")
    row_num = st.number_input("Select borrower index to explain", 0, len(df)-1, 0)
    st.write("Borrower details:")
    st.dataframe(df.iloc[[row_num]])

    try:
        shap_single = explainer(X_transformed[row_num:row_num+1])
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_single[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate waterfall plot: {str(e)}")

else:
    st.info("👆 Upload a CSV file with borrower data matching the expected structure below.")

with st.expander("📋 Download sample template & see required format"):
    sample_df = pd.DataFrame(columns=[
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ])
    st.dataframe(sample_df)

    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv,
        file_name='borrower_template.csv',
        mime='text/csv'
    )

st.markdown("""
ℹ️ **Note:** You may leave non-critical fields blank — the app will automatically fill missing values using the trained model’s preprocessing logic.

🟢 Required: Ensure all columns are present in the upload (even if empty).  
🟡 Optional fields can be left blank if borrower data is incomplete.
""")






import openai

# --- Sidebar: LLM Assistant ---
st.sidebar.markdown("## 🤖 LLM Assistant")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    user_question = st.sidebar.text_area("Ask me about credit scoring, predictions, or SHAP insights:")

    if st.sidebar.button("Ask"):
        try:
            openai.api_key = api_key

            # Inject some context (summary + borrower + SHAP if available)
            context = "This app predicts credit scores as Good, Standard, or Poor using XGBoost.\n"
            if uploaded:
                context += f"Here is a summary of predictions:\n{df['Predicted_Score'].value_counts().to_string()}\n"
                context += f"The top features influencing predictions include EMI, Annual Income, and Credit Utilization.\n"

                # If row number used, include sample borrower's info
                if 'row_num' in locals():
                    borrower_info = df_sample.iloc[row_num].drop("Predicted_Score", errors='ignore')
                    context += f"Sample borrower data:\n{borrower_info.to_string()}\n"

            full_prompt = f"{context}\nUser question: {user_question}\nAssistant:"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains credit scoring model predictions."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.5
            )

            st.sidebar.markdown("### 💬 Assistant's Response")
            st.sidebar.write(response['choices'][0]['message']['content'])

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
else:
    st.sidebar.info("🔐 Enter your OpenAI API key to enable the assistant.")

with st.sidebar.expander("ℹ️ How to get your OpenAI API key"):
    st.markdown("""
    1. Go to [platform.openai.com](https://platform.openai.com)
    2. Sign in or create a free account
    3. Navigate to **API Keys**
    4. Click **Create new secret key**
    5. Copy the key and paste it above
    """)