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

    # SHAP Explainability
    st.subheader("🔍 SHAP Explainability")

    # Sample to avoid memory overload
    sample_size = min(300, X_transformed.shape[0])
    sample_idx = np.random.choice(X_transformed.shape[0], size=sample_size, replace=False)
    X_sample = X_transformed[sample_idx]
    df_sample = df.iloc[sample_idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    row_num = st.number_input("Select borrower row for explanation (sampled)", 0, sample_size - 1, 0)
    st.write("Borrower data:")
    st.dataframe(df_sample.iloc[[row_num]])

    st.markdown("#### 🔎 Why did this borrower get this score?")
    fig = shap.plots._waterfall.waterfall_legacy(shap_values[row_num], show=False)
    st.pyplot(fig)

    with st.expander("📊 Global Feature Importance (Sampled)"):
        st.write("Top factors influencing all predictions (sampled):")
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(bbox_inches='tight')
else:
    st.info("👆 Upload a CSV file to start")


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
