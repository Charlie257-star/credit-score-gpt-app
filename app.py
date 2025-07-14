import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import openai

st.set_page_config(page_title="Credit Score Classifier", layout="wide")
st.title("🔍 Credit Score Classifier with GPT Assistant")

# Load model and label encoder
model = joblib.load("credit_model_tuned.pkl")
label_encoder = joblib.load("label_encoder.pkl")
preprocessor = model.named_steps['preprocess']
clf = model.named_steps['clf']

# Sidebar: OpenAI API Key input
openai_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password")
if openai_key:
    st.sidebar.success("✅ Key loaded. You can now use the assistant.")
else:
    st.sidebar.info("🔐 Enter your OpenAI API key to enable the assistant.")
    with st.sidebar.expander("ℹ️ How to get your OpenAI API key"):
        st.markdown("""
        1. Go to [platform.openai.com](https://platform.openai.com)  
        2. Sign in or create an account  
        3. Navigate to **API Keys**  
        4. Click **Create new secret key**  
        5. Paste the key above  
        """)

# File upload section

st.markdown("""
    ℹ️ **Note:** You may leave non-critical fields blank — the app will automatically fill missing values using the trained model’s preprocessing logic.

    🟢 Required: Ensure all columns are present in the upload (even if empty).  
    🟡 Optional fields can be left blank if borrower data is incomplete.
    👇 If unsure, [download the CSV template](#) below to use as a guide.
    """)

# 📥 Download pre-filled example CSV for testing

st.markdown("""
### 👇 Don't want to upload your own file?

Download and upload a ready-to-use prefilled example file to test the app instantly:
""")

# Load the test.csv file 
sample_csv_path = "test.csv"  

with open(sample_csv_path, "rb") as f:
    st.download_button(
        label="⬇️ Download Example Borrower Data",
        data=f,
        file_name="example_borrower_data.csv",
        mime="text/csv"
    )


# Define template columns (match exactly what model expects)
template_df = pd.DataFrame(columns=[
    'ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation',
    'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Type_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Credit_History_Age',
    'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
    'Payment_Behaviour', 'Monthly_Balance'
])

# Download button + preview
with st.expander("📥 Download Sample CSV Template"):
    st.dataframe(template_df.head(1))  # Just show header
    csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full CSV Template", csv, file_name="credit_score_template.csv", mime="text/csv")
uploaded = st.file_uploader("📁 Upload your borrower CSV file", type=["csv"])

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

    df["Confidence_Flag"] = df["Confidence"].apply(
        lambda x: "⚠️ Low confidence" if x < 0.6 else ""
    )


    st.subheader("📈 Prediction Results")
    styled_df = df[["Predicted_Score", "Confidence", "Confidence_Flag"]].style.applymap(
    lambda val: "color: red;" if val == "⚠️ Low confidence" else ""
    , subset=["Confidence_Flag"]
    )

    st.dataframe(styled_df)
    
    

    

    st.subheader("🧍 Select Borrower for Explanation")
    row_num = st.number_input("Select borrower index to explain", 0, len(df) - 1, 0)
    st.write("Borrower details:")
    st.dataframe(df.iloc[[row_num]])

    st.markdown("""
    🔍 **Note:** Predictions marked with ⚠️ indicate the model was not strongly confident.
         These cases may require further review or additional data.
    """)


    # GPT assistant (only if key is provided)
    if openai_key:
        client = openai.OpenAI(api_key=openai_key)
        st.subheader("💬 Ask the Assistant")
        user_q = st.text_input("Ask a question about this borrower's prediction:")

        if user_q:
            try:
                data_summary = df.describe().T.to_string()
                borrower_input = df.iloc[[row_num]].to_string()

                prompt = f"""You are a credit risk assistant. Explain this borrower's risk and features.

Borrower Features:\n{borrower_input}
Data Summary:\n{data_summary}
Question: {user_q}
Answer:"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful credit risk and ML assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )

                st.success(response.choices[0].message.content)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Upload a borrower file to start.")