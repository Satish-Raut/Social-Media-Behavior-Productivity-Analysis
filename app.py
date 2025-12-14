import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ======================================================
# Streamlit Config
# ======================================================
st.set_page_config(
    page_title="Social Media & Productivity Analytics",
    layout="wide"
)

# ======================================================
# Column Rename Map (RAW → ML Friendly)
# ======================================================
COLUMN_RENAME_MAP = {
    "Age": "age",
    "How many hours do you spend on social media daily?": "daily_social_media_hours",
    "Which platforms do you use the most?": "primary_social_media_platform",
    "At what time do you use social media the most?": "peak_social_media_time",
    "Do you use social media while studying?": "use_social_media_while_studying",
    "Average sleep duration per night": "avg_sleep_hours",
    "How often do you procrastinate because of social media?": "procrastination_frequency",
    "Do you use your phone after getting into bed?": "phone_use_after_bed",
    "Do you feel social media affects your concentration?": "social_media_affects_concentration",
    "How satisfied are you with your productivity?": "productivity_satisfaction"
}

# ======================================================
# Load Dataset (EDA only)
# ======================================================
df_raw = pd.read_csv("MLDataset.csv")
df_raw.columns = df_raw.columns.str.strip()
df = df_raw.rename(columns=COLUMN_RENAME_MAP)

# ======================================================
# Load Models & Artifacts
# ======================================================
log_model = pickle.load(open("logistic_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
ordinal_encoder = pickle.load(open("ordinal_encoder.pkl", "rb"))

# ======================================================
# Ordinal Columns (MUST match training)
# ======================================================
ordinal_cols = [
    "daily_social_media_hours",
    "avg_sleep_hours",
    "use_social_media_while_studying",
    "procrastination_frequency",
    "social_media_affects_concentration"
]

# ======================================================
# Preprocessing (USED BY BOTH MODELS)
# ======================================================
def preprocess_input(raw_df):
    dfp = raw_df.copy()

    # Ordinal encoding
    dfp[ordinal_cols] = ordinal_encoder.transform(dfp[ordinal_cols])

    # Binary encoding
    dfp["phone_use_after_bed"] = (
        dfp["phone_use_after_bed"]
        .astype(str)
        .str.strip()
        .str.title()
        .map({"No": 0, "Yes": 1})
    )

    # One-hot encoding
    dfp = pd.get_dummies(dfp)

    # Safety: remove cluster if present
    if "cluster" in dfp.columns:
        dfp = dfp.drop(columns=["cluster"])

    # Feature alignment
    for col in feature_columns:
        if col not in dfp.columns:
            dfp[col] = 0

    dfp = dfp[feature_columns]

    # Scaling (same scaler used during training)
    return scaler.transform(dfp)

# ======================================================
# Sidebar Navigation
# ======================================================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA Dashboard", "🤖 Productivity Prediction", "🧠 User Clustering"]
)

# ======================================================
# 🏠 HOME
# ======================================================
if menu == "🏠 Home":
    st.title("📱 Social Media Behavior & Productivity Analysis")

    st.markdown("""
    This dashboard analyzes how **social media habits** affect  
    **sleep, focus, procrastination, and productivity**.

    **Models Used**
    - Logistic Regression → Productivity prediction
    - K-Means → User behavior clustering
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", df.shape[0])
    c2.metric("ML Models", "2")
    c3.metric("Features Used", len(feature_columns))

# ======================================================
# 📊 EDA DASHBOARD
# ======================================================
elif menu == "📊 EDA Dashboard":

    analysis = st.selectbox(
        "Select Analysis",
        [
            "Age Distribution",
            "Daily Social Media Usage",
            "Platform Preference",
            "Sleep vs Phone Usage",
            "Procrastination vs Productivity"
        ]
    )

    if analysis == "Age Distribution":
        st.plotly_chart(px.histogram(df, x="age"), use_container_width=True)

    elif analysis == "Daily Social Media Usage":
        st.plotly_chart(
            px.bar(df["daily_social_media_hours"].value_counts().reset_index(),
                   x="daily_social_media_hours", y="count"),
            use_container_width=True
        )

    elif analysis == "Platform Preference":
        st.plotly_chart(
            px.pie(df, names="primary_social_media_platform"),
            use_container_width=True
        )

    elif analysis == "Sleep vs Phone Usage":
        st.plotly_chart(
            px.histogram(df, x="avg_sleep_hours",
                         color="phone_use_after_bed", barmode="group"),
            use_container_width=True
        )

    elif analysis == "Procrastination vs Productivity":
        st.plotly_chart(
            px.histogram(df, x="productivity_satisfaction",
                         color="procrastination_frequency", barmode="group"),
            use_container_width=True
        )

# ======================================================
# 🤖 PRODUCTIVITY PREDICTION
# ======================================================
elif menu == "🤖 Productivity Prediction":
    st.title("🎯 Productivity Prediction")

    user_df = pd.DataFrame([{
        "age": st.slider("Age", 16, 30, 22),
        "daily_social_media_hours": st.selectbox(
            "Daily Social Media Usage",
            ["0–1 hours", "1–2 hours", "2–3 hours", "3–5 hours", "More than 5 hours"]
        ),
        "primary_social_media_platform": st.selectbox(
            "Primary Social Media Platform",
            ["Instagram", "Snapchat", "LinkedIn", "WhatsApp", "YouTube", "Chrome", "Twitter", "Chatgpt"]
        ),
        "peak_social_media_time": st.selectbox(
            "Peak Usage Time",
            ["Morning", "Afternoon", "Evening", "Late Night (after 11 PM)"]
        ),
        "use_social_media_while_studying": st.selectbox(
            "Use Social Media While Studying",
            ["Never", "Sometimes", "Frequently", "Always"]
        ),
        "avg_sleep_hours": st.selectbox(
            "Average Sleep Duration",
            ["Less than 5 hours", "5–6 hours", "6–7 hours", "7–8 hours", "More than 8 hours"]
        ),
        "procrastination_frequency": st.selectbox(
            "Procrastination Frequency",
            ["Never", "Rarely", "Sometimes", "Frequently"]
        ),
        "phone_use_after_bed": st.selectbox("Phone Use After Bed", ["Yes", "No"]),
        "social_media_affects_concentration": st.selectbox(
            "Social Media Affects Concentration",
            ["No", "Sometimes", "Yes"]
        )
    }])

    if st.button("Predict Productivity"):
        pred = log_model.predict(preprocess_input(user_df))[0]
        labels = [
            "Very Dissatisfied",
            "Not Satisfied",
            "Neutral",
            "Satisfied",
            "Highly Satisfied"
        ]
        st.success(f"✅ Predicted Productivity Level: **{labels[pred]}**")

# ======================================================
# 🧠 USER CLUSTERING
# ======================================================
elif menu == "🧠 User Clustering":
    st.title("🧠 User Behavior Clustering")

    cluster_map = {
        0: "📱 High Usage – Low Productivity",
        1: "⚖️ Balanced Users",
        2: "🎯 Disciplined & Productive Users"
    }

    with st.expander("🔍 Cluster Meaning"):
        for k, v in cluster_map.items():
            st.write(f"**Cluster {k}** → {v}")

    user_df = pd.DataFrame([{
        "age": st.slider("Age", 16, 30, 22),
        "daily_social_media_hours": st.selectbox(
            "Daily Social Media Usage",
            ["0–1 hours", "1–2 hours", "2–3 hours", "3–5 hours", "More than 5 hours"]
        ),
        "primary_social_media_platform": st.selectbox(
            "Primary Social Media Platform",
            ["Instagram", "Snapchat", "LinkedIn", "WhatsApp", "YouTube", "Chrome", "Twitter", "Chatgpt"]
        ),
        "peak_social_media_time": st.selectbox(
            "Peak Usage Time",
            ["Morning", "Afternoon", "Evening", "Late Night (after 11 PM)"]
        ),
        "use_social_media_while_studying": st.selectbox(
            "Use Social Media While Studying",
            ["Never", "Sometimes", "Frequently", "Always"]
        ),
        "avg_sleep_hours": st.selectbox(
            "Average Sleep Duration",
            ["Less than 5 hours", "5–6 hours", "6–7 hours", "7–8 hours", "More than 8 hours"]
        ),
        "procrastination_frequency": st.selectbox(
            "Procrastination Frequency",
            ["Never", "Rarely", "Sometimes", "Frequently"]
        ),
        "phone_use_after_bed": st.selectbox("Phone Use After Bed", ["Yes", "No"]),
        "social_media_affects_concentration": st.selectbox(
            "Social Media Affects Concentration",
            ["No", "Sometimes", "Yes"]
        )
    }])

    if st.button("Find User Cluster"):
        cluster = kmeans.predict(preprocess_input(user_df))[0]
        st.info(f"👤 User belongs to: **{cluster_map[cluster]}**")
