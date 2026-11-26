import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import date, time

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Kestrel Service Routing",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Kestrel Home Service Request Routing")
st.subheader("AI-Powered Customer Service Request Classification")

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "kestrel_routing_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Routing model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# =====================================
# 3. CUSTOMER INPUT FORM
# =====================================

with st.form("routing_form"):

    request_text = st.text_area(
        "Customer Request",
        placeholder="Describe the customer's issue..."
    )

    col1, col2 = st.columns(2)

    with col1:
        channel = st.text_input("Channel", value="whatsapp")
        product_family = st.text_input(
            "Product Family",
            value="Water Purifier"
        )
        warranty_status = st.text_input(
            "Warranty Status",
            value="in_warranty"
        )

    with col2:
        source = st.text_input("Source", value="crm")
        request_date = st.date_input(
            "Request Date",
            value=date.today()
        )
        request_time = st.time_input(
            "Request Time",
            value=time(10, 30)
        )

    submitted = st.form_submit_button(
        "Predict Service Team",
        type="primary"
    )

# =====================================
# 4. PREDICTION
# =====================================

if submitted:

    if not request_text.strip():
        st.warning("Please enter a customer request.")

    else:
        try:
            dt = pd.Timestamp.combine(
                request_date,
                request_time
            )

            request_hour = dt.hour
            request_day = dt.dayofweek
            request_month = dt.month
            request_day_of_month = dt.day
            is_weekend = int(request_day >= 5)

            if 5 <= request_hour < 12:
                time_period = "Morning"
            elif 12 <= request_hour < 17:
                time_period = "Afternoon"
            elif 17 <= request_hour < 21:
                time_period = "Evening"
            else:
                time_period = "Night"

            input_data = pd.DataFrame([{
                "request_text": request_text,
                "channel": channel,
                "product_family": product_family,
                "warranty_status": warranty_status,
                "source": source,
                "time_period": time_period,
                "request_hour": request_hour,
                "request_day": request_day,
                "request_month": request_month,
                "request_day_of_month": request_day_of_month,
                "is_weekend": is_weekend
            }])

            feature_columns = [
                "request_text",
                "channel",
                "product_family",
                "warranty_status",
                "source",
                "time_period",
                "request_hour",
                "request_day",
                "request_month",
                "request_day_of_month",
                "is_weekend"
            ]

            input_data = input_data[feature_columns]

            prediction = model.predict(input_data)[0]

            decision_scores = model.decision_function(input_data)

            sorted_scores = sorted(
                decision_scores[0],
                reverse=True
            )

            confidence_margin = (
                sorted_scores[0] - sorted_scores[1]
            )

            if confidence_margin < 0.20:
                routing_status = "Human Review Required"
            else:
                routing_status = "Automatic Routing"

            # =====================================
            # 5. DISPLAY RESULTS
            # =====================================

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Recommended Team: {prediction}")

            col1, col2 = st.columns(2)

            col1.metric(
                "Confidence Margin",
                f"{confidence_margin:.4f}"
            )

            col2.write("**Routing Status**")
            if routing_status == "Automatic Routing":
                col2.success(routing_status)
            else:
                col2.warning(routing_status)

            st.write(
                f"**Request Date:** {dt.strftime('%d-%m-%Y')}"
            )
            st.write(
                f"**Request Time:** {dt.strftime('%I:%M %p')}"
            )

            st.caption(
                "Confidence margin is a decision-score difference, "
                "not a calibrated probability."
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
