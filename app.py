
%%writefile kestrel_huggingface_app/app.py

import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

MODEL_PATH = Path(__file__).parent / "kestrel_routing_model.pkl"

model = joblib.load(MODEL_PATH)

print("Kestrel Routing Model Loaded Successfully!")


# =====================================
# 2. PREDICTION FUNCTION
# =====================================

def predict_team(
    request_text,
    channel,
    product_family,
    warranty_status,
    source,
    request_datetime
):

    try:
        # Validate customer request
        if not request_text or not request_text.strip():
            return "Please enter a customer request."

        if not request_datetime:
            return "Please enter the request date and time."

        # Convert timestamp
        dt = pd.to_datetime(request_datetime, errors="raise")

        # =====================================
        # 3. FEATURE ENGINEERING
        # Same logic as model training
        # =====================================

        request_hour = dt.hour

        request_day = dt.dayofweek

        request_month = dt.month

        request_day_of_month = dt.day

        is_weekend = int(request_day >= 5)

        # Original training time-period logic
        if 5 <= request_hour < 12:
            time_period = "Morning"

        elif 12 <= request_hour < 17:
            time_period = "Afternoon"

        elif 17 <= request_hour < 21:
            time_period = "Evening"

        else:
            time_period = "Night"

        # =====================================
        # 4. PREPARE MODEL INPUT
        # =====================================

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

        # Ensure exact training feature order
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

        # =====================================
        # 5. MODEL PREDICTION
        # =====================================

        prediction = model.predict(input_data)[0]

        # =====================================
        # 6. CONFIDENCE MARGIN
        # =====================================

        decision_scores = model.decision_function(input_data)

        sorted_scores = sorted(
            decision_scores[0],
            reverse=True
        )

        confidence_margin = (
            sorted_scores[0] - sorted_scores[1]
        )

        # This is a decision-score margin,
        # not a calibrated probability.

        if confidence_margin < 0.20:
            routing_status = "Human Review Required"
        else:
            routing_status = "Automatic Routing"

        # =====================================
        # 7. DISPLAY RESULT
        # =====================================

        result = f"""
        PREDICTION RESULT

        Recommended Team: {prediction}

        Confidence Margin: {confidence_margin:.4f}

        Routing Status: {routing_status}

        Request Date: {dt.strftime('%d-%m-%Y')}

        Request Time: {dt.strftime('%I:%M %p')}
        """

        return result

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# =====================================
# 8. GRADIO WEB APPLICATION
# =====================================

with gr.Blocks(title="Kestrel Service Routing") as demo:

    gr.Markdown(
        """
        # Kestrel Home Service Request Routing

        ### AI-Powered Customer Service Request Classification

        Enter customer request details to predict
        the responsible service team.
        """
    )

    with gr.Row():

        request_text = gr.Textbox(
            label="Customer Request",
            placeholder="Describe the customer's issue...",
            lines=4
        )

        request_datetime = gr.Textbox(
            label="Request Date and Time",
            value="2026-11-26 10:30"
        )

    with gr.Row():

        channel = gr.Textbox(
            label="Channel",
            value="whatsapp"
        )

        product_family = gr.Textbox(
            label="Product Family",
            value="Water Purifier"
        )

    with gr.Row():

        warranty_status = gr.Textbox(
            label="Warranty Status",
            value="in_warranty"
        )

        source = gr.Textbox(
            label="Source",
            value="crm"
        )

    predict_button = gr.Button(
        "Predict Service Team",
        variant="primary"
    )

    output = gr.Textbox(
        label="Routing Prediction",
        lines=8
    )

    predict_button.click(
        fn=predict_team,
        inputs=[
            request_text,
            channel,
            product_family,
            warranty_status,
            source,
            request_datetime
        ],
        outputs=output
    )

# =====================================
# 9. LAUNCH APPLICATION
# =====================================

if __name__ == "__main__":
    demo.launch()
