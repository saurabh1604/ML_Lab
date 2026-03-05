import streamlit as st
import pandas as pd
import numpy as np

def show_deployment():
    st.title("🚀 Step 4: Deployment")

    if st.session_state.get("model") is None:
        st.warning("⚠️ Please complete **3. Modeling** and train a model first.")
        return

    model = st.session_state["model"]
    model_name = st.session_state["model_name"]
    raw_df = st.session_state["raw_data"]
    target_col = st.session_state["target_col"]
    problem_type = st.session_state["problem_type"]
    encoded_columns = st.session_state.get("encoded_columns", [])

    scaler = st.session_state.get("pipeline_scaler")
    num_cols = st.session_state.get("num_cols_pipeline", [])
    cat_cols = st.session_state.get("cat_cols_pipeline", [])

    st.markdown("""
    **Deployment** is the final stage of the ML pipeline. Your model is trained and ready! Now, we can integrate it into an application so users (or other systems) can provide new, unseen data and get predictions in real-time.

    Use the interactive widgets below to input new data for prediction.
    """)

    st.header(f"🔮 Live Prediction: {model_name}")

    # Create input widgets based on the RAW data columns
    # We want the user to input data in its natural, unscaled format.
    user_inputs = {}

    # Determine which columns we need to ask for
    input_features = [col for col in raw_df.columns if col != target_col]

    st.markdown("### 📝 Enter New Data Points:")

    # Create rows of 3 columns to organize widgets
    cols = st.columns(3)

    for i, feature in enumerate(input_features):
        col = cols[i % 3]

        with col:
            # Check if feature is numerical or categorical
            if raw_df[feature].dtype in ['float64', 'int64']:
                min_val = float(raw_df[feature].min())
                max_val = float(raw_df[feature].max())
                mean_val = float(raw_df[feature].mean())

                # Use a slider for numerical inputs
                user_inputs[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.2f" if raw_df[feature].dtype == 'float64' else "%d"
                )
            else:
                # Use a selectbox for categorical inputs
                unique_vals = raw_df[feature].dropna().unique().tolist()
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=unique_vals
                )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Deployment Code (Inference)")
        code = f'''# 1. Capture user inputs
new_data = pd.DataFrame([user_inputs])

# 2. Apply the exact same pre-processing as training
# (Encoding & Scaling)
processed_data = preprocess_pipeline(new_data)

# 3. Make the Prediction
prediction = model.predict(processed_data)

print("Prediction:", prediction[0])
'''
        st.code(code, language='python')

    with col2:
        st.markdown("### Real-time Prediction Output")

        if st.button("🔮 Make Prediction", type="primary", use_container_width=True):

            # Convert user inputs to DataFrame
            input_df = pd.DataFrame([user_inputs])

            # 1. Apply Categorical Encoding (One-Hot)
            if cat_cols:
                # We need to ensure all possible columns exist after encoding
                # Because the input data only has one row, get_dummies won't create all columns
                input_encoded = pd.get_dummies(input_df, columns=cat_cols)

                # Reindex to ensure we have exactly the same columns as the training data
                input_processed = input_encoded.reindex(columns=encoded_columns, fill_value=0)
            else:
                input_processed = input_df[encoded_columns]

            # 2. Apply Scaling
            if scaler and num_cols:
                # Only scale the numerical columns
                # Make sure the columns exist before scaling
                valid_num_cols = [c for c in num_cols if c in input_processed.columns]
                if valid_num_cols:
                    input_processed[valid_num_cols] = scaler.transform(input_processed[valid_num_cols])

            # 3. Predict
            prediction = model.predict(input_processed)[0]

            st.markdown("### 🎯 Result:")
            if problem_type == "regression":
                # Assuming price or similar
                st.success(f"## {prediction:,.2f}")
                st.balloons()
            else:
                # Classification diagnosis
                if isinstance(prediction, (int, np.integer)):
                    # Try to map back if needed, but in our case strings are already kept
                    pred_str = str(prediction)
                else:
                    pred_str = str(prediction)

                st.success(f"## {pred_str}")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_processed)[0]
                    classes = model.classes_

                    st.markdown("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': [f"{p*100:.2f}%" for p in probs]
                    })
                    st.dataframe(prob_df, hide_index=True)

                    if max(probs) > 0.8:
                        st.balloons()
