import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc

def show_modeling():
    st.title("🧠 Step 3: Modeling")

    if st.session_state.get("processed_data") is None:
        st.warning("⚠️ Please complete **2. Pre-processing & Feature Eng** first to prepare the data for modeling.")
        return

    df = st.session_state["processed_data"]
    target_col = st.session_state["target_col"]
    problem_type = st.session_state["problem_type"]

    st.markdown("""
    **Modeling** is the core of Machine Learning. It's where an algorithm learns patterns from the historical data (features) to predict the outcome (target).
    """)

    # 1. Train/Test Split
    st.header("1. Train/Test Split")
    st.markdown("We can't evaluate our model on the exact same data it learned from (it could just memorize the answers!). Instead, we split the data into a **Training Set** (to learn from) and a **Testing Set** (to evaluate performance on unseen data).")

    test_size = st.slider("Select Test Size (Percentage)", min_value=10, max_value=50, value=20, step=5)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Splitting Code")
        code = f'''from sklearn.model_selection import train_test_split

X = df.drop('{target_col}', axis=1)
y = df['{target_col}']

# Perform a {100-test_size}/{test_size} split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size/100}, random_state=42)
'''
        st.code(code, language='python')

    with col2:
        st.markdown("### Visual Split")

        # Plot pie chart of the split
        labels = ['Training Set', 'Testing Set']
        values = [100 - test_size, test_size]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text="Train/Test Data Proportion")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Perform Split"):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success(f"✅ Data split successfully! Training Size: {X_train.shape[0]} rows. Testing Size: {X_test.shape[0]} rows.")

    st.markdown("---")

    # 2. Select and Train Model
    st.header("2. Model Selection & Training")

    if st.session_state.get("X_train") is None:
        st.info("👆 Please click **Perform Split** above before training a model.")
        return

    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
        }

    selected_model_name = st.selectbox("Choose a Machine Learning Algorithm:", list(models.keys()))

    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown("### Training Code")
        if selected_model_name == "Linear Regression":
            model_import = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()"
        elif selected_model_name == "Decision Tree Regressor":
            model_import = "from sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor(random_state=42)"
        elif selected_model_name == "Random Forest Regressor":
            model_import = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)"
        elif selected_model_name == "Logistic Regression":
            model_import = "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)"
        elif selected_model_name == "Naive Bayes (Gaussian)":
            model_import = "from sklearn.naive_bayes import GaussianNB\nmodel = GaussianNB()"
        elif selected_model_name == "Decision Tree Classifier":
            model_import = "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(random_state=42)"
        elif selected_model_name == "Random Forest Classifier":
            model_import = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)"

        code = f'''{model_import}

# Train the model on the Training Data
model.fit(X_train, y_train)

# Make predictions on the unseen Testing Data
predictions = model.predict(X_test)
'''
        st.code(code, language='python')

    with col4:
        st.markdown("### Training Execution")

        if st.button("Train Model & Evaluate"):
            model = models[selected_model_name]

            # Train the model
            model.fit(st.session_state["X_train"], st.session_state["y_train"])

            # Predict
            predictions = model.predict(st.session_state["X_test"])

            # Save for deployment
            st.session_state["model"] = model
            st.session_state["model_name"] = selected_model_name

            st.success(f"🎉 **{selected_model_name}** trained successfully!")

            # Evaluation Metrics
            st.markdown("### 📊 Evaluation Metrics")
            if problem_type == "regression":
                mse = mean_squared_error(st.session_state["y_test"], predictions)
                r2 = r2_score(st.session_state["y_test"], predictions)
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**R-squared ($R^2$):** {r2:.4f} (Closer to 1.0 is better)")

                # Actual vs Predicted Plot
                eval_df = pd.DataFrame({
                    'Actual': st.session_state["y_test"],
                    'Predicted': predictions
                })
                fig = px.scatter(eval_df, x='Actual', y='Predicted', title="Actual vs. Predicted Values", opacity=0.6)

                # Add a perfect prediction line
                min_val = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
                max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
                fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                              line=dict(color="Red", dash="dash"))

                st.plotly_chart(fig, use_container_width=True)

            else:
                acc = accuracy_score(st.session_state["y_test"], predictions)
                st.write(f"**Accuracy:** {acc * 100:.2f}%")

                # Confusion Matrix
                # Get all possible classes the model knows about to ensure matrix size matches labels
                classes = model.classes_
                cm = confusion_matrix(st.session_state["y_test"], predictions, labels=classes)

                fig = px.imshow(cm, text_auto=True,
                                x=classes, y=classes,
                                labels=dict(x="Predicted Diagnosis", y="Actual Diagnosis"),
                                title="Confusion Matrix", color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance (if applicable)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    features = st.session_state["X_train"].columns

                    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)

                    fig2 = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                  title="Top 10 Feature Importances")
                    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig2, use_container_width=True)

            st.info("The model is trained and saved! Proceed to **4. Deployment** to make live predictions on new data.")