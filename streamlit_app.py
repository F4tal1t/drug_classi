import streamlit as st
# Streamlit Application
def main():
    st.title("Drug Classification Prediction")

    st.sidebar.header("Input Features")

    # User input for features
    age = st.sidebar.slider("Age", int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].mean()))
    sex = st.sidebar.selectbox("Sex", le_sex.classes_)
    bp = st.sidebar.selectbox("Blood Pressure (BP)", le_bp.classes_)
    cholesterol = st.sidebar.selectbox("Cholesterol", le_cholesterol.classes_)
    na_to_k = st.sidebar.number_input("Sodium to Potassium Ratio (Na_to_K)", float(X['Na_to_K'].min()), float(X['Na_to_K'].max()), float(X['Na_to_K'].mean()))

    # Load model and encoders
    with open("drug_classifier_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("label_encoders.pkl", "rb") as encoders_file:
        encoders = pickle.load(encoders_file)

    # Preprocess user inputs
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "BP": [encoders["BP"].transform([bp])[0]],
        "Cholesterol": [encoders["Cholesterol"].transform([cholesterol])[0]],
        "Na_to_K": [na_to_k]
    })

    # Prediction
    if st.button("Predict Drug Class"):
        prediction = model.predict(input_data)
        predicted_drug = encoders["Drug"].inverse_transform(prediction)
        st.success(f"Predicted Drug Class: {predicted_drug[0]}")

if __name__ == "__main__":
    main()
