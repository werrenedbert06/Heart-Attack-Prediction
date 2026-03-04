import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("preprocessor.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model


def make_prediction(features, scaler, model):
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    input_df   = pd.DataFrame([features], columns=feature_names)
    X_scaled   = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]
    return prediction, proba


def main():
    scaler, model = load_artifacts()

    st.title('Machine Learning Heart Attack Risk Detection (for age 25 to 80 only)')

    age = st.number_input('Umur', min_value=25, max_value=80, value=40, step=1)

    gender_map = {"Female": 0, "Male": 1}
    sex = gender_map[st.selectbox('Gender', options=list(gender_map.keys()))]

    cp = st.radio('Tingkat nyeri dada (0 = tidak ada, 3 = sangat parah)', [0, 1, 2, 3])

    trestbps = st.slider('Tekanan darah', min_value=94.0, max_value=200.0, value=120.0, step=0.1)
    chol     = st.slider('Cholesterol', min_value=126.0, max_value=564.0, value=200.0, step=0.1)

    blood_sugar_bin = {"Yes": 1, "No": 0}
    fbs = blood_sugar_bin[st.selectbox("Apakah anda memiliki masalah gula darah?", options=list(blood_sugar_bin.keys()))]

    ECG_Result = {
        "Normal (Normal Sinus Rhythm)": 0,
        "Abnormal tapi Stabil (Borderline/Arrhythmia)": 1,
        "Gawat Darurat (Critical/Lethal Arrhythmia)": 2
    }
    restecg = ECG_Result[st.selectbox('Hasil ECG', options=list(ECG_Result.keys()))]

    thalach = st.number_input('Detak jantung maksimal saat beraktivitas', min_value=70.0, max_value=202.0, value=100.0, step=0.1)

    exang_bin = {"Yes": 1, "No": 0}
    exang = exang_bin[st.selectbox("Apakah Anda mengalami nyeri dada saat olahraga (Exercise Angina)?", options=list(exang_bin.keys()))]

    oldpeak = st.slider('ST Depression', min_value=0.0, max_value=6.2, value=0.8, step=0.1)
    slope   = st.radio("ST Slope", [0, 1, 2])

    ca = st.selectbox(
        "Jumlah Pembuluh Darah Utama (ca) yang Terdeteksi Fluoroskopi:",
        options=[0, 1, 2, 3],
        help="0 berarti normal, 1-3 menunjukkan jumlah pembuluh darah yang tersumbat."
    )

    thal_ord = {
        "Null": 0,
        "Fixed Defect (Bekas Luka/Serangan Jantung Lama)": 1,
        "Normal (Aliran Darah Lancar)": 2,
        "Reversible Defect (Penyempitan Saat Aktivitas)": 3
    }
    thal = thal_ord[st.selectbox(
        "Hasil Tes Thallium (thal):",
        options=list(thal_ord.keys()),
        help="Menunjukkan kondisi aliran darah ke otot jantung."
    )]

    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result, proba = make_prediction(features, scaler, model)

        label = "🔴 High Risk Heart Attack" if result == 1 else "🟢 Low Risk Heart Attack"
        st.success(f'The prediction is: {label}')

        st.subheader("Prediction Confidence")
        df_proba = pd.DataFrame({"Class": model.classes_, "Probability": proba})
        st.bar_chart(df_proba.set_index("Class"))


if __name__ == '__main__':
    main()

