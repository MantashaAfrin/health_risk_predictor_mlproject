import streamlit as st
import pickle
import numpy as np
import plotly.express as px

model=pickle.load(open("health_predictor.pkl","rb"))
#the model we made is stored in this model variable

encoders=pickle.load(open("label_encoders.pkl","rb"))

st.title("HEALTH RISK PREDICTOR")

age=st.slider("Age",18,80,22)#("Age",min age,max age,default age)
diet=st.selectbox("Diet Quality",['Poor','Average','Good'])
exercise=st.slider("Exercise Day Per Week",0,7,2)
sleep=st.slider("Sleep Hours",2,12,6)
stress=st.selectbox("Stress Level",['Low','Medium','High'])
bmi=st.number_input("BMI",10.0,40.0,20.0)
smoking=st.selectbox("Smoking",["Yes","No"])
alcohol=st.selectbox("Alcohol consumption",['Low','Medium','High'])
family_history=st.selectbox("Family History of Disease",["Yes","No"])

if st.button("Predict Risk"):
    input_data=[age,
            encoders['diet'].transform([diet])[0],
            exercise,
            sleep,
            encoders['stress'].transform([stress])[0],
            bmi,
            encoders['smoking'].transform([smoking])[0],
            encoders['alcohol'].transform([alcohol])[0],
            encoders['family_history'].transform([family_history])[0],
            ]
    
    prediction=model.predict([input_data])
    probs=model.predict([input_data])

    risk_label=encoders['risk_level'].inverse_transform([prediction[0]])[0]

    st.success(risk_label)

#Bar chart for lifestyle factors

    # ---------------- NUMERIC FACTORS ----------------

numeric_factors = {
    "BMI": bmi,
    "Exercise": exercise,
    "Sleep": sleep
}

numeric_fig = px.bar(
    x=list(numeric_factors.keys()),
    y=list(numeric_factors.values()),
    labels={"x": "Factors", "y": "Value"},
    title="Your Numeric Health Factors"
)

st.plotly_chart(numeric_fig, use_container_width=True)


# ---------------- CATEGORICAL FACTORS ----------------

categorical_factors = {
    "Diet": diet,
    "Stress": stress,
    "Smoking": smoking,
    "Alcohol": alcohol,
    "Family History": family_history
}

st.subheader("Your Lifestyle Factors")

for factor, value in categorical_factors.items():
    st.write(f"**{factor}:** {value}")