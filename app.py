import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes AI Project", layout="wide")

# โหลดข้อมูล
@st.cache_data
def load_data():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('diabetes.csv', names=names)
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = SVC(kernel='linear').fit(X_scaled, y)

# --- Sidebar เมนู ---
st.sidebar.title("📌 เมนูรายงานโปรเจ็ค")
menu = st.sidebar.radio("เลือกหัวข้อนำเสนอ:", 
    ["หน้าหลัก", "1. การกำหนดปัญหา", "2. Data Preprocessing", "3. สร้างโมเดล ML", "4. ประเมินโมเดล", "5. โปรแกรมใช้งาน"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
st.sidebar.write("รหัส: [ใส่รหัส]")
st.sidebar.write("ชื่อ: [ใส่ชื่อ]")
st.sidebar.write("หมู่เรียน: [ใส่หมู่เรียน]")

# --- หน้า Content ตามเมนู ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction Project")
    st.write("ยินดีต้อนรับสู่รายงานสรุปผลโปรเจ็คทำนายโรคเบาหวานด้วย Machine Learning")
    st.info("ใช้เมนูด้านซ้ายมือเพื่อเลือกดูเนื้อหาตามหัวข้อที่กำหนด 1-5")

elif menu == "1. การกำหนดปัญหา":
    st.subheader("1. การกำหนดปัญหาและ Dataset")
    st.write("- **ปัญหา:** เบาหวานเป็นโรคที่ต้องคัดกรองเบื้องต้นเพื่อลดความเสี่ยง")
    st.write("- **Dataset:** ใช้ Pima Indians Diabetes Database เนื่องจากมีความเป็นมาตรฐานและตัวแปรครอบคลุมสุขภาพ")

elif menu == "2. Data Preprocessing":
    st.subheader("2. Data Preprocessing")
    st.write("มีการจัดการข้อมูลดังนี้:")
    st.code("df[cols].replace(0, np.nan) # แก้ไขค่า 0 ที่เป็นข้อมูลผิดพลาด\ndf.fillna(df.mean()) # เติมค่าว่างด้วยค่าเฉลี่ย", language='python')
    st.dataframe(df.head())

elif menu == "3. สร้างโมเดล ML":
    st.subheader("3. การสร้างโมเดล Machine Learning")
    st.write("เลือกใช้ **Support Vector Machine (SVM)**")
    st.latex(r'''หลักการ: หาเส้นแบ่ง (Hyperplane) ที่สร้างระยะห่าง (Margin) ระหว่างข้อมูลสองกลุ่มให้กว้างที่สุด''')

elif menu == "4. ประเมินโมเดล":
    st.subheader("4. การประเมินและเปรียบเทียบโมเดล")
    st.metric("ความแม่นยำ (Accuracy)", f"{accuracy_score(y, model.predict(X_scaled))*100:.2f}%")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(y, model.predict(X_scaled)), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

elif menu == "5. โปรแกรมใช้งาน":
    st.subheader("5. Streamlit Application")
    st.write("กรอกข้อมูลด้านล่างเพื่อทำนายผล")
    col1, col2 = st.columns(2)
    p = col1.number_input('ตั้งครรภ์', 0, 20, 1)
    g = col1.number_input('ระดับน้ำตาล', 0, 200, 120)
    bp = col2.number_input('ความดัน', 0, 140, 70)
    bmi = col2.number_input('BMI', 0.0, 70.0, 25.0)
    
    if st.button("ทำนายผล"):
        res = model.predict(scaler.transform([[p,g,bp,20,79,bmi,0.5,30]]))
        if res[0] == 1: st.error("มีความเสี่ยง")
        else: st.success("ปกติ")