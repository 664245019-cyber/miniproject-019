import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Diabetes AI Report", page_icon="🩺", layout="centered")

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

# --- Sidebar ---
st.sidebar.title("📑 เมนูโปรเจ็ค")
# ใช้ Selectbox แทน Radio Button (ดูคลีนกว่า)
menu = st.sidebar.selectbox("เลือกหัวข้อนำเสนอ:", 
    ["หน้าหลัก", "1. การกำหนดปัญหา", "2. Data Preprocessing", "3. สร้างโมเดล ML", "4. ประเมินโมเดล", "5. โปรแกรมใช้งาน"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 ข้อมูลผู้พัฒนา")
st.sidebar.caption("รหัส: [ใส่รหัสของคุณ]")
st.sidebar.caption("ชื่อ: [ใส่ชื่อของคุณ]")
st.sidebar.caption("หมู่เรียน: [ใส่หมู่เรียนของคุณ]")

# --- Main Content ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction Project")
    st.success("ยินดีต้อนรับสู่รายงานสรุปผล Machine Learning")
    st.write("เลือกหัวข้อจากเมนูด้านซ้ายเพื่อเริ่มต้นการนำเสนอ")

elif menu == "1. การกำหนดปัญหา":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.info("ปัญหา: การตรวจคัดกรองเบาหวานเบื้องต้นช่วยลดภาวะแทรกซ้อนได้")
    st.write("**Dataset:** เลือกใช้ Pima Indians Diabetes Database เนื่องจากเป็นชุดข้อมูลมาตรฐานสากลที่มีตัวแปรทางสุขภาพครบถ้วน")

elif menu == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.write("ขั้นตอนการเตรียมข้อมูล:")
    st.code("1. แทนที่ค่า 0 ด้วยค่าเฉลี่ย (Mean)\n2. ทำ Feature Scaling ด้วย StandardScaler", language='python')
    st.dataframe(df.head(), use_container_width=True)

elif menu == "3. สร้างโมเดล ML":
    st.header("3. การสร้างโมเดล Machine Learning")
    st.write("เลือกใช้โมเดล **Support Vector Machine (SVM)**")
    st.write("ทฤษฎี: ใช้ Hyperplane แยกข้อมูลสองกลุ่ม (เป็น/ไม่เป็นเบาหวาน) โดยเพิ่มระยะ Margin ให้กว้างที่สุดเพื่อความแม่นยำ")

elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    acc = accuracy_score(y, model.predict(X_scaled))
    st.metric("ความแม่นยำ (Accuracy)", f"{acc*100:.2f}%")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(pd.crosstab(y, model.predict(X_scaled)), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

elif menu == "5. โปรแกรมใช้งาน":
    st.header("5. Streamlit Application")
    st.write("ทดลองกรอกข้อมูลเพื่อทำนายผล")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        p = col1.number_input('จำนวนครั้งที่ตั้งครรภ์', 0, 20, 1)
        g = col1.number_input('ระดับน้ำตาล', 0, 200, 120)
        bp = col2.number_input('ความดัน', 0, 140, 70)
        bmi = col2.number_input('BMI', 0.0, 70.0, 25.0)
        submit = st.form_submit_button("ทำนายผลความเสี่ยง")
        
    if submit:
        res = model.predict(scaler.transform([[p,g,bp,20,79,bmi,0.5,30]]))
        if res[0] == 1: st.error("⚠️ พบความเสี่ยงเป็นโรคเบาหวาน")
        else: st.success("✅ ไม่พบความเสี่ยง (ปกติ)")