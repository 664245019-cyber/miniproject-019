import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes AI Report", page_icon="🩺", layout="wide")

# เพิ่มสีสันให้ Sidebar และเมนู
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

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
model = SVC(kernel='linear').fit(scaler.fit_transform(X), y)

# --- Sidebar (Bar Menu) ---
with st.sidebar:
    st.title("🩺 Project Menu")
    menu = st.radio(
        "เลือกหัวข้อนำเสนอ:",
        ["หน้าหลัก", "1. ปัญหาและ Dataset", "2. Data Preprocessing", "3. สร้างโมเดล ML", "4. ประเมินโมเดล", "5. เว็บแอปใช้งาน"],
        index=0
    )
    st.markdown("---")
    st.subheader("👨‍💻 ข้อมูลผู้พัฒนา")
    st.caption("รหัส: [664245019]")
    st.caption("ชื่อ: [นายคณิศร จันทรสูตร]")
    st.caption("หมู่เรียน: [66/43]")

# --- หน้า Content ---
if menu == "หน้าหลัก":
    st.title("Welcome to Diabetes Prediction Project")
    st.write("เลือกหัวข้อจากแถบด้านซ้ายเพื่อดูรายละเอียดรายงาน")
elif menu == "1. ปัญหาและ Dataset":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.info("ใช้ Pima Indians Diabetes Database วิเคราะห์ความเสี่ยงเบาหวาน")
elif menu == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.write("ขั้นตอน: การทำความสะอาดข้อมูล (Handling Zero Values) และ Feature Scaling")
    st.dataframe(df.head(), use_container_width=True)
elif menu == "3. สร้างโมเดล ML":
    st.header("3. การสร้างโมเดล ML")
    st.write("โมเดลที่ใช้: **Support Vector Machine (SVM)**")
    st.latex(r"Minimize: \frac{1}{2}||w||^2")
elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    acc = accuracy_score(y, model.predict(scaler.transform(X)))
    st.metric("ความแม่นยำ (Accuracy)", f"{acc*100:.2f}%")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(pd.crosstab(y, model.predict(scaler.transform(X))), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
elif menu == "5. เว็บแอปใช้งาน":
    st.header("5. Streamlit Application")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        p = col1.number_input('ตั้งครรภ์', 0, 20, 1)
        g = col1.number_input('ระดับน้ำตาล', 0, 200, 120)
        bp = col2.number_input('ความดัน', 0, 140, 70)
        bmi = col2.number_input('BMI', 0.0, 70.0, 25.0)
        if st.form_submit_button("ทำนายผล"):
            res = model.predict(scaler.transform([[p,g,bp,20,79,bmi,0.5,30]]))
            if res[0] == 1: st.error("เสี่ยงเบาหวาน")
            else: st.success("ปกติ")