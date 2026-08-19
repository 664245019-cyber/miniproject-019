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

# --- หน้า Content ที่เพิ่มเนื้อหาให้อัดแน่น ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction Project")
    st.markdown("### รายงานสรุปผลโปรเจ็ค Machine Learning เพื่อการคัดกรองเบาหวาน")
    st.write("โปรเจ็คนี้พัฒนาขึ้นเพื่อประยุกต์ใช้โมเดล SVM ในการทำนายความเสี่ยงโรคเบาหวานโดยใช้ข้อมูลสุขภาพที่เป็นมาตรฐานสากล")
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", use_container_width=True)

elif menu == "1. ปัญหาและ Dataset":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.markdown("""
    **ปัญหาที่พบ:** โรคเบาหวานเป็นภัยเงียบที่ไม่แสดงอาการในช่วงแรก การตรวจพบเร็วช่วยลดค่าใช้จ่ายในการรักษา
    **Dataset ที่เลือกใช้:** Pima Indians Diabetes Dataset
    * **จำนวนตัวอย่าง:** 768 รายการ
    * **ตัวแปรต้น (Features):** 8 ตัวแปรสุขภาพ ได้แก่ Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    * **ตัวแปรตาม (Target):** Outcome (0 = ปกติ, 1 = เป็นเบาหวาน)
    **เหตุผล:** ข้อมูลชุดนี้มีความสมบูรณ์ทางสถิติและเป็นที่ยอมรับในการเรียนรู้ Machine Learning ทั่วโลก
    """)

elif menu == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.write("ข้อมูลดิบผ่านกระบวนการทำความสะอาดอย่างเป็นระบบ:")
    st.table(pd.DataFrame({
        "ขั้นตอน": ["จัดการ Missing Values", "จัดการค่า 0 ผิดปกติ", "Feature Scaling"],
        "วิธีการ": ["ใช้การเติมค่าเฉลี่ย (Mean Imputation)", "เปลี่ยนค่า 0 ในคอลัมน์สุขภาพเป็น NaN", "ใช้ StandardScaler ปรับสเกลข้อมูลให้เป็นมาตรฐาน"],
        "ผลลัพธ์": ["ข้อมูลไม่สูญหาย", "ข้อมูลสะอาดขึ้น", "ลดความเหลื่อมล้ำของข้อมูล"]
    }))
    st.write("ตัวอย่างข้อมูลที่ผ่านการทำ Preprocessing แล้ว:")
    st.dataframe(df.head(10), use_container_width=True)

elif menu == "3. สร้างโมเดล ML":
    st.header("3. การสร้างโมเดล Machine Learning")
    st.markdown("""
    **อัลกอริทึม:** Support Vector Machine (SVM)
    * **ทำไมต้อง SVM:** เพราะมีประสิทธิภาพสูงในการแยกกลุ่มข้อมูล (Classification) ที่มีความซับซ้อนและข้อมูลไม่เป็นเส้นตรง
    * **Linear Kernel:** เราเลือกใช้ Linear Kernel เนื่องจากความสัมพันธ์ของตัวแปรมีความเป็นเส้นตรงที่ค่อนข้างชัดเจน
    * **Hyperplane:** หลักการคือการหาเส้นแบ่งที่ดีที่สุดระหว่างกลุ่ม Positive และ Negative โดยการรักษาระยะห่าง (Margin) ให้มากที่สุด
    """)

elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    acc = accuracy_score(y, model.predict(scaler.transform(X)))
    st.metric("Accuracy ของโมเดล SVM", f"{acc*100:.2f}%")
    st.write("Confusion Matrix เพื่อดูการทายถูก/ผิด:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pd.crosstab(y, model.predict(scaler.transform(X))), annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
    st.pyplot(fig)
    st.write("จากผลลัพธ์พบว่าโมเดลมีความแม่นยำสูง สามารถใช้เป็นเครื่องมือคัดกรองเบื้องต้นได้จริง")

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