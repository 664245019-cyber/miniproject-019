import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ปรับแต่งหน้าเว็บให้ดูสะอาด (Modern Look)
st.set_page_config(page_title="Diabetes AI Predictor", page_icon="🩺", layout="wide")

# เพิ่ม CSS ให้ดูสวยขึ้นเล็กน้อย
st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันโหลดข้อมูล (Cache เพื่อความเร็ว)
@st.cache_data
def load_data():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('diabetes.csv', names=names)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()

# 3. Sidebar (ข้อมูลผู้พัฒนา & ปรับตั้งค่า)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100) # ไอคอนสุขภาพ
    st.title("User Profile")
    st.write("**รหัส:** [ุ664245019]")
    st.write("**ชื่อ:** [นายคณิศร จันทรสูตร]")
    st.write("**หมู่เรียน:** [664245019]")
    st.markdown("---")
    st.header("📋 ป้อนข้อมูลสุขภาพ")
    # ปรับ Input ให้ใช้ง่ายขึ้น
    p = st.slider('จำนวนครั้งที่ตั้งครรภ์', 0, 17, 1)
    g = st.number_input('ระดับน้ำตาลในเลือด', 0, 200, 120)
    bp = st.number_input('ความดันโลหิต', 0, 140, 70)
    st_val = st.number_input('ความหนาผิวหนัง', 0, 100, 20)
    ins = st.number_input('อินซูลิน', 0, 900, 79)
    bmi = st.number_input('ดัชนีมวลกาย (BMI)', 0.0, 70.0, 25.0)
    dpf = st.number_input('ประวัติครอบครัว (DPF)', 0.0, 2.5, 0.45)
    age = st.slider('อายุ', 1, 100, 30)

# --- ส่วนเนื้อหาหลัก ---
st.title("🩺 Diabetes AI Prediction")
st.markdown("ใช้โมเดล **SVM (Support Vector Machine)** วิเคราะห์ความเสี่ยงโรคเบาหวานอย่างแม่นยำ")

# แสดงผลแบบ Tab เพื่อความเป็นระเบียบ (เทรนด์เว็บสมัยใหม่)
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Preprocessing", "📈 Model Performance", "🚀 ทำนายความเสี่ยง"])

with tab1:
    st.subheader("ข้อมูลสุขภาพและการเตรียมข้อมูล")
    st.write("ชุดข้อมูลที่ผ่านการจัดการค่าศูนย์ (Missing Values) เรียบร้อยแล้ว")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.subheader("ประสิทธิภาพของโมเดล (SVM)")
    # Train Model แบบด่วน
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    model = SVC(kernel='linear')
    model.fit(scaler.fit_transform(X_train), y_train)
    acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
    
    col_a, col_b = st.columns(2)
    col_a.metric("ความแม่นยำ (Accuracy)", f"{acc*100:.2f}%")
    col_b.write("กราฟ Confusion Matrix แสดงการจำแนกประเภท:")
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(y_test, model.predict(scaler.transform(X_test))), annot=True, fmt='d', cmap='Greens')
    st.pyplot(fig)

with tab3:
    st.subheader("ประเมินความเสี่ยงรายบุคคล")
    if st.button("ประมวลผลความเสี่ยง"):
        input_data = pd.DataFrame([[p, g, bp, st_val, ins, bmi, dpf, age]], 
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        prediction = model.predict(scaler.transform(input_data))
        
        if prediction[0] == 1:
            st.error("⚠️ พบความเสี่ยง: ผลการวิเคราะห์บ่งชี้ว่ามีความเสี่ยงเป็นเบาหวาน")
        else:
            st.success("✅ ปกติ: ผลการวิเคราะห์ไม่พบความเสี่ยงเบาหวาน")