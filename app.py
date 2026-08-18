import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเว็บให้ดูกว้างขวางและสบายตา
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# --- 1. ส่วนข้อมูลผู้พัฒนา (Sidebar) ---
st.sidebar.markdown("## 👨‍💻 ข้อมูลผู้พัฒนา")
# st.sidebar.image("profile.jpg", width=120) # (ถ้ามีรูป นำไฟล์รูปชื่อ profile.jpg มาใส่ร่วมในโฟลเดอร์ได้ครับ)
st.sidebar.markdown("""
- **รหัสประจำตัว:** [664245019]
- **ชื่อ-นามสกุล:** [นายคณิศร จันทรสูตร]
- **หมู่เรียน:** [66/43]
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ กำหนดค่าข้อมูลสุขภาพ")

# --- โหลดและเตรียมข้อมูล (Caching เพื่อความเร็ว) ---
@st.cache_data
def load_data():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv('diabetes.csv', names=names)
    # Preprocessing: จัดการค่า 0
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    return df

try:
    df = load_data()
except:
    st.error("⚠️ ไม่พบไฟล์ diabetes.csv กรุณาอัปโหลดไฟล์ dataset ไว้ใน GitHub พร้อมกับ app.py")

# --- ส่วนหัวของเว็บไซต์ ---
st.title("🩺 Diabetes Prediction Web Application")
st.markdown("ระบบวิเคราะห์และทำนายความเสี่ยงโรคเบาหวานด้วย Machine Learning (Support Vector Machine)")
st.markdown("---")

# แบ่งหน้าจอเป็น 2 คอลัมน์หลักเพื่อให้ดูสะอาดตา
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 1. การกำหนดปัญหาและ Dataset")
    st.write("ชุดข้อมูลที่ใช้คือ **Pima Indians Diabetes Database** ซึ่งมีความเหมาะสมในการจำแนกความเสี่ยงโรคเบาหวาน ตัวอย่างข้อมูล 5 แถวแรกหลังผ่านกระบวนการ Preprocessing:")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("🛠️ 2. Data Preprocessing & Model")
    st.info("""
    - **Preprocessing:** จัดการค่าศูนย์ (0) ที่ผิดปกติในข้อมูลสุขภาพด้วยการแทนที่ค่าเฉลี่ย (Mean)
    - **Feature Scaling:** ปรับสเกลข้อมูลด้วย `StandardScaler`
    - **Machine Learning:** ใช้โมเดล **SVM (Support Vector Machine)**
    """)

# --- เทรนโมเดล ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.markdown("---")

# --- ส่วนประเมินผลโมเดล (ข้อ 4) ---
st.subheader("📈 4. การประเมินและเปรียบเทียบโมเดล")
m_col1, m_col2 = st.columns([1, 2])

with m_col1:
    st.metric(label="ความแม่นยำของโมเดล (Accuracy)", value=f"{acc*100:.2f}%")
    st.write("โมเดลมีความสามารถในการทำนายผลลัพธ์ได้อย่างมีประสิทธิภาพ เหมาะสำหรับการนำมาทำระบบคัดกรองเบื้องต้น")

with m_col2:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

st.markdown("---")

# --- ส่วนทดลองใช้งาน (ข้อ 5) ---
st.subheader("💻 5. Streamlit Application (ทดลองกรอกข้อมูลทำนายผล)")
st.write("กรอกข้อมูลสุขภาพของคุณที่แถบด้านซ้าย (Sidebar) แล้วกดปุ่มทำนายด้านล่างได้เลยครับ")

# รับค่าจาก Sidebar สำหรับทำนาย
def user_input_features():
    pregnancies = st.sidebar.number_input('จำนวนครั้งที่ตั้งครรภ์ (Pregnancies)', 0, 20, 1)
    glucose = st.sidebar.number_input('ระดับน้ำตาลในเลือด (Glucose)', 0, 200, 110)
    bp = st.sidebar.number_input('ความดันโลหิต (BloodPressure)', 0, 140, 70)
    skin = st.sidebar.number_input('ความหนาผิวหนัง (SkinThickness)', 0, 100, 20)
    insulin = st.sidebar.number_input('ระดับอินซูลิน (Insulin)', 0, 900, 79)
    bmi = st.sidebar.number_input('ดัชนีมวลกาย (BMI)', 0.0, 70.0, 24.5)
    dpf = st.sidebar.number_input('ประวัติครอบครัว (DiabetesPedigreeFunction)', 0.0, 2.5, 0.45)
    age = st.sidebar.number_input('อายุ (Age)', 1, 100, 28)
    
    data = {'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp, 
            'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi, 
            'DiabetesPedigreeFunction': dpf, 'Age': age}
    return pd.DataFrame([data])

input_df = user_input_features()

# ปุ่มกดทำนายตรงกลางจอสวยๆ
if st.button("🚀 คลิกเพื่อทำนายผลความเสี่ยง", use_container_width=True):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("🚨 **ผลการทำนาย:** มีความเสี่ยงเป็นโรคเบาหวาน (ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยอย่างละเอียด)")
    else:
        st.success("✅ **ผลการทำนาย:** ปกติ (ไม่พบความเสี่ยงโรคเบาหวาน)")