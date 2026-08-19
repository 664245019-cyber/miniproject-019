import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# แบ่งชุดข้อมูลสำหรับเทรนและทดสอบโมเดล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# เทรนโมเดลหลัก (SVM)
model = SVC(kernel='linear').fit(X_train_scaled, y_train)

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
    st.caption("รหัส: 664245019")
    st.caption("ชื่อ: นายคณิศร จันทรสูตร")
    st.caption("หมู่เรียน: 66/43")

# --- หน้า Content ---
if menu == "หน้าหลัก":
    st.title("🩺 Diabetes Prediction Project")
    st.markdown("### รายงานสรุปผลโปรเจ็ค Machine Learning เพื่อการคัดกรองเบาหวาน")
    st.write("โปรเจ็คนี้พัฒนาขึ้นเพื่อประยุกต์ใช้โมเดล Machine Learning ในการทำนายความเสี่ยงโรคเบาหวานโดยใช้ข้อมูลสุขภาพที่เป็นมาตรฐานสากล")
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
    **อัลกอริทึมหลัก:** Support Vector Machine (SVM)
    * **ทำไมต้อง SVM:** เพราะมีประสิทธิภาพสูงในการแยกกลุ่มข้อมูล (Classification) ที่มีความซับซ้อน
    * **Linear Kernel:** เราเลือกใช้ Linear Kernel เนื่องจากความสัมพันธ์ของตัวแปรมีความเป็นเส้นตรงค่อนข้างชัดเจน
    * **Hyperplane:** หลักการคือการหาเส้นแบ่งที่ดีที่สุดระหว่างกลุ่ม Positive และ Negative โดยการรักษาระยะห่าง (Margin) ให้มากที่สุด
    """)

elif menu == "4. ประเมินโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    st.write("เปรียบเทียบประสิทธิภาพระหว่างโมเดล Machine Learning หลายรูปแบบ:")
    
    # เทรนและวัดผลหลายโมเดลเพื่อให้ตรงโจทย์การเปรียบเทียบ
    models = {
        "SVM (Linear)": SVC(kernel='linear'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }
    
    results = []
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        score = accuracy_score(y_test, m.predict(X_test_scaled))
        results.append({"Model": name, "Accuracy (%)": round(score * 100, 2)})
    
    res_df = pd.DataFrame(results)
    st.table(res_df)
    
    st.write("Confusion Matrix ของโมเดลหลัก (SVM):")
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pd.crosstab(y_test, model.predict(X_test_scaled)), annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    st.pyplot(fig)

elif menu == "5. เว็บแอปใช้งาน":
    st.header("5. Streamlit Application (ระบบทำนายผล)")
    st.write("เลือกโมเดล Machine Learning ที่ต้องการใช้ และกรอกข้อมูลสุขภาพให้ครบถ้วนทั้ง 8 ปัจจัย")
    
    # ทำเป็นปุ่มเลือกเรียงกัน 3 อัน (Horizontal Radio)
    model_choice = st.radio(
        "เลือกโมเดลสำหรับทำนายผล:",
        ["SVM (Linear)", "Random Forest", "Logistic Regression"],
        horizontal=True
    )
    
    # กำหนดโมเดลตามปุ่มที่เลือก
    if model_choice == "SVM (Linear)": 
        active_model = SVC(kernel='linear')
    elif model_choice == "Random Forest": 
        active_model = RandomForestClassifier(random_state=42)
    else: 
        active_model = LogisticRegression(max_iter=500)
    
    active_model.fit(X_train_scaled, y_train)
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            p = st.number_input('จำนวนครั้งที่ตั้งครรภ์ (Pregnancies)', 0, 20, 1)
            g = st.number_input('ระดับน้ำตาลในเลือด (Glucose)', 0, 200, 110)
            bp = st.number_input('ความดันโลหิต (BloodPressure)', 0, 140, 70)
            skin = st.number_input('ความหนาผิวหนัง (SkinThickness)', 0, 100, 20)
            
        with col2:
            ins = st.number_input('ระดับอินซูลิน (Insulin)', 0, 900, 79)
            bmi = st.number_input('ดัชนีมวลกาย (BMI)', 0.0, 70.0, 25.0)
            dpf = st.number_input('ประวัติครอบครัว (DiabetesPedigreeFunction)', 0.0, 2.5, 0.5)
            age = st.number_input('อายุ (Age)', 1, 100, 30)
            
        submit = st.form_submit_button(f"🚀 ประเมินความเสี่ยงด้วย {model_choice}", use_container_width=True)
        
    if submit:
        input_data = pd.DataFrame([[p, g, bp, skin, ins, bmi, dpf, age]], columns=X.columns)
        prediction = active_model.predict(scaler.transform(input_data))
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"🚨 **ผลการทำนายด้วย {model_choice}:** พบความเสี่ยงเป็นโรคเบาหวาน (ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเชิงลึก)")
        else:
            st.success(f"✅ **ผลการทำนายด้วย {model_choice}:** ไม่พบความเสี่ยง (สุขภาพปกติ)")