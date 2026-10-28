from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
import numpy as np
import requests
import os

app = FastAPI(title="Sepsis Clinical Prediction API")

MODEL_PATH = "final_clinical_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في تحميل ملف النموذج: {e}")
else:
    print(f"⚠️ ملف النموذج غير موجود في المسار: {MODEL_PATH}")

BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "YOUR_BLYNK_AUTH_TOKEN")

class SingleReading(BaseModel):
    heart_rate: float
    o2_sat: float
    temperature: float

class PatientVitalsBatch(BaseModel):
    readings: list[SingleReading]

def extract_features(readings: list[SingleReading]):
    """
    اقتصاص أحدث 5 قراءات متتابعة واستخراج الخصائص الـ 8 بالترتيب الدقيق:
    1. mean_hr
    2. mean_spo2
    3. mean_temp
    4. std_hr
    5. std_spo2
    6. latest_hr
    7. latest_spo2
    8. latest_temp
    """
    # أخذ أحدث 5 قراءات فقط لضمان مطابقة النافذة الزمنية للتدريب
    last_5_readings = readings[-5:]

    hrs = [r.heart_rate for r in last_5_readings]
    o2s = [r.o2_sat for r in last_5_readings]
    tmps = [r.temperature for r in last_5_readings]

    mean_hr = float(np.mean(hrs))
    mean_spo2 = float(np.mean(o2s))
    mean_temp = float(np.mean(tmps))

    std_hr = float(np.std(hrs)) if len(hrs) > 1 else 0.0
    std_spo2 = float(np.std(o2s)) if len(o2s) > 1 else 0.0

    latest_hr = hrs[-1]
    latest_spo2 = o2s[-1]
    latest_temp = tmps[-1]

    features = np.array([[
        mean_hr,
        mean_spo2,
        mean_temp,
        std_hr,
        std_spo2,
        latest_hr,
        latest_spo2,
        latest_temp
    ]])
    return features

def trigger_blynk_alert(message: str) -> bool:
    if not BLYNK_AUTH_TOKEN or BLYNK_AUTH_TOKEN == "YOUR_BLYNK_AUTH_TOKEN":
        print("⚠️ توكن Blynk غير مُعرّف.")
        return False

    url = f"https://blynk.cloud/external/api/notify?token={BLYNK_AUTH_TOKEN}&message={message}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            print("✅ تم إرسال التنبيه إلى Blynk بنجاح.")
            return True
        else:
            print(f"⚠️ استجابة Blynk كود: {res.status_code}")
            return False
    except Exception as e:
        print(f"❌ خطأ في الاتصال برابط Blynk: {e}")
        return False

@app.get("/")
def home():
    return {
        "status": "Online",
        "model_loaded": model is not None,
        "message": "Sepsis Early Detection API is Ready"
    }

@app.post("/predict")
def predict_sepsis(payload: PatientVitalsBatch):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="خطأ حرج: نموذج الذكاء الاصطناعي غير محمل على السيرفر."
        )

    readings = payload.readings

    if not readings or len(readings) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="يتطلب النظام 5 قراءات متتابعة على الأقل للتحليل."
        )

    latest = readings[-1]
    if latest.heart_rate > 120 or latest.o2_sat < 90 or latest.temperature > 38.5:
        trigger_blynk_alert("إنذار عاجل: المؤشرات الحيوية تجاوزت الحدود المألوفة!")
        return {
            "risk_status": "CRITICAL_RULES_EXCEEDED",
            "message": "مؤشر عاجل: العلامات الحيوية خارج النطاق الطبيعي المألوف",
            "sepsis_predicted": True,
            "clinical_note": "مؤشر تنبئي استرشادي للطاقم الطبي."
        }

    try:
        features = extract_features(readings)
        prediction = int(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء معالجة الخصائص والتنبؤ: {str(e)}"
        )

    if prediction == 1:
        trigger_blynk_alert("تنبيه مبكر: نمط الذكاء الاصطناعي يشير لاحتمالية تدهور الحالة (خطر إنتان)")
        return {
            "risk_status": "AI_WARNING_SEPSIS_RISK",
            "message": "تنبيه مبكر: تم رصد نمط إحصائي يشير لاحتمالية تدهور الحالة",
            "sepsis_predicted": True,
            "clinical_note": "مؤشر تنبئي استرشادي يتطلب تقييم الطبيب المختص."
        }

    return {
        "risk_status": "NORMAL",
        "message": "المؤشرات الحيوية ضمن النطاق الطبيعي المستقر",
        "sepsis_predicted": False,
        "clinical_note": "مؤشر تنبئي استرشادي."
    }