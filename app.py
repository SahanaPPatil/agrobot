import os
import json
import numpy as np
import io 
from functools import lru_cache
from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from deep_translator import GoogleTranslator
import difflib
from rapidfuzz import fuzz, process

# --------- CONFIGURATION ----------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

ALLOWED_EXT = {"png", "jpg", "jpeg"}
MODEL_PATH = "cnn/my_model.keras"
CLASS_NAMES_PATH = "cnn/class_names.json"
SYM_DB_PATH = "cnn/symptoms_db.json"
USERS_DB_PATH = "users.json"

app = Flask(__name__)
app.secret_key = "agro_bot_secure_key_123" # Required for sessions

# --------- LOAD DATA ----------
model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("Warning: model load failed:", e)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def load_symptoms():
    with open(SYM_DB_PATH, "r") as f:
        return json.load(f)

symptoms_db = load_symptoms()

def load_users():
    if not os.path.exists(USERS_DB_PATH):
        # Default users if file missing
        default_data = {"admin": {"password": "admin123", "role": "admin"}, "farmer": {"password": "user123", "role": "farmer"}}
        with open(USERS_DB_PATH, "w") as f:
            json.dump(default_data, f)
        return default_data
    with open(USERS_DB_PATH, "r") as f:
        return json.load(f)

# --------- TRANSLATION CACHE ----------
@lru_cache(maxsize=2000)
def translate_cached(text: str, target_lang: str):
    if not text: return ""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text

# --------- spaCy & NLP Logic ----------
nlp_spacy = None
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    print("spaCy fallback active.")

def normalize_tokens_using_spacy(text: str):
    t = text.lower().strip()
    for ch in ['.', ',', '!', '?', ':', ';', '(', ')', '"', "'"]:
        t = t.replace(ch, ' ')
    t = " ".join(t.split())
    if nlp_spacy:
        doc = nlp_spacy(t)
        return [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
    return [w for w in t.split() if w.isalpha()]

def build_phrase_index(db: dict):
    index = {}
    for cls, info in db.items():
        phrases = info.get("symptoms", []) or []
        norm_phrases = [" ".join(normalize_tokens_using_spacy(p)) for p in phrases]
        index[cls] = norm_phrases
    return index

SYM_INDEX = build_phrase_index(symptoms_db)

# --------- CORE MATCHING LOGIC ----------
def detect_crop(text: str):
    t = text.lower()
    crops = ["tomato", "potato", "pepper"]
    for c in crops:
        if c in t: return c
    return None

def match_class_from_text_lemmatized(user_text: str, db: dict, index: dict, threshold: int = 60):
    user_toks = normalize_tokens_using_spacy(user_text)
    user_norm = " ".join(user_toks) if user_toks else user_text.lower()
    class_scores = {}
    for cls in db.keys():
        phrases = index.get(cls, [])
        best_for_cls = max([fuzz.token_set_ratio(user_norm, ph) for ph in phrases] + [0])
        cls_norm = cls.replace("___", " ").replace("_", " ").lower()
        class_scores[cls] = max(best_for_cls, fuzz.token_set_ratio(user_norm, cls_norm))
    
    ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] < threshold: return None, 0, []
    return ranked[0][0], round(ranked[0][1], 2), ranked[:6]

def get_info_for_class(predicted_name: str, db: dict):
    if predicted_name in db: return predicted_name, db[predicted_name], False
    matches = difflib.get_close_matches(predicted_name, list(db.keys()), n=1, cutoff=0.5)
    return (matches[0], db[matches[0]], True) if matches else (None, None, False)

def predict_image_class(file_stream):
    img = image.load_img(file_stream, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(arr, axis=0))[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(np.max(preds))

# --------- AUTHENTICATION ROUTES ----------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()
        
        if username in users and users[username]["password"] == password:
            session["user"] = username
            session["role"] = users[username]["role"]
            
            # 🔥 FIX: Direct redirect based on role
            if session["role"] == "admin":
                return redirect(url_for("admin_panel"))
            else:
                return redirect(url_for("chat_ui"))
        
        return "Invalid Credentials. Please try again."
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------- ADMIN ROUTES (CRUD) ----------

@app.route("/admin")
def admin_panel():
    if session.get("role") != "admin": 
        return redirect(url_for("login"))
    return render_template("admin.html", database=symptoms_db)

@app.route("/admin/add", methods=["POST"])
def add_entry():
    if session.get("role") != "admin": return "Unauthorized", 403
    disease = request.form.get("disease_name")
    symptoms_db[disease] = {
        "cause": request.form.get("cause"),
        "treatment": request.form.get("treatment"),
        "symptoms": [s.strip() for s in request.form.get("symptoms").split(",")]
    }
    with open(SYM_DB_PATH, "w") as f: json.dump(symptoms_db, f, indent=4)
    global SYM_INDEX
    SYM_INDEX = build_phrase_index(symptoms_db)
    return redirect(url_for("admin_panel"))

@app.route("/admin/delete/<name>")
def delete_entry(name):
    if session.get("role") != "admin": return "Unauthorized", 403
    if name in symptoms_db:
        del symptoms_db[name]
        with open(SYM_DB_PATH, "w") as f: json.dump(symptoms_db, f, indent=4)
        global SYM_INDEX
        SYM_INDEX = build_phrase_index(symptoms_db)
    return redirect(url_for("admin_panel"))

# --------- MAIN CHAT ROUTES ----------

@app.route("/")
def chat_ui():
    # 🔥 FIX: Force login for the home page
    if "user" not in session: 
        return redirect(url_for("login"))
    return render_template("chat.html", role=session.get("role"))

@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session: return jsonify({"error": "Unauthorized"}), 401
    file = request.files.get("file")
    text = request.form.get("text", "").strip()
    lang = request.form.get("lang", "en").strip() or "en"

    if file and file.filename != "":
        file.seek(0)
        file_stream = io.BytesIO(file.read())
        cls, conf = predict_image_class(file_stream)
        matched_name, info, is_fallback = get_info_for_class(cls, symptoms_db)
        if info:
            return jsonify({
                "type": "image", "class": cls, "confidence": round(conf * 100, 2),
                "cause": translate_cached(info['cause'], lang),
                "treatment": translate_cached(info['treatment'], lang),
                "note": f"Matched: {matched_name}" if is_fallback else ""
            })

    if text:
        en_text = GoogleTranslator(source='auto', target='en').translate(text) if lang != "en" else text
        crop = detect_crop(en_text)
        filtered_db = {k: v for k, v in symptoms_db.items() if k.lower().startswith(crop or "")} or symptoms_db
        best_cls, score, _ = match_class_from_text_lemmatized(en_text, filtered_db, SYM_INDEX)
        
        if best_cls:
            info = symptoms_db[best_cls]
            return jsonify({
                "type": "text", "class": best_cls, "score": score,
                "cause": translate_cached(info['cause'], lang),
                "treatment": translate_cached(info['treatment'], lang)
            })
        return jsonify({"type": "text", "message": translate_cached("No match found.", lang)})

    return jsonify({"type": "none", "message": "No input."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)