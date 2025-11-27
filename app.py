# app.py — spaCy lemmatization + rapidfuzz scoring (single best match)
import os
import json
import numpy as np
import io 
from functools import lru_cache
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from deep_translator import GoogleTranslator
import difflib
from rapidfuzz import fuzz, process

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

ALLOWED_EXT = {"png", "jpg", "jpeg"}

MODEL_PATH = "cnn/my_model.keras"
CLASS_NAMES_PATH = "cnn/class_names.json"
SYM_DB_PATH = "cnn/symptoms_db.json"

model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    
    print("Warning: model load failed:", e)
    model = None

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

with open(SYM_DB_PATH, "r") as f:
    symptoms_db = json.load(f)

# --------- TRANSLATION CACHE ----------
@lru_cache(maxsize=2000)
def translate_cached(text: str, target_lang: str):
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print("Translate error:", e)
        return text

# --------- spaCy lemmatizer (preferred) ----------
nlp_spacy = None
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("spaCy loaded: using lemmatization for NLP")
except Exception as e:
    nlp_spacy = None
    print("spaCy not available or model not installed. Falling back to simple tokenization (no lemmatization).")
    print("To enable lemmatization install: pip install spacy && python -m spacy download en_core_web_sm")

# --------- text normalization using lemma if available ----------
def normalize_tokens_using_spacy(text: str):
    """
    Return list of lemmas (lowercase) excluding stopwords / punctuation.
    Requires spaCy model loaded. If spaCy not present, returns fallback tokens.
    """
    t = text.lower().strip()
    # remove simple punctuation
    for ch in ['.', ',', '!', '?', ':', ';', '(', ')', '"', "'"]:
        t = t.replace(ch, ' ')
    t = " ".join(t.split())
    if nlp_spacy:
        doc = nlp_spacy(t)
        tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
        if tokens:
            return tokens

    return [w for w in t.split() if w.isalpha()]

# --------- build phrase index (lemmatized phrases) ----------
def build_phrase_index(db: dict):
    index = {}
    for cls, info in db.items():
        phrases = info.get("symptoms", []) or []
        norm_phrases = []
        for p in phrases:
            toks = normalize_tokens_using_spacy(p)
            if toks:
                norm_phrases.append(" ".join(toks))
            else:
                norm_phrases.append(p.lower())
        index[cls] = norm_phrases
    return index

SYM_INDEX = build_phrase_index(symptoms_db)

# --------- crop helper ----------
def detect_crop(text: str):
    t = text.lower()
    t = t.replace("potatoes", "potato").replace("tomatoes", "tomato").replace("peppers", "pepper")
    crops = [
        "tomato", "potato", "pepper", "chili", "chilli",
        "bell pepper", "pepper,_bell", "pepper__bell"
    ]
    for c in crops:
        if c in t:
            return c
    return None

# --------- matching using rapidfuzz ----------
def match_class_from_text_lemmatized(user_text: str, db: dict, index: dict = SYM_INDEX, threshold: int = 60):
    """
    Uses token_set_ratio (rapidfuzz) between user (lemmatized) and each class's lemmatized symptom phrase.
    Returns (best_class_or_None, best_score_0_100, top_matches_list)
    - threshold is 0..100, recommended 55-70 depending on strictness.
    """
    if not user_text or not user_text.strip():
        return (None, 0.0, [])

    user_toks = normalize_tokens_using_spacy(user_text)
    if not user_toks:
        user_norm = user_text.lower()
    else:
        user_norm = " ".join(user_toks)

    class_scores = {}
    for cls in db.keys():
        best_for_cls = 0
        phrases = index.get(cls, [])
        for ph in phrases:
            # token_set_ratio is order-insensitive and robust to extra/missing tokens
            score = fuzz.token_set_ratio(user_norm, ph)
            if score > best_for_cls:
                best_for_cls = score
        # also compare against class name itself (normalized)
        cls_norm = cls.replace("___", " ").replace("_", " ").lower()
        name_score = fuzz.token_set_ratio(user_norm, cls_norm)
        if name_score > best_for_cls:
            best_for_cls = name_score
        class_scores[cls] = best_for_cls

    # optional keyword boosts (small additive)
    keyword_boosts = {
        "powder": ["powdery"], "white": ["powdery"], "mildew": ["powdery"],
        "web": ["spider_mites"], "mosaic": ["mosaic", "tobacco_mosaic_virus", "tomato_mosaic_virus"],
        "yellow": ["yellow_leaf", "yellow_leaf_curl"]
    }
    low_text = user_text.lower()
    for kw, tokens in keyword_boosts.items():
        if kw in low_text:
            # fuzzy find matching classes and add small boost
            candidates = process.extract(kw, list(db.keys()), scorer=fuzz.partial_ratio, limit=6)
            for cls_match, score_val, _ in candidates:
                class_scores[cls_match] = min(100, class_scores.get(cls_match, 0) + 8)

    # rank classes
    ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    top_matches = [(cls, round(score, 2)) for cls, score in ranked[:6]]

    if not ranked:
        return (None, 0.0, [])

    best_cls, best_score = ranked[0]
    if best_score >= threshold:
        return (best_cls, round(best_score, 2), top_matches)
    else:
        return (None, round(best_score, 2), top_matches)

# --------- Name normalization & fallback lookup ----------
def normalize_name_variants(name: str):
    variants = set()
    variants.add(name)
    variants.add(name.replace("___", "_"))
    variants.add(name.replace("___", " "))
    variants.add(name.replace("_", " "))
    variants.add(name.replace(" ", "_"))
    variants.add("_".join([p for p in name.split("_") if p != ""]))
    return list(variants)

def get_info_for_class(predicted_name: str, db: dict, fuzzy_cutoff=0.5):
    if predicted_name in db:
        return predicted_name, db[predicted_name], False
    for v in normalize_name_variants(predicted_name):
        if v in db:
            return v, db[v], True
    keys = list(db.keys())
    matches = difflib.get_close_matches(predicted_name, keys, n=1, cutoff=fuzzy_cutoff)
    if matches:
        m = matches[0]
        return m, db[m], True
    return None, None, False

# --------- Image predict (UPDATED for in-memory handling) ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict_image_class(file_stream):
    """Predicts class from an image file stream (in memory)."""
    if model is None:
        raise RuntimeError("Model not loaded. Place your Keras model at 'cnn/my_model.keras' or disable image branch.")
    
    # Use image.load_img with an io.BytesIO object
    img = image.load_img(file_stream, target_size=(128, 128))
    
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(np.max(preds))  # class_name, confidence(0-1)

# --------- Flask app ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def chat_ui():
    return render_template("chat.html")

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    # This route remains for potentially serving previously saved images or other static files
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts:
      - form field 'text' (user text)
      - file 'file' (optional image)
      - form field 'lang' (target language, ISO code like 'en','hi','mr','kn','te')
    Returns single best match for text branch (if above threshold).
    """
    file = request.files.get("file")
    text = request.form.get("text", "").strip()
    lang = request.form.get("lang", "en").strip() or "en"

    # ---------- IMAGE PATH (UPDATED to use in-memory stream) ----------
    if file and file.filename != "" and allowed_file(file.filename):
        # Read the file content into an in-memory buffer
        try:
            # Move pointer to the start of the file for reading
            file.seek(0)
            file_stream = io.BytesIO(file.read())
            
            # Predict using the in-memory stream
            cls, conf = predict_image_class(file_stream)
        
        except Exception as e:
            return jsonify({"type":"error", "message": f"Image model error: {e}"}), 500

        matched_name, info, is_fallback = get_info_for_class(cls, symptoms_db, fuzzy_cutoff=0.5)

        cause_en = info.get("cause","") if info else ""
        treatment_en = info.get("treatment","") if info else ""

        if lang != "en":
            cause_out = translate_cached(cause_en, lang)
            treatment_out = translate_cached(treatment_en, lang)
        else:
            cause_out = cause_en
            treatment_out = treatment_en

        note = f"(showing info for {matched_name})" if is_fallback and matched_name else ""
        return jsonify({
            "type": "image",
            "class": cls,
            "confidence": round(conf * 100, 2),
            "cause": cause_out or "No cause information available.",
            "treatment": treatment_out or "No treatment information available.",
            "note": note
        })

    # ---------- TEXT PATH (No changes needed here) ----------
    if text:
        # normalize some plurals
        tt = text.lower().replace("potatoes", "potato").replace("tomatoes", "tomato").replace("peppers", "pepper")

        # translate user text -> en for matching (if needed)
        text_for_matching = tt
        if lang != "en":
            try:
                text_for_matching = GoogleTranslator(source=lang, target='en').translate(tt)
            except Exception as e:
                print("Translate (user->en) failed:", e)
                text_for_matching = tt

        # optional crop filtering
        crop = detect_crop(text_for_matching)
        if crop:
            filtered_db = {k: v for k, v in symptoms_db.items() if k.lower().startswith(crop)}
            if not filtered_db:
                filtered_db = {k: v for k, v in symptoms_db.items() if crop in k.lower()}
        else:
            filtered_db = symptoms_db

        # quick keyword rule (keeps quick behavior)
        keywords = ["yellow", "yellowing", "wilting", "wilt", "brown", "spot", "spots",
                     "powdery", "mildew", "curl", "curling", "web", "webbing", "mosaic",
                     "mold", "rotting", "rot", "holes", "holes in leaves"]
        text_l = text_for_matching.lower()
        for kw in keywords:
            if kw in text_l:
                for cls_key, cls_info in filtered_db.items():
                    for phrase in cls_info.get("symptoms", []):
                        if kw in phrase.lower():
                            cause_en = cls_info.get("cause", "")
                            treatment_en = cls_info.get("treatment", "")
                            if lang != "en":
                                cause_out = translate_cached(cause_en, lang)
                                treatment_out = translate_cached(treatment_en, lang)
                            else:
                                cause_out = cause_en
                                treatment_out = treatment_en
                            return jsonify({
                                "type": "text",
                                "class": cls_key,
                                "score": 60.0,
                                "cause": cause_out,
                                "treatment": treatment_out,
                                "note": f"Quick match by keyword '{kw}'"
                            })

        # main matching (single best)
        MAIN_THRESHOLD = 60  # recommended tuning: 55-70
        best_cls, best_score, top_matches = match_class_from_text_lemmatized(text_for_matching, filtered_db, threshold=MAIN_THRESHOLD)

        # if not confident and filtered_db was smaller, try whole DB as fallback
        if (best_cls is None or best_score < MAIN_THRESHOLD) and (filtered_db is not symptoms_db):
            best_cls, best_score, top_matches = match_class_from_text_lemmatized(text_for_matching, symptoms_db, threshold=MAIN_THRESHOLD)

        if best_cls:
            info = symptoms_db.get(best_cls, {})
            cause_en = info.get("cause", "")
            treatment_en = info.get("treatment", "")
            if lang != "en":
                cause_out = translate_cached(cause_en, lang)
                treatment_out = translate_cached(treatment_en, lang)
            else:
                cause_out = cause_en
                treatment_out = treatment_en
            return jsonify({
                "type": "text",
                "class": best_cls,
                "score": best_score,
                "cause": cause_out,
                "treatment": treatment_out
            })
        else:
            msg = "No matching symptom found. Try adding the crop name (e.g. 'tomato' or 'potato') or mention extra words like 'yellow spots', 'white powder', 'webbing'."
            if lang != "en":
                msg = translate_cached(msg, lang)
            return jsonify({
                "type": "text",
                "class": None,
                "score": 0,
                "cause": "",
                "treatment": "",
                "message": msg
            })

    return jsonify({"type":"none", "message":"No input provided."})

if __name__ == "__main__":
    # For dev only; use a proper WSGI server in production
    app.run(host="0.0.0.0", port=5000, debug=True)