"""PredictSkin backend. Run: uvicorn main:app --reload  ->  http://127.0.0.1:8000"""
import hashlib, io, re, secrets, sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from inference import DEVICE, MODEL_CONFIG, MODELS, _transform, predict_one

app = FastAPI()
db = sqlite3.connect("predictskin.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, salt TEXT, pw TEXT)")
db.execute("CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER)")
db.execute("CREATE TABLE IF NOT EXISTS kv(user_id INTEGER, k TEXT, v TEXT, PRIMARY KEY(user_id,k))")

# ---------- auth (only name, email, hashed password are stored; images are never saved)
def _hash(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()

class Auth(BaseModel):
    name: str = ""
    email: str
    password: str

@app.post("/api/signup")
def signup(a: Auth):
    salt = secrets.token_hex(8)
    try:
        cur = db.execute("INSERT INTO users(name,email,salt,pw) VALUES(?,?,?,?)", (a.name, a.email.lower(), salt, _hash(a.password, salt)))
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    return _session(cur.lastrowid, a.name)

@app.post("/api/login")
def login(a: Auth):
    r = db.execute("SELECT id,name,salt,pw FROM users WHERE email=?", (a.email.lower(),)).fetchone()
    if not r or _hash(a.password, r[2]) != r[3]:
        raise HTTPException(401, "Wrong email or password")
    return _session(r[0], r[1])

def _session(uid, name):
    t = secrets.token_urlsafe(24)
    db.execute("INSERT INTO sessions VALUES(?,?)", (t, uid)); db.commit()
    return {"token": t, "name": name}

def _user(authorization):
    t = (authorization or "").replace("Bearer ", "")
    r = db.execute("SELECT user_id FROM sessions WHERE token=?", (t,)).fetchone()
    if not r:
        raise HTTPException(401, "Please log in")
    return r[0]

# ---------- disease info (keyword match on the predicted label; edit freely)
INFO = [
 ("contact dermatitis|poison ivy", "A red, itchy rash after skin touches an irritant or allergen.", "Not an infection: it is a reaction to things like plants, metals, fragrances or soaps. Scratching can let bacteria in.", "No food causes it; avoiding the trigger substance matters most."),
 ("seborrheic dermatitis", "Flaky, greasy, red patches, often on the scalp, face or chest.", "Linked to Malassezia, a yeast on the skin (a fungus, not bacteria), and the skin's inflammatory response.", "No firm food rules; some people notice flares with alcohol or very sugary foods."),
 ("seborrheic keratos|benign", "Harmless waxy or stuck-on growths that become common with age.", "Not an infection; the cause is unclear and it often runs in families.", "No food link. See a doctor if a spot changes quickly."),
 ("lupus|connective|scleroderma|sarcoid|systemic", "Immune-related conditions that can affect the skin and other organs.", "Not caused by bacteria; the immune system is involved.", "No specific food causes them. Sun protection and a balanced diet are usually advised, and a doctor should assess it."),
 ("photodermat|light disease|pigment|solar", "Skin reactions to sunlight, or changes in skin colour.", "Triggered by UV light, sometimes together with medicines or products; not bacterial.", "No food to avoid in general; sun protection matters. Ask a pharmacist if any medicine increases sun sensitivity."),
 ("folliculitis|hidradenitis", "Inflamed hair follicles causing small bumps or painful lumps.", "Folliculitis is often caused by Staphylococcus aureus (bacteria). Hidradenitis is an inflammatory condition, not simply an infection.", "No food is proven to cause it; some people report that sugary foods or dairy make hidradenitis worse, but evidence is limited."),
 ("bullous|erythema multiforme|neutrophilic|vasculitis|granuloma|pityriasis|rubra", "Inflammatory skin conditions that can cause blisters, target-shaped spots, bumps or rings.", "Usually immune-related or triggered by infections or medicines; not simply bacterial.", "No specific food to avoid. Blistering or fast-spreading rashes should be seen by a doctor soon."),
 ("drug|exanthem", "A rash caused by a medicine or a viral illness.", "A reaction to a medicine or a virus; not usually bacterial.", "No food link. Do not stop a prescribed medicine on your own; speak to a doctor quickly, especially with fever, blisters or breathing trouble."),
 ("nematode|larva|leishmania", "Skin infections from parasites, usually picked up from soil, sand or sandflies.", "Caused by parasites (such as hookworm larvae or Leishmania), not bacteria.", "No food link. A doctor must treat it; covering the skin helps prevent it."),
 ("hair|alopecia", "Hair loss has many possible causes.", "Not usually bacterial; genetics, stress, thyroid problems, nutrition and immune conditions can all play a part.", "No single food causes it; crash diets and low protein or iron intake may contribute."),
 ("urticaria|hives", "Itchy raised welts that come and go.", "An immune reaction, not bacterial; triggers can include foods, medicines, heat or sunlight.", "Only foods you personally know trigger it."),
 ("vascular|ichthyosis", "Vascular tumours are benign blood-vessel growths; ichthyosis is dry, scaly skin that is often inherited.", "Not infections; ichthyosis is usually genetic.", "No food link; regular moisturising helps dry, scaly skin."),
 ("acne|rosacea", "Acne is a common condition where pores clog and become inflamed.", "Cutibacterium acnes (bacteria) can inflame clogged pores.", "Sugary drinks, sweets and very refined foods may worsen acne for some people."),
 ("psoriasis|lichen", "Psoriasis often shows as raised, scaly plaques. It is long-term and can flare up.", "Not caused by bacteria (immune-related); a strep throat infection can trigger flares in some people.", "Alcohol, deep-fried and highly processed foods may worsen flares for some people."),
 ("eczema|atopic|dermatitis", "Eczema causes dry, itchy, inflamed skin and often comes and goes.", "Not caused by bacteria, but Staphylococcus aureus can colonise the skin and worsen it.", "Any food you are personally allergic to; for some people eggs, milk or nuts may trigger flares."),
 ("bacterial|cellulitis|impetigo", "Bacterial skin infections cause red, warm, sometimes crusted or painful skin.", "Staphylococcus aureus or Streptococcus pyogenes are common causes.", "No food causes it; very sugary foods may slow healing."),
 ("fungal|tinea|ringworm|candid|nail", "Fungal infections cause itchy, ring-shaped or scaly patches, or thick nails.", "Caused by fungi (such as Trichophyton or Candida), not bacteria.", "Excess sugar and refined carbs may encourage Candida overgrowth."),
 ("scabies|infestation", "Scabies causes intense itching, worse at night, from tiny mites in the skin.", "Caused by the Sarcoptes scabiei mite, not bacteria (scratching can add a bacterial infection).", "No food restrictions; treatment and washing bedding and clothes matter most."),
 ("melanoma|malignant|carcinoma|keratosis", "This group includes lesions that can be skin cancer and need early checking.", "Not bacterial; sun (UV) exposure is a major risk factor.", "No food causes it; limiting alcohol is generally advised. Please see a dermatologist soon."),
 ("wart|viral|herpes|molluscum|zoster", "Viral skin conditions often show as bumps, blisters or warty growths.", "Caused by viruses (such as HPV or herpes viruses), not bacteria.", "No specific foods to avoid; rest and good nutrition support recovery."),
]
DEFAULT = ("A skin pattern that looks similar to this condition was found.", "The cause depends on the condition and can be bacterial, fungal, viral or immune-related.", "No specific foods are proven to cause it; a balanced diet helps skin health.")

def info_for(label):
    for key, *rest in INFO:
        if re.search(key, label.lower()):
            return rest
    return list(DEFAULT)

EMB = "dermnet" if "dermnet" in MODELS else next(iter(MODELS))

# ---------- backup: manually added reference images (cosine similarity of embeddings)
def embed(img):
    c = MODEL_CONFIG[EMB]
    x = _transform(c["img_size"], c.get("mean", [0.485, 0.456, 0.406]), c.get("std", [0.229, 0.224, 0.225]))(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return F.normalize(MODELS[EMB].features(x).mean((2, 3))[0], dim=0)

REF = []
for p in Path("reference_dataset").glob("*/*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        try: REF.append((p.parent.name, embed(Image.open(p).convert("RGB"))))
        except Exception as e: print("skip", p, e)

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), authorization: str = Header(None)):
    _user(authorization)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    # main plan: both trained models, keep the single most confident top-1
    tops = [predict_one(n, img, 1)[0] for n in MODELS]
    best = max(tops, key=lambda t: t["confidence"])
    label, conf = best["label"], best["confidence"]
    if conf < 45 and REF:  # backup: only when the models are unsure
        v = embed(img)
        sim, ref = max((float(v @ e), n) for n, e in REF)
        if sim >= 0.8: label = ref
    about, cause, avoid = info_for(label)
    return {"disease": re.sub(r" (Photos|pictures)\b.*", "", label), "about": about, "cause": cause, "avoid": avoid}

class KV(BaseModel):
    value: str

@app.get("/api/kv/{key}")
def kv_get(key: str, authorization: str = Header(None)):
    uid = _user(authorization)
    r = db.execute("SELECT v FROM kv WHERE user_id=? AND k=?", (uid, key)).fetchone()
    return {"value": r[0] if r else None}

@app.put("/api/kv/{key}")
def kv_put(key: str, body: KV, authorization: str = Header(None)):
    uid = _user(authorization)
    if not re.fullmatch(r"[a-z_]{1,30}", key):
        raise HTTPException(400, "Bad key")
    db.execute("INSERT OR REPLACE INTO kv VALUES(?,?,?)", (uid, key, body.value[:200000])); db.commit()
    return {"ok": True}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home(): return FileResponse("static/index.html")
