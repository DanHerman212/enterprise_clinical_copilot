"""Synthetic demo cohort — discharge-note generator (Task 3).

Renders a coherent, fully-fictional discharge summary per patient from
`evaluation/agent/results/synthetic_cohort.json`, in the app's note format + `___` redaction
style, consistent with each patient's features (age, sex, LOS, meds count,
procedures, oncology, discharge location, labs) and risk band.

Usage (repo root):
    .venv/bin/python scripts/agent/generate_synthetic_notes.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COHORT_PATH = REPO / "evaluation/agent/results/synthetic_cohort.json"
OUT_PATH = REPO / "evaluation/agent/results/synthetic_notes.json"


def _header(sex: str) -> str:
    return (
        "\nName:  ___                      Unit No:   ___\n"
        " \n"
        "Admission Date:  ___              Discharge Date:   ___\n"
        " \n"
        "Date of Birth:  ___             Sex:   " + sex + "\n"
        " \n"
        "Service: MEDICINE\n"
        " \n"
        "Allergies: \n___\n"
        " \n"
        "Attending: ___.\n"
        " \n"
    )


def _meds_block(meds: list[tuple[str, str, str, str]]) -> str:
    """meds: [(drug+dose, route, freq, rx_line)] -> numbered discharge meds block."""
    lines = ["Discharge Medications:"]
    for i, (drug, route, freq, rx) in enumerate(meds, 1):
        lines.append(f"{i}.  {drug} {route} {freq}")
        if rx:
            lines.append(f"RX *{rx} \nDisp #*{len(meds) * 10} Tablet Refills:*0")
    return "\n".join(lines)


def _closing(mr_ms: str, body: str) -> str:
    return (
        "Discharge Instructions:\n"
        f"Dear {mr_ms} ___,\n\n"
        f"{body}\n\n"
        "We wish you the best,\n"
        "Your ___ team\n"
        " \n"
        "Followup Instructions:\n___\n"
    )


# --- shared renderer + per-archetype clinical variants -----------------------


def _note(sex: str, v: dict) -> str:
    """Assemble a full discharge summary from a variant dict."""
    mr_ms = "Mr." if sex == "M" else "Ms."
    n = _header(sex)
    n += f"Chief Complaint:\n{v['complaint']}\n \n"
    n += f"Major Surgical or Invasive Procedure:\n{v['procedure']}\n \n"
    n += f"History of Present Illness:\n{v['hpi']}\n \n"
    n += "Past Medical History:\n" + "".join(f"#{x}\n" for x in v["pmh"]) + " \n"
    n += f"Brief Hospital Course:\n{v['course']}\n \n"
    n += _meds_block(v["meds"])
    n += f"\n \nDischarge Disposition:\n{v['disposition']}\n"
    n += ("\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
          f"# {v['dx_primary']}\n\nSECONDARY DIAGNOSIS:\n====================\n" +
          "".join(f"# {x}\n" for x in v["dx_secondary"]))
    n += ("\nDischarge Condition:\nMental Status: Clear and coherent.\n"
          "Level of Consciousness: Alert and interactive.\n"
          f"Activity Status: {v['condition']}.\n")
    n += _closing(mr_ms, v["instructions"])
    return n


TEMPLATES = {
    "routine_short": [
        {"complaint": "Abdominal pain and vomiting", "procedure": "None",
         "hpi": "Two days of crampy abdominal pain, nausea and one episode of non-bloody emesis. No fever, no diarrhea. Improved with IV fluids and antiemetics; labs unremarkable.",
         "pmh": ["Mild asthma"],
         "course": "Uncomplicated gastroenteritis managed with IV fluids, antiemetics, and a clear liquid diet.",
         "meds": [("Ondansetron 4 mg", "PO", "Q8H PRN", "ondansetron 4 mg 1 tablet by mouth every 8 hours as needed for nausea")],
         "disposition": "Home", "dx_primary": "Gastroenteritis",
         "dx_secondary": ["Dehydration", "Mild asthma"], "condition": "Ambulatory - Independent",
         "instructions": "You were treated for a stomach illness. Drink plenty of fluids and call us if the pain, vomiting, or fevers return."},
        {"complaint": "Chest pain", "procedure": "None",
         "hpi": "One episode of substernal chest pressure at rest, non-exertional, without radiation. Serial troponins negative, EKG unchanged; pain resolved.",
         "pmh": ["Gastroesophageal reflux"],
         "course": "Low-risk chest pain ruled out. Telemetry, serial troponins and EKG all reassuring.",
         "meds": [("Omeprazole 20 mg", "PO", "DAILY", "omeprazole 20 mg 1 capsule by mouth daily")],
         "disposition": "Home", "dx_primary": "Chest pain, non-cardiac (rule-out MI)",
         "dx_secondary": ["Gastroesophageal reflux"], "condition": "Ambulatory - Independent",
         "instructions": "Your chest pain was not from your heart. Follow up with your doctor and seek care if it returns or worsens."},
        {"complaint": "Painful urination and fever", "procedure": "None",
         "hpi": "Dysuria, urinary frequency and low-grade fever. Urinalysis consistent with infection; treated with fluids and antibiotics.",
         "pmh": ["Recurrent urinary tract infections"],
         "course": "Uncomplicated urinary tract infection; responded well to antibiotics.",
         "meds": [("Nitrofurantoin 100 mg", "PO", "BID", "nitrofurantoin 100 mg 1 capsule by mouth twice a day")],
         "disposition": "Home", "dx_primary": "Urinary tract infection",
         "dx_secondary": ["Recurrent UTI"], "condition": "Ambulatory - Independent",
         "instructions": "You were treated for a urinary infection. Finish the antibiotics and drink plenty of water."},
    ],
    "observation": [
        {"complaint": "Syncope", "procedure": "None",
         "hpi": "Brief episode of syncope at home, no seizure activity, no chest pain. Telemetry and labs unremarkable; orthostatics improved with hydration.",
         "pmh": ["None"],
         "course": "Observation admission; no arrhythmia on telemetry, workup negative.",
         "meds": [],
         "disposition": "Home", "dx_primary": "Syncope, vasovagal", "dx_secondary": ["None"],
         "condition": "Ambulatory - Independent",
         "instructions": "You were evaluated after a fainting spell. Rise slowly and stay hydrated."},
        {"complaint": "Chest discomfort", "procedure": "None",
         "hpi": "Atypical chest discomfort; serial troponins negative, EKG normal. No cardiac risk factors.",
         "pmh": ["None"],
         "course": "Observation with telemetry; ruled out for acute coronary syndrome.",
         "meds": [],
         "disposition": "Home", "dx_primary": "Chest pain, atypical (rule-out ACS)",
         "dx_secondary": ["None"], "condition": "Ambulatory - Independent",
         "instructions": "Your chest discomfort was not cardiac. Follow up with your primary doctor."},
        {"complaint": "Headache", "procedure": "None",
         "hpi": "Severe unilateral headache with photophobia, improving with analgesics and quiet. Neurologic exam normal; head imaging negative.",
         "pmh": ["Migraine"],
         "course": "Observation; migraine treated, no acute intracranial process.",
         "meds": [("Sumatriptan 50 mg", "PO", "PRN", "sumatriptan 50 mg 1 tablet by mouth as needed for headache")],
         "disposition": "Home", "dx_primary": "Migraine headache", "dx_secondary": ["Migraine"],
         "condition": "Ambulatory - Independent",
         "instructions": "You were treated for a migraine. Rest in a quiet, dark room and take your headache medicine as needed."},
    ],
    "minor_elective": [
        {"complaint": "Right upper quadrant pain", "procedure": "Laparoscopic cholecystectomy",
         "hpi": "Symptomatic cholelithiasis; underwent elective laparoscopic cholecystectomy, uncomplicated postoperative course.",
         "pmh": ["Cholelithiasis"],
         "course": "Elective laparoscopic cholecystectomy; pathology confirmed chronic cholecystitis.",
         "meds": [("Acetaminophen 650 mg", "PO", "Q6H PRN", "acetaminophen 650 mg 1 tablet by mouth every 6 hours as needed for pain")],
         "disposition": "Home", "dx_primary": "Symptomatic cholelithiasis, s/p laparoscopic cholecystectomy",
         "dx_secondary": ["None"], "condition": "Ambulatory - Independent",
         "instructions": "You had your gallbladder removed. Keep the incisions clean and dry; call us for fever or worsening pain."},
        {"complaint": "Left groin bulge", "procedure": "Inguinal hernia repair",
         "hpi": "Painful left inguinal hernia; underwent elective open inguinal hernia repair, uncomplicated.",
         "pmh": ["Inguinal hernia"],
         "course": "Elective inguinal hernia repair; recovery uncomplicated.",
         "meds": [("Acetaminophen 650 mg", "PO", "Q6H PRN", "acetaminophen 650 mg 1 tablet by mouth every 6 hours as needed for pain")],
         "disposition": "Home", "dx_primary": "Inguinal hernia, s/p repair", "dx_secondary": ["None"],
         "condition": "Ambulatory - Independent",
         "instructions": "You had a hernia repair. Avoid heavy lifting for six weeks and keep the incision clean."},
    ],
    "diabetic_foot": [
        {"complaint": "Left foot ulcer with swelling", "procedure": "Incision and drainage of left foot",
         "hpi": "Worsening left foot ulcer with surrounding cellulitis; IV antibiotics and incision and drainage, improved.",
         "pmh": ["Type 2 diabetes mellitus", "Hypertension", "Peripheral neuropathy"],
         "course": "Diabetic foot cellulitis; I&D performed, wound improved with IV antibiotics and tight glucose control.",
         "meds": [("Amoxicillin-Clavulanate 875 mg", "PO", "BID", "amoxicillin-clavulanate 875 mg 1 tablet by mouth twice a day"),
                 ("Metformin 1000 mg", "PO", "BID", "metformin 1000 mg 1 tablet by mouth twice a day")],
         "disposition": "Home with services", "dx_primary": "Diabetic foot infection with cellulitis",
         "dx_secondary": ["Type 2 diabetes mellitus", "Hypertension", "Peripheral neuropathy"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a foot infection. Keep the wound clean, finish the antibiotics, and check your blood sugars."},
        {"complaint": "Right foot wound with exposed bone", "procedure": "Incision and drainage / bone debridement",
         "hpi": "Right foot ulcer probing to bone; osteomyelitis suspected, IV antibiotics and surgical debridement.",
         "pmh": ["Type 2 diabetes mellitus", "Peripheral arterial disease"],
         "course": "Osteomyelitis of the right foot; debridement, IV antibiotics, wound care.",
         "meds": [("Ciprofloxacin 750 mg", "PO", "BID", "ciprofloxacin 750 mg 1 tablet by mouth twice a day"),
                 ("Metformin 1000 mg", "PO", "BID", "metformin 1000 mg 1 tablet by mouth twice a day")],
         "disposition": "Home with services", "dx_primary": "Right foot osteomyelitis",
         "dx_secondary": ["Type 2 diabetes mellitus", "Peripheral arterial disease"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a bone infection in your foot. Keep weight off the foot and complete your antibiotics."},
        {"complaint": "Left great toe discoloration", "procedure": "Toe amputation",
         "hpi": "Gangrenous left great toe with surrounding infection; underwent toe amputation.",
         "pmh": ["Type 2 diabetes mellitus", "Peripheral neuropathy"],
         "course": "Ischemic/neuropathic toe gangrene; amputation, wound care, glucose optimization.",
         "meds": [("Amoxicillin-Clavulanate 875 mg", "PO", "BID", "amoxicillin-clavulanate 875 mg 1 tablet by mouth twice a day"),
                 ("Aspirin 81 mg", "PO", "DAILY", "aspirin 81 mg 1 tablet by mouth daily")],
         "disposition": "Home with services", "dx_primary": "Gangrene of left great toe, s/p amputation",
         "dx_secondary": ["Type 2 diabetes mellitus", "Peripheral neuropathy"],
         "condition": "Ambulatory with assistance",
         "instructions": "You had a toe removed due to infection. Keep it clean and dry and follow up with podiatry."},
    ],
    "ckd_pneumonia": [
        {"complaint": "Fever and productive cough", "procedure": "None",
         "hpi": "Fever, productive cough and hypoxia; imaging showed right lower lobe consolidation, treated with IV antibiotics.",
         "pmh": ["Chronic kidney disease stage 3", "Hypertension", "Type 2 diabetes"],
         "course": "Community-acquired pneumonia; IV antibiotics, supplemental oxygen, renal function stable.",
         "meds": [("Amoxicillin 875 mg", "PO", "BID", "amoxicillin 875 mg 1 tablet by mouth twice a day"),
                 ("Amlodipine 5 mg", "PO", "DAILY", "amlodipine 5 mg 1 tablet by mouth daily")],
         "disposition": "Home with services", "dx_primary": "Pneumonia, right lower lobe",
         "dx_secondary": ["Chronic kidney disease", "Hypertension", "Type 2 diabetes mellitus"],
         "condition": "Ambulatory - Independent",
         "instructions": "You were treated for pneumonia. Finish your antibiotics and keep your follow-up appointment."},
        {"complaint": "Confusion and fever", "procedure": "None",
         "hpi": "Sepsis from pneumonia with acute kidney injury; IV fluids and antibiotics, renal function gradually recovered.",
         "pmh": ["Chronic kidney disease stage 3", "Hypertension"],
         "course": "Sepsis from pneumonia; resuscitated, antibiotics, AKI managed with careful fluid balance.",
         "meds": [("Levofloxacin 750 mg", "PO", "DAILY", "levofloxacin 750 mg 1 tablet by mouth daily"),
                 ("Lisinopril 10 mg", "PO", "DAILY", "lisinopril 10 mg 1 tablet by mouth daily")],
         "disposition": "Home with services", "dx_primary": "Sepsis secondary to pneumonia with acute kidney injury",
         "dx_secondary": ["Chronic kidney disease", "Hypertension"], "condition": "Ambulatory with assistance",
         "instructions": "You had a serious infection affecting your kidneys. Take your medicines and follow up for repeat labs."},
        {"complaint": "Shortness of breath and pedal edema", "procedure": "None",
         "hpi": "Pneumonia with fluid overload and hyperkalemia in the setting of CKD; treated with antibiotics and careful diuresis.",
         "pmh": ["Chronic kidney disease", "Heart failure"],
         "course": "Pneumonia complicated by volume overload and hyperkalemia; antibiotics, diuresis, potassium management.",
         "meds": [("Cefpodoxime 200 mg", "PO", "BID", "cefpodoxime 200 mg 1 tablet by mouth twice a day"),
                 ("Furosemide 40 mg", "PO", "DAILY", "furosemide 40 mg 1 tablet by mouth daily")],
         "disposition": "Rehab", "dx_primary": "Pneumonia with volume overload and hyperkalemia",
         "dx_secondary": ["Chronic kidney disease", "Heart failure"], "condition": "Ambulatory with assistance",
         "instructions": "You were treated for pneumonia and fluid buildup. Limit salt and fluids and follow up with your kidney doctor."},
    ],
    "postop_infection": [
        {"complaint": "Surgical wound redness and drainage", "procedure": "Ileocolic resection (index) / wound debridement",
         "hpi": "Returned days after ileocolic resection with wound erythema and purulent drainage; superficial surgical site infection.",
         "pmh": ["Crohn's disease", "Hypertension"],
         "course": "Superficial SSI; wound opened and packed, IV antibiotics, gradual improvement.",
         "meds": [("Cephalexin 500 mg", "PO", "QID", "cephalexin 500 mg 1 tablet by mouth four times a day"),
                 ("Mesalamine 1.2 g", "PO", "BID", "mesalamine 1.2 g 1 tablet by mouth twice a day")],
         "disposition": "Home with services", "dx_primary": "Surgical site infection (superficial)",
         "dx_secondary": ["Crohn's disease", "Hypertension"], "condition": "Ambulatory - Independent",
         "instructions": "You had a wound infection after surgery. Keep the wound clean and packed as instructed."},
        {"complaint": "Wound drainage after colectomy", "procedure": "Colectomy (index) / wound exploration",
         "hpi": "Purulent drainage from a colectomy incision; wound explored, superficial infection drained.",
         "pmh": ["Diverticulitis", "Hypertension"],
         "course": "Post-colectomy wound infection; opened, packed, antibiotics.",
         "meds": [("Ciprofloxacin 500 mg", "PO", "BID", "ciprofloxacin 500 mg 1 tablet by mouth twice a day"),
                 ("Metronidazole 500 mg", "PO", "TID", "metronidazole 500 mg 1 tablet by mouth three times a day")],
         "disposition": "Home with services", "dx_primary": "Postoperative wound infection after colectomy",
         "dx_secondary": ["Diverticulitis", "Hypertension"], "condition": "Ambulatory - Independent",
         "instructions": "You had a wound infection after your colon surgery. Finish the antibiotics and keep the wound clean."},
        {"complaint": "Incisional swelling and fluid leak", "procedure": "Hysterectomy (index) / seroma drainage",
         "hpi": "Incisional seroma with superficial wound separation; drained, no deep infection.",
         "pmh": ["Uterine fibroids"],
         "course": "Post-hysterectomy seroma; aspirated, wound care, observation.",
         "meds": [("Cephalexin 500 mg", "PO", "QID", "cephalexin 500 mg 1 tablet by mouth four times a day")],
         "disposition": "Home", "dx_primary": "Incisional seroma after hysterectomy", "dx_secondary": ["Uterine fibroids"],
         "condition": "Ambulatory - Independent",
         "instructions": "You had a fluid pocket at your incision. Keep it dry and call us if it becomes red or painful."},
    ],
    "elderly_chf": [
        {"complaint": "Shortness of breath and leg swelling", "procedure": "None",
         "hpi": "Progressive dyspnea, orthopnea and bilateral edema in the setting of heart failure; IV diuresis, weight loss, improved oxygen requirement.",
         "pmh": ["Heart failure with reduced EF", "Coronary artery disease", "Hypertension", "Atrial fibrillation", "Chronic kidney disease"],
         "course": "Acute on chronic heart failure; IV furosemide, daily weights, GDMT optimization.",
         "meds": [("Furosemide 80 mg", "PO", "DAILY", "furosemide 80 mg 1 tablet by mouth daily"),
                 ("Carvedilol 12.5 mg", "PO", "BID", "carvedilol 12.5 mg 1 tablet by mouth twice a day"),
                 ("Warfarin 3 mg", "PO", "DAILY", "warfarin 3 mg 1 tablet by mouth daily")],
         "disposition": "Skilled Nursing Facility", "dx_primary": "Acute on chronic systolic heart failure",
         "dx_secondary": ["Coronary artery disease", "Hypertension", "Atrial fibrillation", "Chronic kidney disease"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a heart failure flare-up. Weigh yourself daily, limit fluids and salt, and take your medicines."},
        {"complaint": "Sudden shortness of breath at rest", "procedure": "None (non-invasive ventilation)",
         "hpi": "Flash pulmonary edema; required non-invasive ventilation and aggressive IV diuresis, improved over several days.",
         "pmh": ["Heart failure with preserved EF", "Hypertension", "Obstructive sleep apnea"],
         "course": "Flash pulmonary edema; BiPAP, IV nitroglycerin and furosemide, transitioned to oral regimen.",
         "meds": [("Furosemide 60 mg", "PO", "DAILY", "furosemide 60 mg 1 tablet by mouth daily"),
                 ("Enalapril 5 mg", "PO", "BID", "enalapril 5 mg 1 tablet by mouth twice a day"),
                 ("Metoprolol 25 mg", "PO", "BID", "metoprolol 25 mg 1 tablet by mouth twice a day")],
         "disposition": "Rehab", "dx_primary": "Flash pulmonary edema (acute heart failure)",
         "dx_secondary": ["Heart failure with preserved EF", "Hypertension", "Obstructive sleep apnea"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a severe heart failure episode. Continue your heart medicines and keep your follow-up."},
        {"complaint": "Worsening swelling and fatigue", "procedure": "None",
         "hpi": "Heart failure exacerbation with acute kidney injury on diuretics; diuretic dose adjusted, renal function monitored.",
         "pmh": ["Heart failure", "Chronic kidney disease", "Diabetes"],
         "course": "Heart failure with AKI; careful diuresis, renal function recovered.",
         "meds": [("Torsemide 20 mg", "PO", "DAILY", "torsemide 20 mg 1 tablet by mouth daily"),
                 ("Spironolactone 25 mg", "PO", "DAILY", "spironolactone 25 mg 1 tablet by mouth daily")],
         "disposition": "Skilled Nursing Facility", "dx_primary": "Heart failure exacerbation with acute kidney injury",
         "dx_secondary": ["Heart failure", "Chronic kidney disease", "Diabetes"], "condition": "Ambulatory with assistance",
         "instructions": "You were treated for fluid buildup with some strain on your kidneys. Follow up for repeat blood work."},
    ],
    "oncology_infection": [
        {"complaint": "Fever and fatigue", "procedure": "None (port placed)",
         "hpi": "Febrile neutropenia after chemotherapy; broad-spectrum antibiotics, G-CSF, blood counts recovered.",
         "pmh": ["Metastatic non-small cell lung cancer", "COPD", "Hypertension", "History of PE"],
         "course": "Febrile neutropenia; IV cefepime, G-CSF, transfusion support, counts recovered.",
         "meds": [("Levofloxacin 750 mg", "PO", "DAILY", "levofloxacin 750 mg 1 tablet by mouth daily"),
                 ("Apixaban 5 mg", "PO", "BID", "apixaban 5 mg 1 tablet by mouth twice a day")],
         "disposition": "Hospice", "dx_primary": "Febrile neutropenia",
         "dx_secondary": ["Metastatic non-small cell lung cancer", "COPD", "Hypertension", "History of PE"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a low white blood cell count with fever. Contact your oncology team for any new fevers."},
        {"complaint": "Fevers and shaking chills", "procedure": "None",
         "hpi": "Bacteremia in the setting of metastatic colon cancer; blood cultures positive, IV antibiotics, source controlled.",
         "pmh": ["Metastatic colon cancer", "Hypertension"],
         "course": "Sepsis from line-associated bacteremia; IV antibiotics, line managed, improved.",
         "meds": [("Ceftriaxone 1 g", "IV", "DAILY", "ceftriaxone 1 g intravenously daily"),
                 ("Ondansetron 8 mg", "PO", "Q8H PRN", "ondansetron 8 mg 1 tablet by mouth every 8 hours as needed for nausea")],
         "disposition": "Home with services", "dx_primary": "Sepsis secondary to bacteremia",
         "dx_secondary": ["Metastatic colon cancer", "Hypertension"], "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a bloodstream infection. Finish the IV antibiotics and follow up with oncology."},
        {"complaint": "Mouth pain and fever", "procedure": "None",
         "hpi": "Neutropenic fever with mucositis after induction chemotherapy; supportive care, antimicrobials, counts recovered.",
         "pmh": ["Acute leukemia", "Diabetes"],
         "course": "Neutropenic fever with mucositis; broad antimicrobials, mouth care, transfusion support.",
         "meds": [("Ciprofloxacin 500 mg", "PO", "BID", "ciprofloxacin 500 mg 1 tablet by mouth twice a day"),
                 ("Nystatin swish", "PO", "QID", "nystatin oral suspension swish and swallow four times a day")],
         "disposition": "Rehab", "dx_primary": "Neutropenic fever with mucositis",
         "dx_secondary": ["Acute leukemia", "Diabetes"], "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a low blood count with mouth sores and fever. Call us immediately for any new fever."},
    ],
    "copd_readmission": [
        {"complaint": "Worsening shortness of breath", "procedure": "None",
         "hpi": "Increased dyspnea and wheezing with severe COPD; bronchodilators, steroids and oxygen, improved.",
         "pmh": ["Severe COPD", "Coronary artery disease", "Hypertension", "OSA"],
         "course": "COPD exacerbation; nebulized bronchodilators, systemic steroids, low-flow oxygen.",
         "meds": [("Prednisone 40 mg", "PO", "DAILY", "prednisone 40 mg 1 tablet by mouth daily with a taper"),
                 ("Tiotropium 18 mcg", "INH", "DAILY", "tiotropium 18 mcg 1 capsule inhaled daily"),
                 ("Albuterol 90 mcg", "INH", "Q4H PRN", "albuterol 90 mcg 2 puffs inhaled every 4 hours as needed")],
         "disposition": "Rehab", "dx_primary": "COPD exacerbation",
         "dx_secondary": ["Severe COPD", "Coronary artery disease", "Hypertension", "Obstructive sleep apnea"],
         "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a COPD flare-up. Use your inhalers as directed, finish the steroid taper, and call us if your breathing worsens."},
        {"complaint": "Cough with increased phlegm", "procedure": "None",
         "hpi": "COPD exacerbation with purulent sputum; treated with antibiotics, bronchodilators and steroids.",
         "pmh": ["Severe COPD", "Asthma-COPD overlap"],
         "course": "Infectious COPD exacerbation; antibiotics, nebulizers, steroids.",
         "meds": [("Azithromycin 250 mg", "PO", "DAILY", "azithromycin 250 mg 1 tablet by mouth daily"),
                 ("Fluticasone-Salmeterol", "INH", "BID", "fluticasone-salmeterol 1 inhalation twice a day")],
         "disposition": "Home with services", "dx_primary": "COPD exacerbation with infection",
         "dx_secondary": ["Severe COPD", "Asthma-COPD overlap"], "condition": "Ambulatory with assistance",
         "instructions": "You were treated for a lung infection with a COPD flare. Finish your medicines and use your inhalers."},
    ],
}


def main() -> int:
    cohort = json.loads(COHORT_PATH.read_text())["patients"]
    # Round-robin variant assignment per archetype for patient-to-patient variety.
    variant_counter: dict[str, int] = {}
    out = []
    for p in cohort:
        archetype = p["archetype"]
        variants = TEMPLATES.get(archetype)
        if not variants:
            print(f"!! no template for archetype {archetype} (hadm {p['hadm_id']})")
            continue
        i = variant_counter.get(archetype, 0)
        variant_counter[archetype] = i + 1
        v = variants[i % len(variants)]
        sex = "M" if p["features"]["gender"] == 0 else "F"
        note = _note(sex, v)
        out.append({"hadm_id": p["hadm_id"], "archetype": archetype,
                    "band": p["band"], "variant": i % len(variants), "note": note})
    OUT_PATH.write_text(json.dumps({"n": len(out), "patients": out}, indent=2))
    print(f"Wrote {OUT_PATH} ({len(out)} notes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
