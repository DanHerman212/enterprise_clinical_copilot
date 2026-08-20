"""Synthetic demo cohort — discharge-note generator (Task 3).

Renders a coherent, fully-fictional discharge summary per patient from
`eval/results/synthetic_cohort.json`, in the app's note format + `___` redaction
style, consistent with each patient's features (age, sex, LOS, meds count,
procedures, oncology, discharge location, labs) and risk band.

Usage (repo root):
    .venv/bin/python projects/agent-harness/scripts/generate_synthetic_notes.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COHORT_PATH = REPO / "projects/agent-harness/eval/results/synthetic_cohort.json"
OUT_PATH = REPO / "projects/agent-harness/eval/results/synthetic_notes.json"


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


# --- per-archetype renderers: fn(patient, feats) -> full note string ---------

def _routine_short(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nAbdominal pain and vomiting\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with a history of mild asthma who presented "
        "with two days of crampy abdominal pain, nausea and one episode of non-bloody emesis. "
        "No fever, no diarrhea. Symptoms improved with IV fluids and antiemetics. "
        "Labs were unremarkable; hemoglobin stable, sodium within normal limits. "
        f"He/she was observed for {int(f['index_los_days'])} day(s) and discharged home in "
        "stable condition.\n"
        " \n"
        "Past Medical History:\n#Mild asthma\n"
        " \n"
        "Brief Hospital Course:\n"
        f"{int(f['index_los_days'])}-day admission for uncomplicated gastroenteritis/abdominal "
        "pain. Managed with IV fluids, antiemetics, and a clear liquid diet. "
        "No procedures required. Discharged home on supportive care.\n"
        " \n"
    )
    note += _meds_block([("Ondansetron 4 mg", "PO", "Q8H PRN", "ondansetron 4 mg 1 tablet by mouth every 8 hours as needed for nausea")])
    note += "\n \nDischarge Disposition:\nHome\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Gastroenteritis\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Dehydration\n# Mild asthma\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory - Independent.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for a stomach illness. Please drink plenty of fluids and "
                    "call us if the pain, vomiting, or fevers return.")
    return note


def _observation(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nSyncope\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old who presented after a brief episode of "
        "syncope at home. No seizure activity, no chest pain. Initial workup including "
        "electrocardiogram, telemetry and basic labs was unremarkable. "
        f"He/she was observed for {int(f['index_los_days'])} day(s) and discharged home in "
        "stable condition.\n"
        " \n"
        "Past Medical History:\nNone\n"
        " \n"
        "Brief Hospital Course:\n"
        f"Observation admission for syncope. Telemetry showed no arrhythmia; orthostatics "
        "improved with hydration. Discharged home.\n \n"
    )
    note += _meds_block([])
    note += "\n \nDischarge Disposition:\nHome\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Syncope, vasovagal\n\nSECONDARY DIAGNOSIS:\n====================\n# None\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory - Independent.\n"
    )
    note += _closing("Mr./Ms.", "You were evaluated after a fainting spell. Please rise slowly and stay "
                    "well hydrated, and follow up with your primary care physician.")
    return note


def _minor_elective(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nRight upper quadrant pain\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nLaparoscopic cholecystectomy\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with symptomatic cholelithiasis who "
        "underwent elective laparoscopic cholecystectomy. Postoperative course was "
        f"uncomplicated; he/she stayed {int(f['index_los_days'])} day(s) and was discharged "
        "home tolerating a regular diet.\n"
        " \n"
        "Past Medical History:\n#Cholelithiasis\n"
        " \n"
        "Brief Hospital Course:\n"
        "Elective laparoscopic cholecystectomy. Pathology confirmed chronic cholecystitis. "
        "Pain controlled with oral analgesics. Discharged home.\n \n"
    )
    note += _meds_block([
        ("Acetaminophen 650 mg", "PO", "Q6H PRN", "acetaminophen 650 mg 1 tablet by mouth every 6 hours as needed for pain"),
        ("Oxycodone 5 mg", "PO", "Q6H PRN", "oxycodone 5 mg 1 tablet by mouth every 6 hours as needed for breakthrough pain"),
    ])
    note += "\n \nDischarge Disposition:\nHome\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Symptomatic cholelithiasis, s/p laparoscopic cholecystectomy\n\n"
        "SECONDARY DIAGNOSIS:\n====================\n# None\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory - Independent.\n"
    )
    note += _closing("Mr./Ms.", "You had your gallbladder removed. Keep the incisions clean and dry; "
                    "call us for fever, worsening pain, or redness.")
    return note


def _diabetic_foot(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nLeft foot ulcer with swelling\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nIncision and drainage of left foot\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with type 2 diabetes who presented with a "
        "worsening left foot ulcer and surrounding cellulitis. He/she was started on IV "
        f"antibiotics, underwent incision and drainage, and improved over a {int(f['index_los_days'])}-day "
        "admission. Wound cultures grew mixed skin flora. Discharged on oral antibiotics "
        "with home health wound care.\n"
        " \n"
        "Past Medical History:\n#Type 2 diabetes mellitus\n#Hypertension\n#Peripheral neuropathy\n"
        " \n"
        "Brief Hospital Course:\n"
        f"Cellulitis of the left foot with a deep ulcer. IV vancomycin and piperacillin-"
        "tazobactam, incision and drainage, tight glucose control. "
        "Wound improved; discharged with home health services.\n \n"
    )
    note += _meds_block([
        ("Amoxicillin-Clavulanate 875 mg", "PO", "BID", "amoxicillin-clavulanate 875 mg 1 tablet by mouth twice a day"),
        ("Metformin 1000 mg", "PO", "BID", "metformin 1000 mg 1 tablet by mouth twice a day"),
        ("Lisinopril 10 mg", "PO", "DAILY", "lisinopril 10 mg 1 tablet by mouth daily"),
    ])
    note += "\n \nDischarge Disposition:\nHome with services\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Diabetic foot infection with cellulitis\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Type 2 diabetes mellitus\n# Hypertension\n# Peripheral neuropathy\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory with assistance.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for a foot infection. Keep the wound clean and "
                    "offloaded, finish the antibiotics, and check your blood sugars.")
    return note


def _ckd_pneumonia(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nFever and productive cough\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with chronic kidney disease and "
        "hypertension who presented with fever, productive cough and hypoxia. Chest imaging "
        "showed a right lower lobe consolidation. He/she was treated with IV antibiotics and "
        f"respiratory support, and improved over a {int(f['index_los_days'])}-day admission. "
        "Renal function was monitored closely.\n"
        " \n"
        "Past Medical History:\n#Chronic kidney disease stage 3\n#Hypertension\n#Type 2 diabetes\n"
        " \n"
        "Brief Hospital Course:\n"
        f"Community-acquired pneumonia. IV ceftriaxone and azithromycin, supplemental oxygen. "
        "Sodium and renal function remained stable; discharged home with home health follow-up.\n \n"
    )
    note += _meds_block([
        ("Amoxicillin 875 mg", "PO", "BID", "amoxicillin 875 mg 1 tablet by mouth twice a day"),
        ("Amlodipine 5 mg", "PO", "DAILY", "amlodipine 5 mg 1 tablet by mouth daily"),
        ("Atorvastatin 20 mg", "PO", "QPM", "atorvastatin 20 mg 1 tablet by mouth at bedtime"),
    ])
    note += "\n \nDischarge Disposition:\nHome with services\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Pneumonia, right lower lobe\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Chronic kidney disease\n# Hypertension\n# Type 2 diabetes mellitus\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory - Independent.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for pneumonia. Finish your antibiotics, keep your "
                    "follow-up appointment, and seek care if the fevers or shortness of breath return.")
    return note


def _postop_infection(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nSurgical wound redness and drainage\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nIleocolic resection (index) / wound debridement\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old who returned {int(f['index_los_days'])} days "
        "after an ileocolic resection with wound erythema and purulent drainage. A superficial "
        "surgical site infection was diagnosed; wound opened and packed, IV antibiotics started, "
        "with gradual improvement.\n"
        " \n"
        "Past Medical History:\n#Crohn's disease\n#Hypertension\n"
        " \n"
        "Brief Hospital Course:\n"
        f"Superficial surgical site infection. IV vancomycin and piperacillin-tazobactam, wound "
        "opened and packed, negative wound cultures. "
        "Discharged on oral antibiotics with visiting nurse wound care.\n \n"
    )
    note += _meds_block([
        ("Cephalexin 500 mg", "PO", "QID", "cephalexin 500 mg 1 tablet by mouth four times a day"),
        ("Mesalamine 1.2 g", "PO", "BID", "mesalamine 1.2 g 1 tablet by mouth twice a day"),
    ])
    note += "\n \nDischarge Disposition:\nHome with services\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Surgical site infection (superficial)\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Crohn's disease\n# Hypertension\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory - Independent.\n"
    )
    note += _closing("Mr./Ms.", "You had a wound infection after surgery. Keep the wound clean and "
                    "packed as instructed; call us for fever, spreading redness, or worsening drainage.")
    return note


def _elderly_chf(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nShortness of breath and leg swelling\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with heart failure with reduced ejection "
        "fraction, who presented with progressive dyspnea, orthopnea and bilateral lower "
        "extremity edema. He/she was treated with IV diuresis, and improved over a "
        f"{int(f['index_los_days'])}-day admission. Weight decreased, oxygen requirement resolved. "
        "Labs showed mildly elevated RDW and stable renal function on diuretics.\n"
        " \n"
        "Past Medical History:\n#Heart failure with reduced EF\n#Coronary artery disease\n"
        "#Hypertension\n#Atrial fibrillation\n#Chronic kidney disease\n"
        " \n"
        "Brief Hospital Course:\n"
        f"{int(f['index_los_days'])}-day admission for acute on chronic heart failure. IV "
        "furosemide, daily weights, strict I/O, optimization of GDMT. "
        "Discharged to a skilled nursing facility on oral diuretics and heart failure medications.\n \n"
    )
    note += _meds_block([
        ("Furosemide 80 mg", "PO", "DAILY", "furosemide 80 mg 1 tablet by mouth daily"),
        ("Carvedilol 12.5 mg", "PO", "BID", "carvedilol 12.5 mg 1 tablet by mouth twice a day"),
        ("Lisinopril 10 mg", "PO", "DAILY", "lisinopril 10 mg 1 tablet by mouth daily"),
        ("Warfarin 3 mg", "PO", "DAILY", "warfarin 3 mg 1 tablet by mouth daily"),
        ("Atorvastatin 40 mg", "PO", "QPM", "atorvastatin 40 mg 1 tablet by mouth at bedtime"),
    ])
    note += "\n \nDischarge Disposition:\nSkilled Nursing Facility\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Acute on chronic systolic heart failure\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Coronary artery disease\n# Hypertension\n# Atrial fibrillation\n# Chronic kidney disease\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory with assistance.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for a heart failure flare-up. Weigh yourself daily, "
                    "limit fluids and salt, and take your heart medicines as prescribed.")
    return note


def _oncology_infection(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nFever and fatigue\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone (port placed)\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with metastatic non-small cell lung cancer "
        "on active chemotherapy, who presented with fevers and neutropenia. He/she was admitted "
        f"for broad-spectrum antibiotics and supportive care over a {int(f['index_los_days'])}-day "
        "course. Blood counts recovered; cultures remained negative. "
        "Discharged with close oncology follow-up.\n"
        " \n"
        "Past Medical History:\n#Metastatic non-small cell lung cancer\n#COPD\n#Hypertension\n"
        "#History of PE on anticoagulation\n"
        " \n"
        "Brief Hospital Course:\n"
        f"Febrile neutropenia. IV cefepime, G-CSF support, transfusion for anemia. "
        "Neutrophil count recovered; discharged in improved condition.\n \n"
    )
    note += _meds_block([
        ("Levofloxacin 750 mg", "PO", "DAILY", "levofloxacin 750 mg 1 tablet by mouth daily"),
        ("Apixaban 5 mg", "PO", "BID", "apixaban 5 mg 1 tablet by mouth twice a day"),
        ("Pantoprazole 40 mg", "PO", "DAILY", "pantoprazole 40 mg 1 tablet by mouth daily"),
        ("Ondansetron 8 mg", "PO", "Q8H PRN", "ondansetron 8 mg 1 tablet by mouth every 8 hours as needed for nausea"),
    ])
    note += "\n \nDischarge Disposition:\nHospice\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# Febrile neutropenia\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Metastatic non-small cell lung cancer\n# COPD\n# Hypertension\n# History of PE\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory with assistance.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for a low white blood cell count with fever. "
                    "Contact your oncology team immediately for any new fevers.")
    return note


def _copd_readmission(p, f):
    sex = "M" if f["gender"] == 0 else "F"
    note = _header(sex)
    note += (
        "Chief Complaint:\nWorsening shortness of breath\n"
        " \n"
        "Major Surgical or Invasive Procedure:\nNone\n"
        " \n"
        "History of Present Illness:\n"
        f"Mr./Ms. ___ is a {int(f['age'])} year old with severe COPD and multiple prior "
        f"admissions, who presented with increased dyspnea and wheezing. He/she required "
        "bronchodilators, steroids and supplemental oxygen, and improved over a "
        f"{int(f['index_los_days'])}-day admission. Discharged on an optimized inhaler regimen "
        "to a rehabilitation facility.\n"
        " \n"
        "Past Medical History:\n#Severe COPD\n#Coronary artery disease\n#Hypertension\n#OSA\n"
        " \n"
        "Brief Hospital Course:\n"
        f"COPD exacerbation. Nebulized bronchodilators, systemic steroids, low-flow oxygen. "
        "Respiratory status improved; discharged on dual bronchodilator therapy.\n \n"
    )
    note += _meds_block([
        ("Prednisone 40 mg", "PO", "DAILY", "prednisone 40 mg 1 tablet by mouth daily with a taper"),
        ("Tiotropium 18 mcg", "INH", "DAILY", "tiotropium 18 mcg 1 capsule inhaled daily"),
        ("Albuterol 90 mcg", "INH", "Q4H PRN", "albuterol 90 mcg 2 puffs inhaled every 4 hours as needed"),
        ("Azithromycin 250 mg", "PO", "DAILY", "azithromycin 250 mg 1 tablet by mouth daily"),
    ])
    note += "\n \nDischarge Disposition:\nRehab\n"
    note += (
        "\nDischarge Diagnosis:\nPRIMARY DIAGNOSIS:\n==================\n"
        "# COPD exacerbation\n\nSECONDARY DIAGNOSIS:\n====================\n"
        "# Severe COPD\n# Coronary artery disease\n# Hypertension\n# Obstructive sleep apnea\n"
        "\nDischarge Condition:\nMental Status: Clear and coherent.\n"
        "Level of Consciousness: Alert and interactive.\n"
        "Activity Status: Ambulatory with assistance.\n"
    )
    note += _closing("Mr./Ms.", "You were treated for a COPD flare-up. Use your inhalers as directed, "
                    "finish the steroid taper, and call us if your breathing worsens.")
    return note


TEMPLATES = {
    "routine_short": _routine_short,
    "observation": _observation,
    "minor_elective": _minor_elective,
    "diabetic_foot": _diabetic_foot,
    "ckd_pneumonia": _ckd_pneumonia,
    "postop_infection": _postop_infection,
    "elderly_chf": _elderly_chf,
    "oncology_infection": _oncology_infection,
    "copd_readmission": _copd_readmission,
}


def main() -> int:
    cohort = json.loads(COHORT_PATH.read_text())["patients"]
    out = []
    for p in cohort:
        fn = TEMPLATES.get(p["archetype"])
        if fn is None:
            print(f"!! no template for archetype {p['archetype']} (hadm {p['hadm_id']})")
            continue
        note = fn(p, p["features"])
        out.append({"hadm_id": p["hadm_id"], "archetype": p["archetype"],
                    "band": p["band"], "note": note})
    OUT_PATH.write_text(json.dumps({"n": len(out), "patients": out}, indent=2))
    print(f"Wrote {OUT_PATH} ({len(out)} notes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
