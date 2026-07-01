# -*- coding: utf-8 -*-
"""Health_data.ipynb

FDA Drug vs. Supplement Adverse Event Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_score,
                              recall_score, f1_score)

st.set_page_config(page_title="FDA Adverse Event Dashboard", layout="wide")
st.title("FDA Drug vs. Supplement Adverse Event Dashboard")
st.write("Comparing reported seriousness of adverse events between drugs and dietary supplements using Gaussian Naive Bayes.")

# ── API key (optional) ───────────────────────────────────────────────
# Reads a key from Streamlit secrets if you've set one. Works without it,
# just at the lower no-key rate limit (1,000 requests/day).
try:
    FDA_API_KEY = st.secrets["FDA_API_KEY"]
    KEY_PARAM = f"&api_key={FDA_API_KEY}"
except (KeyError, FileNotFoundError):
    KEY_PARAM = ""

# ── Safe fetch helper ────────────────────────────────────────────────
# Makes the request, checks the status, and confirms 'results' exists
# before returning. Returns None on any failure instead of crashing.
def fetch_fda_json(url):
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    data = response.json()
    if 'results' not in data:
        return None
    return data


# ── Structure check (logs only) ──────────────────────────────────────
_sample = fetch_fda_json(f"https://api.fda.gov/drug/event.json?limit=10{KEY_PARAM}")
if _sample:
    print(_sample.keys())
    first_record = _sample['results'][0]
    print(first_record.keys())
    for i, record in enumerate(_sample['results']):
        print(f"Record {i}: serious = {record.get('serious')}")


@st.cache_data(ttl=600)  # cache for 10 minutes
def load_drug_data(limit=500):
    url = f"https://api.fda.gov/drug/event.json?limit={limit}{KEY_PARAM}"
    data = fetch_fda_json(url)
    if data is None:
        return None

    records = []
    for record in data['results']:
        patient = record.get('patient', {})
        death_info = patient.get('patientdeath', None)
        records.append({
            'serious': 1 if int(record.get('serious', 2)) == 1 else 0,
            'patientonsetage': patient.get('patientonsetage', None),
            'patientsex': patient.get('patientsex', None),
            'patientdeath': 1 if death_info is not None else 0,
        })

    df = pd.DataFrame(records)
    df['patientonsetage'] = pd.to_numeric(df['patientonsetage'], errors='coerce')
    df['patientonsetage'] = df['patientonsetage'].fillna(df['patientonsetage'].median())
    df['patientsex'] = pd.to_numeric(df['patientsex'], errors='coerce')
    return df

df_fda = load_drug_data(limit=500)

# Stop cleanly if the pull failed (most often the daily rate limit)
if df_fda is None or df_fda.empty:
    st.error(
        "Could not load drug data from openFDA. This is usually the daily rate "
        "limit (no API key = 1,000 requests/day). Add a free API key as the "
        "FDA_API_KEY secret for 120,000/day, or try again later."
    )
    st.stop()

st.subheader("Drug Adverse Event Data (Raw Sample)")
st.write(f"Shape: {df_fda.shape[0]} rows, {df_fda.shape[1]} columns")
st.dataframe(df_fda.head(20))

st.write("Missing values per column:")
st.write(df_fda.isnull().sum())

st.write("Serious value counts:")
st.write(df_fda['serious'].value_counts())
st.subheader("Data Cleaning")

# Convert age to numeric, fill nulls with median
df_fda['patientonsetage'] = pd.to_numeric(df_fda['patientonsetage'], errors='coerce')
df_fda['patientonsetage'] = df_fda['patientonsetage'].fillna(df_fda['patientonsetage'].median())
df_fda['patientsex'] = pd.to_numeric(df_fda['patientsex'], errors='coerce')

st.write("Nulls remaining after cleaning:")
st.write(df_fda.isnull().sum())

st.subheader("Baseline Model: Drug Adverse Events")
st.markdown("""
**Why cross-validation matters:** a single train/test split can be misleading —
a single split swung between 84%, 36%, and 50% accuracy depending on the random slice.
Averaging across multiple folds (cv=3) gave a stable **67.97% ± 1.85%** — a far more
trustworthy estimate of real performance.
""")

df_fda_model = df_fda.dropna()
X = df_fda_model.drop(columns=['serious']).copy()
y = df_fda_model['serious'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=3)
acc = scores.mean()

col1, col2 = st.columns(2)
col1.metric("Rows used", df_fda.shape[0])
col2.metric("CV Accuracy", f"{acc:.2%}", f"± {scores.std():.2%}")


# Refresh pipeline with new data + 4 panel metrics chart
st.subheader("Live Pipeline: Repeated Sampling")
st.write("Re-pulls fresh data from the API across multiple cycles to test how stable the model's metrics are as data changes.")

def refresh_pipeline(n_cycles=3, wait_seconds=10):
    results_log = []

    for cycle in range(n_cycles):
        offset = random.randint(0, 1000)
        url = f"https://api.fda.gov/drug/event.json?limit=500&skip={offset}{KEY_PARAM}"
        data = fetch_fda_json(url)
        if data is None:
            continue  # skip this cycle if the call failed

        records = []
        for record in data['results']:
            patient = record.get('patient', {})
            death_info = patient.get('patientdeath', None)
            records.append({
                'serious':         1 if int(record.get('serious', 2)) == 1 else 0,
                'patientonsetage': patient.get('patientonsetage', None),
                'patientsex':      patient.get('patientsex', None),
                'patientdeath':    1 if death_info is not None else 0,
            })

        df = pd.DataFrame(records)
        df['patientonsetage'] = pd.to_numeric(df['patientonsetage'], errors='coerce')
        df['patientsex'] = pd.to_numeric(df['patientsex'], errors='coerce')
        df['patientonsetage'] = df['patientonsetage'].fillna(df['patientonsetage'].median())
        df = df.dropna()
        df = df.drop_duplicates()

        if df.empty or df['serious'].nunique() < 2:
            continue  # need both classes to train

        X = df.drop(columns=['serious']).copy()
        y = df['serious'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = cross_val_score(model, X, y, cv=3)
        acc  = scores.mean()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        results_log.append({
            'cycle': cycle + 1, 'rows': df.shape[0],
            'cv_acc': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4)
        })

        if cycle < n_cycles - 1:
            time.sleep(wait_seconds)

    return pd.DataFrame(results_log)

n_cycles = st.slider("Number of refresh cycles", 2, 5, 3)

if st.button("Run pipeline refresh"):
    with st.spinner("Pulling fresh data and retraining..."):
        log = refresh_pipeline(n_cycles=n_cycles, wait_seconds=2)

    if log.empty:
        st.warning("No cycles returned usable data — likely the rate limit. Try again later or add an API key.")
    else:
        st.write("Results log:")
        st.dataframe(log)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(log['cycle'], log['cv_acc'], marker='o', color='steelblue')
        axes[0,0].set_title('CV Accuracy per Cycle'); axes[0,0].set_ylim(0, 1)
        axes[0,1].plot(log['cycle'], log['precision'], marker='o', color='darkorange')
        axes[0,1].set_title('Precision per Cycle'); axes[0,1].set_ylim(0, 1)
        axes[1,0].plot(log['cycle'], log['recall'], marker='o', color='green')
        axes[1,0].set_title('Recall per Cycle'); axes[1,0].set_ylim(0, 1)
        axes[1,1].plot(log['cycle'], log['f1'], marker='o', color='purple')
        axes[1,1].set_title('F1 Score per Cycle'); axes[1,1].set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)

# Adding drug recall information + enhanced model
st.subheader("Enhancing the Model: Drug Recall History")
st.write("Checking whether a drug involved in an adverse event has ever been subject to an FDA recall.")

@st.cache_data(ttl=600)
def get_recalled_drugs(limit=100):
    recall_url = f"https://api.fda.gov/drug/enforcement.json?limit={limit}{KEY_PARAM}"
    recall_data = fetch_fda_json(recall_url)
    if recall_data is None:
        return set()  # degrade gracefully: no recall flags, app continues

    recalled_drugs = set()
    for record in recall_data['results']:
        product = record.get('product_description', '').lower()
        drug_name = product.split()[0] if product else None
        if drug_name:
            drug_name = drug_name.strip('.,;')
            if len(drug_name) > 2:
                recalled_drugs.add(drug_name)
    return recalled_drugs

recalled_drugs = get_recalled_drugs(limit=100)
st.write(f"Total recalled drug names identified: {len(recalled_drugs)}")
st.write(sorted(list(recalled_drugs))[:15])

@st.cache_data(ttl=600)
def build_enhanced_dataset(_recalled_drugs, limit=500):
    offset = random.randint(0, 1000)
    url = f"https://api.fda.gov/drug/event.json?limit={limit}&skip={offset}{KEY_PARAM}"
    data = fetch_fda_json(url)
    if data is None:
        return None

    records = []
    for record in data['results']:
        patient = record.get('patient', {})
        death_info = patient.get('patientdeath', None)

        drugs = patient.get('drug', [])
        drug_names = []
        for drug in drugs:
            name = drug.get('medicinalproduct', '').lower().strip('.,;')
            if name:
                drug_names.append(name.split()[0])

        was_recalled = 1 if any(d in _recalled_drugs for d in drug_names) else 0

        records.append({
            'serious':         1 if int(record.get('serious', 2)) == 1 else 0,
            'patientonsetage': patient.get('patientonsetage', None),
            'patientsex':      patient.get('patientsex', None),
            'was_recalled':    was_recalled
        })

    df = pd.DataFrame(records)
    df['patientonsetage'] = pd.to_numeric(df['patientonsetage'], errors='coerce')
    df['patientsex'] = pd.to_numeric(df['patientsex'], errors='coerce')
    df['patientonsetage'] = df['patientonsetage'].fillna(df['patientonsetage'].median())
    df = df.dropna()
    df = df.drop_duplicates()
    return df

df_enhanced = build_enhanced_dataset(recalled_drugs, limit=500)

if df_enhanced is None or df_enhanced.empty or df_enhanced['serious'].nunique() < 2:
    st.error("Could not build the enhanced dataset (rate limit or not enough class variety in this pull). Try again later or add an API key.")
    st.stop()

st.write(f"Enhanced dataset shape: {df_enhanced.shape}")
st.write("Recall flag distribution:")
st.write(df_enhanced['was_recalled'].value_counts())
st.dataframe(df_enhanced.head())

X_new = df_enhanced.drop(columns=['serious']).copy()
y_new = df_enhanced['serious'].copy()

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
model_new = GaussianNB()
model_new.fit(X_train, y_train)
y_pred_new = model_new.predict(X_test)
scores_new = cross_val_score(model_new, X_new, y_new, cv=3)

st.write("**Enhanced Model (with recall history) Results:**")
col1, col2 = st.columns(2)
col1.metric("Features used", len(X_new.columns))
col2.metric("CV Accuracy", f"{scores_new.mean():.2%}", f"± {scores_new.std():.2%}")

# Format table for precision, recall, f1
report_dict = classification_report(y_test, y_pred_new, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
st.dataframe(report_df)

# Structure check on the food endpoint (logs only)
_supp_sample = fetch_fda_json(f"https://api.fda.gov/food/event.json?limit=10{KEY_PARAM}")
if _supp_sample:
    first = _supp_sample['results'][0]
    print("Outcomes:", first.get('outcomes'))
    print("Reactions:", first.get('reactions'))
    print("Consumer:", first.get('consumer'))
    print("Products:", first.get('products'))

# Mapping outcomes to binary target
st.subheader("Comparing Drugs vs. Dietary Supplements")
st.write("Building a matching model for dietary supplement adverse events, using the same approach as the drug model above.")

serious_outcomes = {
    'Death', 'Life Threatening', 'Hospitalization', 'Disability',
    'Congenital Anomaly', 'Required Intervention',
    'Other Serious Outcome', 'Other Serious or Important Medical Event'
}

@st.cache_data(ttl=600)
def build_supplement_dataset(limit=500):
    offset = random.randint(0, 1000)
    supp_url = f"https://api.fda.gov/food/event.json?limit={limit}&skip={offset}&search=products.industry_code:54{KEY_PARAM}"
    supp_data = fetch_fda_json(supp_url)
    if supp_data is None:
        return None

    records_supp = []
    for record in supp_data['results']:
        consumer = record.get('consumer', {})
        outcomes = record.get('outcomes', [])
        products = record.get('products', [])

        is_serious = 1 if any(o in serious_outcomes for o in outcomes) else 0

        gender = consumer.get('gender', None)
        gender_code = 1 if gender == 'Male' else (2 if gender == 'Female' else None)

        industry_code = products[0].get('industry_code', None) if products else None
        is_supplement = 1 if industry_code == '54' else 0

        records_supp.append({
            'serious': is_serious,
            'gender': gender_code,
            'is_supplement': is_supplement
        })

    df = pd.DataFrame(records_supp)
    df['gender'] = df['gender'].fillna(df['gender'].median())
    df = df.dropna()
    return df

df_supp = build_supplement_dataset(limit=500)

if df_supp is None or df_supp.empty or df_supp['serious'].nunique() < 2:
    st.error("Could not build the supplement dataset (rate limit or not enough class variety in this pull). Try again later or add an API key.")
    st.stop()

st.write(f"Supplement dataset shape: {df_supp.shape}")
st.write("Serious event rate:")
st.write(df_supp['serious'].value_counts(normalize=True).round(3) * 100)
st.dataframe(df_supp.head())

X_supp = df_supp.drop(columns=['serious']).copy()
y_supp = df_supp['serious'].copy()

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_supp, y_supp, test_size=0.2, random_state=42, stratify=y_supp)

model_supp = GaussianNB()
model_supp.fit(X_train_s, y_train_s)
y_pred_s = model_supp.predict(X_test_s)
scores_supp = cross_val_score(model_supp, X_supp, y_supp, cv=3)

st.write("**Supplement Model Results:**")
col1, col2 = st.columns(2)
col1.metric("Rows used", df_supp.shape[0])
col2.metric("CV Accuracy", f"{scores_supp.mean():.2%}", f"± {scores_supp.std():.2%}")

supp_report = classification_report(y_test_s, y_pred_s, zero_division=0, output_dict=True)
supp_report_df = pd.DataFrame(supp_report).transpose().round(3)
st.dataframe(supp_report_df)

# Adding rebalancing to supplement model to see if it improves precision/recall
st.subheader("Does Rebalancing Change the Picture?")
st.write("Downsampling the majority class to test whether the supplement model's high accuracy holds up once both classes are equally represented.")

df_supp_majority = df_supp[df_supp.serious == 1]
df_supp_minority = df_supp[df_supp.serious == 0]

if len(df_supp_minority) > 0 and len(df_supp_majority) > 0:
    from sklearn.utils import resample

    df_supp_majority_down = resample(
        df_supp_majority, replace=False,
        n_samples=len(df_supp_minority), random_state=42
    )
    df_supp_balanced = pd.concat([df_supp_majority_down, df_supp_minority])

    st.write("Balanced class counts:")
    st.write(df_supp_balanced['serious'].value_counts())

    X_bal = df_supp_balanced.drop(columns=['serious']).copy()
    y_bal = df_supp_balanced['serious'].copy()

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    model_bal = GaussianNB()
    model_bal.fit(X_train_b, y_train_b)
    y_pred_b = model_bal.predict(X_test_b)
    scores_bal = cross_val_score(model_bal, X_bal, y_bal, cv=3)

    col1, col2 = st.columns(2)
    col1.metric("Rows after balancing", df_supp_balanced.shape[0])
    col2.metric("Balanced CV Accuracy", f"{scores_bal.mean():.2%}", f"± {scores_bal.std():.2%}")

    bal_report = classification_report(y_test_b, y_pred_b, zero_division=0, output_dict=True)
    bal_report_df = pd.DataFrame(bal_report).transpose().round(3)
    st.dataframe(bal_report_df)

    before_after_df = pd.DataFrame({
        'Version': ['Original (imbalanced)', 'Rebalanced'],
        'CV Accuracy': [scores_supp.mean(), scores_bal.mean()],
        'Class 0 F1-score': [supp_report['0']['f1-score'], bal_report['0']['f1-score']],
        'Class 1 F1-score': [supp_report['1']['f1-score'], bal_report['1']['f1-score']]
    })
    st.write("Before vs. after rebalancing:")
    st.dataframe(before_after_df.round(3))
else:
    st.write("Not enough minority-class samples in this pull to rebalance. Try refreshing the page to pull a new sample.")

st.subheader("Drug vs. Supplement: Side-by-Side")
st.warning(
    "The supplement model may show higher raw accuracy, but this can be misleading "
    "due to severe class imbalance (most supplement events are flagged serious). "
    "As with the Transaction project, high accuracy on imbalanced data can simply mean "
    "the model defaults to predicting the majority class — check precision and recall "
    "before trusting the accuracy number alone."
)
# =====================================================================
# MATCHED THERAPEUTIC CLASS: Drugs vs. Supplements  (new)
# =====================================================================


comparison_df = pd.DataFrame({
    'Metric': ['CV Accuracy', 'Serious Event Rate'],
    'Drug Model': [scores_new.mean(), y_new.mean()],
    'Supplement Model': [scores_supp.mean(), y_supp.mean()]
})
st.dataframe(comparison_df.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['Drugs', 'Supplements'], [y_new.mean(), y_supp.mean()], color=['steelblue', 'darkorange'])
axes[0].set_title('Serious Event Rate')
axes[0].set_ylabel('Proportion Serious')
axes[0].set_ylim(0, 1)
axes[1].bar(['Drugs', 'Supplements'], [scores_new.mean(), scores_supp.mean()], color=['steelblue', 'darkorange'])
axes[1].set_title('Model CV Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
plt.tight_layout()
st.pyplot(fig)

# =====================================================================
# MATCHED THERAPEUTIC CLASS: Drugs vs. Supplements
# =====================================================================
st.subheader("Matched Therapeutic Class: Drugs vs. Supplements")
st.write(
    "Rather than comparing all drugs to all supplements, this matches a single "
    "condition — the drug class that treats it (via the FDA's Established "
    "Pharmacologic Class) against supplements marketed for the same condition. "
    "Holding the condition constant makes the comparison closer to like-for-like."
)

# condition -> real FDA EPC strings (from the count endpoint) + supplement keywords
CONDITION_MAP = {
    "Diabetes / blood sugar": {
        "drug_epc": ["Sulfonylurea [EPC]", "Insulin Analog [EPC]",
                     "GLP-1 Receptor Agonist [EPC]",
                     "Sodium-Glucose Cotransporter 2 Inhibitor [EPC]"],
        "supp_keywords": ["berberine", "cinnamon", "chromium", "bitter melon", "alpha lipoic"],
    },
    "Cholesterol / heart": {
        "drug_epc": ["HMG-CoA Reductase Inhibitor [EPC]", "PCSK9 Inhibitor [EPC]",
                     "Dietary Cholesterol Absorption Inhibitor [EPC]"],
        "supp_keywords": ["red yeast rice", "niacin", "plant sterol", "fish oil", "omega", "garlic"],
    },
    "Depression / mood": {
        "drug_epc": ["Serotonin Reuptake Inhibitor [EPC]", "Mood Stabilizer [EPC]"],
        "supp_keywords": ["st john", "st. john", "sam-e", "saffron", "5-htp"],
    },
    "Joint / arthritis": {
        "drug_epc": ["Nonsteroidal Anti-inflammatory Drug [EPC]", "Antirheumatic Agent [EPC]"],
        "supp_keywords": ["glucosamine", "chondroitin", "turmeric", "curcumin", "msm"],
    },
}

condition = st.selectbox("Choose a condition to compare", list(CONDITION_MAP.keys()))
cfg = CONDITION_MAP[condition]

def drug_serious_rate(epc_list, limit=200):
    clauses = "+".join(f'patient.drug.openfda.pharm_class_epc:"{c.replace(" ", "+")}"'
                       for c in epc_list)
    url = f"https://api.fda.gov/drug/event.json?search=({clauses})&limit={limit}{KEY_PARAM}"
    data = fetch_fda_json(url)
    if data is None:
        return None, 0
    flags = [1 if int(r.get("serious", 2)) == 1 else 0 for r in data["results"]]
    n = len(flags)
    return (sum(flags) / n if n else None), n

def supp_serious_rate(keywords, limit=500):
    url = f"https://api.fda.gov/food/event.json?limit={limit}&search=products.industry_code:54{KEY_PARAM}"
    data = fetch_fda_json(url)
    if data is None:
        return None, 0
    flags = []
    for r in data["results"]:
        names = " ".join(p.get("name_brand", "").lower() for p in r.get("products", []))
        if any(k in names for k in keywords):
            outcomes = r.get("outcomes", [])
            flags.append(1 if any(o in serious_outcomes for o in outcomes) else 0)
    n = len(flags)
    return (sum(flags) / n if n else None), n

MIN_N = 10  # below this, a rate is noise, not signal

if st.button("Compare this class"):
    with st.spinner("Pulling matched-class data..."):
        d_rate, d_n = drug_serious_rate(cfg["drug_epc"])
        s_rate, s_n = supp_serious_rate(cfg["supp_keywords"])

    col1, col2 = st.columns(2)

    # Drug side
    if d_rate is not None and d_n >= MIN_N:
        col1.metric("Drug serious rate", f"{d_rate:.1%}", f"n = {d_n}")
    elif d_rate is not None:
        col1.warning(f"Only {d_n} matched drug events — too few to report a reliable rate.")
    else:
        col1.warning("No drug data returned for this class.")

    # Supplement side
    if s_rate is not None and s_n >= MIN_N:
        col2.metric("Supplement serious rate", f"{s_rate:.1%}", f"n = {s_n}")
    elif s_rate is not None:
        col2.warning(f"Only {s_n} matched supplement events — too few to report a reliable rate.")
    else:
        col2.warning("No supplements matched these keywords in this pull.")

    st.info(
        "Read with care: these are reported-event rates, not risk. FAERS (drugs) and "
        "CAERS (supplements) are different reporting systems with different volumes and "
        "reporters, so a gap may reflect who reports, not real danger. Sample sizes (n) "
        "and the supplement keyword list are shown so the matching stays transparent. "
        "Rates are suppressed when fewer than 10 events match."
    )

# =====================================================================
# SUMMARY: All matched classes at a glance
# =====================================================================
st.subheader("All Matched Classes at a Glance")
st.write(
    "Serious-event rates across every condition, drugs vs. supplements. "
    "Each bar is labeled with its sample size (n). Bars from fewer than "
    f"{MIN_N} events are shown faded — their rates are unreliable and should not be compared."
)

if st.button("Build summary across all conditions"):
    rows = []
    with st.spinner("Pulling all conditions..."):
        for cond_name, cond_cfg in CONDITION_MAP.items():
            d_rate, d_n = drug_serious_rate(cond_cfg["drug_epc"])
            s_rate, s_n = supp_serious_rate(cond_cfg["supp_keywords"])
            rows.append({
                "Condition": cond_name,
                "Drug rate": d_rate, "Drug n": d_n,
                "Supplement rate": s_rate, "Supplement n": s_n,
            })
    summary = pd.DataFrame(rows)

    # Show the numbers as a table first (honest, exact, includes n)
    st.dataframe(summary.round(3))

    # Grouped bar chart with n labels and faded low-n bars
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(summary))
    width = 0.38

    for i, row in summary.iterrows():
        # drug bar
        d_alpha = 1.0 if row["Drug n"] >= MIN_N else 0.35
        s_alpha = 1.0 if row["Supplement n"] >= MIN_N else 0.35
        d_val = row["Drug rate"] if row["Drug rate"] is not None else 0
        s_val = row["Supplement rate"] if row["Supplement rate"] is not None else 0

        ax.bar(x[i] - width/2, d_val, width, color="steelblue", alpha=d_alpha)
        ax.bar(x[i] + width/2, s_val, width, color="darkorange", alpha=s_alpha)
        ax.text(x[i] - width/2, d_val + 0.02, f"n={row['Drug n']}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, s_val + 0.02, f"n={row['Supplement n']}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["Condition"], rotation=20, ha="right")
    ax.set_ylabel("Serious-event rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Serious-Event Rate by Condition (faded = n < 10, unreliable)")
    ax.legend(["Drugs", "Supplements"])
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Faded bars come from very small supplement samples and are not "
        "statistically meaningful. The consistent finding is the reporting-system "
        "asymmetry (FAERS vs. CAERS), not a reliable drug-vs-supplement risk difference."
    )
#Added revised code above to rebuild data frame
# records_supp = []
# for record in supp_data['results']:
#     consumer = record.get('consumer', {})
#     outcomes = record.get('outcomes', [])
#     products = record.get('products', [])

#     # Target — any serious outcome?
#     is_serious = 1 if any(o in serious_outcomes for o in outcomes) else 0

#     # Gender
#     gender = consumer.get('gender', None)
#     gender_code = 1 if gender == 'Male' else (2 if gender == 'Female' else None)

#     # Product type
#     industry_code = products[0].get('industry_code', None) if products else None
#     is_supplement = 1 if industry_code == '54' else 0

#     records_supp.append({
#         'serious':       is_serious,
#         'gender':        gender_code,
#         'is_supplement': is_supplement
#     })

# df_supp = pd.DataFrame(records_supp)
# print(df_supp.shape)
# print(df_supp['serious'].value_counts())
# print(df_supp['is_supplement'].value_counts())
#print(df_supp.head())
