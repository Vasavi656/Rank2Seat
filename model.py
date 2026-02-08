# model.py
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------
# NORMALIZATION FUNCTIONS
# ------------------------------------
def normalize_college(name):
    if pd.isna(name):
        return None
    x = str(name).upper()
    x = x.replace(".", "").replace(",", "")
    x = x.replace("ENGG", "ENGINEERING")
    x = x.replace("ENG", "ENGINEERING")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_category(cat):
    if pd.isna(cat):
        return None
    c = str(cat).upper().strip().replace("_", "").replace("-", "")
    mapping = {
        "BCA": "BC-A",
        "BCB": "BC-B",
        "BCC": "BC-C",
        "BCD": "BC-D",
        "BCE": "BC-E",
        "OC": "OC",
        "SC": "SC",
        "ST": "ST",
        "EWS": "OC_EWS",
        "OCEWS": "OC_EWS",
    }
    return mapping.get(c, c)


def normalize_gender(g):
    if pd.isna(g):
        return None
    g = str(g).upper().strip()
    if g in ["F", "FEMALE"]:
        return "F"
    if g in ["M", "MALE"]:
        return "M"
    return g


# ------------------------------------
# LOAD ALL YEARS DATA
# ------------------------------------
files = [
    "sample_data/2019.csv",
    "sample_data/2020.csv",
    "sample_data/2022.csv",
    "sample_data/2023.csv",
    "sample_data/2024.csv",
]

dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

# Normalize dataset
data["CollegeName"] = data["CollegeName"].apply(normalize_college)
data["Branch"] = data["Branch"].astype(str).str.upper().str.strip()
data["Gender"] = data["Gender"].apply(normalize_gender)
data["Category"] = data["Category"].apply(normalize_category)

# ------------------------------------
# PREPARE ML TRAINING DATA
# ------------------------------------
rows = []
for _, r in data.iterrows():
    cutoff = int(r["CutoffRank"])
    for diff in [-15000, -8000, -3000, -1000, -200, 200, 1000, 3000, 8000]:
        rank = cutoff - diff
        if rank <= 0:
            continue
        rows.append({
            "RankDiff": cutoff - rank,
            "Seat": 1 if rank <= cutoff else 0
        })

train_df = pd.DataFrame(rows)
X = train_df[["RankDiff"]]
y = train_df["Seat"]

# ------------------------------------
# TRAIN ML MODEL
# ------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])
model.fit(X, y)

# ------------------------------------
# PREDICTION FUNCTION
# ------------------------------------
def predict_probability(rank, college_code, branch, category, gender):
    category = normalize_category(category)
    branch = branch.upper().strip()
    gender = normalize_gender(gender)

    subset = data[
        (data["CollegeCode"] == college_code) &
        (data["Branch"] == branch) &
        (data["Category"] == category) &
        (data["Gender"] == gender)
    ]

    if subset.empty:
        return None, None, None, None

    avg_cutoff = int(subset["CutoffRank"].mean())
    latest_cutoff = int(subset["CutoffRank"].iloc[-1])

    rank_diff = avg_cutoff - rank
    probability = model.predict_proba([[rank_diff]])[0][1] * 100
    probability = round(min(probability, 99.0), 2)

    yearwise = subset  # optional for future tables

    return probability, avg_cutoff, latest_cutoff, yearwise
