import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ======================================================
# NORMALIZATION FUNCTIONS
# ======================================================

def normalize_college(name):
    if pd.isna(name):
        return name

    name = str(name).upper().strip()

    # remove punctuation
    name = name.replace(".", "").replace(",", "")

    # standardize common words
    name = name.replace("ENGG", "ENGINEERING")
    name = name.replace("ENG ", "ENGINEERING ")

    # remove multiple spaces
    name = re.sub(r"\s+", " ", name)

    return name


def normalize_category(cat):
    if pd.isna(cat):
        return cat

    cat = str(cat).upper().strip()
    cat = cat.replace("_", "").replace("-", "").replace(" ", "")

    mapping = {
        "BCA": "BC-A",
        "BCB": "BC-B",
        "BCC": "BC-C",
        "BCD": "BC-D",
        "BCE": "BC-E",
        "OCEWS": "OC-EWS",
        "EWS": "OC-EWS",
        "OC": "OC",
        "SC": "SC",
        "ST": "ST"
    }

    return mapping.get(cat, cat)


def normalize_gender(g):
    if pd.isna(g):
        return g

    g = str(g).upper().strip()

    if g in ["FEMALE", "F"]:
        return "F"
    if g in ["MALE", "M"]:
        return "M"

    return g


# ======================================================
# LOAD AND CLEAN DATA
# ======================================================

def load_data():

    files = [
        ("sample_data/2019.csv", 2019),
        ("sample_data/2020.csv", 2020),
        ("sample_data/2022.csv", 2022),
        ("sample_data/2023.csv", 2023),
        ("sample_data/2024.csv", 2024),
    ]

    dfs = []

    for file, year in files:
        df = pd.read_csv(file)
        df["Year"] = year
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Apply normalization
    df["CollegeName"] = df["CollegeName"].apply(normalize_college)
    df["Category"] = df["Category"].apply(normalize_category)
    df["Gender"] = df["Gender"].apply(normalize_gender)
    df["Branch"] = df["Branch"].astype(str).str.upper().str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    return df


data = load_data()


# ======================================================
# GENERATE SYNTHETIC TRAINING DATA
# ======================================================

rows = []

for _, r in data.iterrows():
    cutoff = int(r["CutoffRank"])

    for rank in range(max(1, cutoff - 6000), cutoff + 6000, 1500):

        seat = 1 if rank <= cutoff else 0

        rows.append({
            "UserRank": rank,
            "Category": r["Category"],
            "Gender": r["Gender"],
            "CollegeName": r["CollegeName"],
            "Branch": r["Branch"],
            "Year": r["Year"],
            "SeatPossible": seat
        })

train_df = pd.DataFrame(rows)


# ======================================================
# ENCODE CATEGORICAL FEATURES
# ======================================================

encoders = {}

for col in ["Category", "Gender", "CollegeName", "Branch"]:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

X = train_df[[
    "UserRank",
    "Category",
    "Gender",
    "CollegeName",
    "Branch",
    "Year"
]]

y = train_df["SeatPossible"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Logistic Regression Model Trained Successfully")


# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_probability(rank, college_name, branch, category, gender):

    input_df = pd.DataFrame([{
        "UserRank": rank,
        "Category": encoders["Category"].transform([normalize_category(category)])[0],
        "Gender": encoders["Gender"].transform([normalize_gender(gender)])[0],
        "CollegeName": encoders["CollegeName"].transform([normalize_college(college_name)])[0],
        "Branch": encoders["Branch"].transform([branch.upper().strip()])[0],
        "Year": 2024
    }])

    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    return round(prob * 100, 2)
