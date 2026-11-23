# Dataset

- **Source**: Kaggle, *Heart Failure Prediction Dataset* by `fedesoriano`
  (<https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>)
- **Rows**: 918
- **Features**: 11 clinical features
- **Target**: `HeartDisease` (1 = disease, 0 = no disease)
- **File**: `data/raw/heart.csv`

A copy is committed for reproducibility. Check the original page for the current license.
Educational use only; not for diagnosis or treatment decisions.

## Schema

`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`,
`MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, `HeartDisease`.

`Cholesterol = 0` and `RestingBP = 0` are treated as missing, not real measurements.
