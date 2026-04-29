import pandas as pd
import pickle
import numpy as np

def prepare_dataframe(df, model, label_encoders, scaler):

    df1 = df.copy()

    # Feature Engineering
    df1["NumberOfLanguagesKnown"] = df1["LanguageHaveWorkedWith"].apply(
        lambda x: 0 if str(x) == "nan" or x == "" else len(str(x).split(";"))
    )

    df1["NumberOfDatabasesKnown"] = df1["DatabaseHaveWorkedWith"].apply(
        lambda x: 0 if str(x) == "nan" or x == "" else len(str(x).split(";"))
    )

    df1["NumberOfLearningSources"] = df1["LearnCode"].apply(
        lambda x: 0 if str(x) == "nan" or x == "" else len(str(x).split(";"))
    )

    # Convert numeric
    df1["YearsCode"] = pd.to_numeric(df1["YearsCode"], errors="coerce").fillna(0).astype(int)
    df1["YearsCodePro"] = pd.to_numeric(df1["YearsCodePro"], errors="coerce").fillna(0).astype(int)
    df1["WorkExp"] = pd.to_numeric(df1["WorkExp"], errors="coerce").fillna(0).astype(int)

    # Categorize into bins like training
    bins = [0, 2, 5, 10, 20, 30, 40, 50, float('inf')]

    df1['ExperienceCategory'] = pd.cut(df1['WorkExp'], bins=bins, labels=False, right=False).fillna(0).astype(int)
    df1['YearsCodeCategory'] = pd.cut(df1['YearsCode'], bins=bins, labels=False, right=False).fillna(0).astype(int)
    df1['YearsCodeProCategory'] = pd.cut(df1['YearsCodePro'], bins=bins, labels=False, right=False).fillna(0).astype(int)

    # Columns to use
    train_columns = [
       'Age',
       'AISelect',
       'OrgSize',
       'DevType',
       "RemoteWork",
       'Currency',
       "EdLevel",
       "ExperienceCategory",
       "YearsCodeCategory",
       "YearsCodeProCategory",
       "NumberOfDatabasesKnown",
       "NumberOfLanguagesKnown",
       "NumberOfLearningSources"
    ]

    X = df1[train_columns].copy()

    # Encode categorical columns with safe fallback for unknown values
    for col in train_columns:
        if col in label_encoders:
            classes = list(label_encoders[col].classes_)
            is_numeric_classes = len(classes) > 0 and np.issubdtype(type(classes[0]), np.integer)

            def map_category(value):
                if pd.isna(value):
                    if is_numeric_classes:
                        return classes[0] if classes else 0
                    return classes[0] if classes else ""

                if is_numeric_classes:
                    try:
                        numeric_value = int(value)
                    except Exception:
                        numeric_value = None

                    if numeric_value in classes:
                        return numeric_value
                    if len(classes) > 0 and numeric_value is not None:
                        return min(classes, key=lambda x: abs(int(x) - numeric_value))
                    return classes[0] if classes else 0

                value_str = str(value).strip()
                if value_str in classes:
                    return value_str
                if "Other" in classes:
                    return "Other"
                return classes[0] if classes else value_str

            X[col] = X[col].apply(map_category)
            X[col] = label_encoders[col].transform(X[col])

    # Scale the data
    X_scaled = scaler.transform(X)

    # Get prediction
    prediction = model.predict(X_scaled)

    return prediction