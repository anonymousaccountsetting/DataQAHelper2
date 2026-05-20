import os
import json
from typing import Dict, Any, List

import pandas as pd
from pydantic import BaseModel, Field

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# =========================================================
# 0. API key
# =========================================================
# Option 1:
# Set your API key in the environment before running this script:
#   export OPENAI_API_KEY="your_api_key"
# or on Windows:
#   set OPENAI_API_KEY=your_api_key
#
# Option 2:
# Uncomment the following two lines and fill in your key directly.
# This is not recommended for public repositories.

# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================================================
# 1. Experiment settings
# =========================================================

OPENAI_MODEL = "gpt-4"
TEMPERATURE = 0.0

# Fill in the dataset path.
# Examples:
# DATASET_PATH = "regression_data/Realestate.csv"
# DATASET_PATH = "Realestate.csv"
DATASET_PATH = "YOUR_DATASET_PATH.csv"

# Fill in the background knowledge for the current experiment.
BACKGROUND_KNOWLEDGE = """
[Fill in the background knowledge here.]
For example:
This is a dataset about ...
Please use ... as independent variables and ... as the dependent variable and use a ... model to answer the following questions.
""".strip()

# Fill in the question list.
QUESTIONS = [
    "Question 1",
    "Question 2",
    "Question 3"
]

# Fill in the independent variables actually used in the experiment.
X_COLUMNS = [
    "X_COLUMN_1",
    "X_COLUMN_2",
    # "X_COLUMN_3",
    # ...
]

# Fill in the dependent variable.
Y_COLUMN = "Y_COLUMN"

# Supported task types:
# "regression", "binary_classification", "multiclass_classification"
TASK_TYPE = "regression"

# Fill in the configured model for the current experiment.
# Supported models:
# Regression:
#   "linear_regression"
#   "random_forest_regressor"
#   "gradient_boosting_regressor"
# Binary classification:
#   "logistic_regression"
#   "lda_classifier"
#   "ridge_classifier"
# Multiclass classification:
#   "lda_classifier"
#   "decision_tree_classifier"
#   "random_forest_classifier"
CONFIGURED_MODEL_NAME = "linear_regression"

ALLOWED_MODELS = {
    "regression": [
        "linear_regression",
        "random_forest_regressor",
        "gradient_boosting_regressor"
    ],
    "binary_classification": [
        "logistic_regression",
        "lda_classifier",
        "ridge_classifier"
    ],
    "multiclass_classification": [
        "lda_classifier",
        "decision_tree_classifier",
        "random_forest_classifier"
    ]
}

# Fill in the output filename if needed.
OUTPUT_PATH = "langchain_agent_outputs.json"

# =========================================================
# 2. Data loading and validation
# =========================================================

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df


def validate_columns(df: pd.DataFrame, x_columns: List[str], y_column: str) -> None:
    missing = [c for c in x_columns + [y_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def get_xy(df: pd.DataFrame, x_columns: List[str], y_column: str):
    validate_columns(df, x_columns, y_column)
    X = df[x_columns].copy()
    y = df[y_column].copy()
    return X, y

# =========================================================
# 3. Model runners
# =========================================================

def run_regression_model(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    model_name: str
) -> Dict[str, Any]:
    X, y = get_xy(df, x_columns, y_column)

    if model_name == "linear_regression":
        raw_model = LinearRegression()
        raw_model.fit(X, y)
        y_pred = raw_model.predict(X)
        r2 = r2_score(y, y_pred)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

        std_model = LinearRegression()
        std_model.fit(X_scaled, y_scaled)

        raw_coefficients = {
            col: float(coef) for col, coef in zip(x_columns, raw_model.coef_)
        }
        standardized_coefficients = {
            col: float(coef) for col, coef in zip(x_columns, std_model.coef_)
        }

        return {
            "task_type": "regression",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "r_squared": float(r2),
            "intercept": float(raw_model.intercept_),
            "raw_coefficients": raw_coefficients,
            "standardized_coefficients": standardized_coefficients,
        }

    elif model_name == "random_forest_regressor":
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        feature_importances = {
            col: float(score) for col, score in zip(x_columns, model.feature_importances_)
        }

        return {
            "task_type": "regression",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "r_squared": float(r2),
            "feature_importances": feature_importances
        }

    elif model_name == "gradient_boosting_regressor":
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        feature_importances = {
            col: float(score) for col, score in zip(x_columns, model.feature_importances_)
        }

        return {
            "task_type": "regression",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "r_squared": float(r2),
            "feature_importances": feature_importances
        }

    else:
        raise ValueError(f"Unsupported regression model: {model_name}")


def run_binary_model(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    model_name: str
) -> Dict[str, Any]:
    X, y = get_xy(df, x_columns, y_column)

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        raw_coefficients = {
            col: float(coef) for col, coef in zip(x_columns, model.coef_[0])
        }

        return {
            "task_type": "binary_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "coefficients": raw_coefficients
        }

    elif model_name == "lda_classifier":
        model = LinearDiscriminantAnalysis()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        coef = model.coef_[0]
        coefficients = {col: float(c) for col, c in zip(x_columns, coef)}

        return {
            "task_type": "binary_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "coefficients": coefficients
        }

    elif model_name == "ridge_classifier":
        model = RidgeClassifier()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        coef = model.coef_[0]
        coefficients = {col: float(c) for col, c in zip(x_columns, coef)}

        return {
            "task_type": "binary_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "coefficients": coefficients
        }

    else:
        raise ValueError(f"Unsupported binary classification model: {model_name}")


def run_multiclass_model(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    model_name: str
) -> Dict[str, Any]:
    X, y = get_xy(df, x_columns, y_column)

    if model_name == "lda_classifier":
        model = LinearDiscriminantAnalysis()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        abs_coef = abs(model.coef_).mean(axis=0)
        scores = {col: float(v) for col, v in zip(x_columns, abs_coef)}

        return {
            "task_type": "multiclass_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "mean_absolute_coefficients": scores
        }

    elif model_name == "decision_tree_classifier":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        scores = {col: float(v) for col, v in zip(x_columns, model.feature_importances_)}

        return {
            "task_type": "multiclass_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "feature_importances": scores
        }

    elif model_name == "random_forest_classifier":
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        scores = {col: float(v) for col, v in zip(x_columns, model.feature_importances_)}

        return {
            "task_type": "multiclass_classification",
            "model_name": model_name,
            "x_columns": x_columns,
            "y_column": y_column,
            "training_accuracy": float(acc),
            "feature_importances": scores
        }

    else:
        raise ValueError(f"Unsupported multiclass classification model: {model_name}")

# =========================================================
# 4. LangChain tools
# =========================================================

class InspectDatasetInput(BaseModel):
    dataset_path: str = Field(description="Path to the CSV dataset file.")


@tool(args_schema=InspectDatasetInput)
def inspect_dataset(dataset_path: str) -> str:
    """Inspect the dataset and return its columns, shape, and preview rows."""
    df = load_dataset(dataset_path)
    result = {
        "shape": df.shape,
        "columns": list(df.columns),
        "preview_rows": df.head(5).to_dict(orient="records")
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


class RunConfiguredAnalysisInput(BaseModel):
    dataset_path: str = Field(description="Path to the CSV dataset file.")


@tool(args_schema=RunConfiguredAnalysisInput)
def run_configured_analysis(dataset_path: str) -> str:
    """
    Run the configured analysis for the current experiment settings.
    The task type, X columns, Y column, and configured model are defined in the experiment settings.
    """
    df = load_dataset(dataset_path)

    configured_model = CONFIGURED_MODEL_NAME

    if configured_model not in ALLOWED_MODELS[TASK_TYPE]:
        raise ValueError(
            f"Configured model '{configured_model}' is not allowed for task type '{TASK_TYPE}'. "
            f"Allowed models: {ALLOWED_MODELS[TASK_TYPE]}"
        )

    if TASK_TYPE == "regression":
        result = run_regression_model(df, X_COLUMNS, Y_COLUMN, configured_model)
    elif TASK_TYPE == "binary_classification":
        result = run_binary_model(df, X_COLUMNS, Y_COLUMN, configured_model)
    elif TASK_TYPE == "multiclass_classification":
        result = run_multiclass_model(df, X_COLUMNS, Y_COLUMN, configured_model)
    else:
        raise ValueError(f"Unsupported task type: {TASK_TYPE}")

    return json.dumps(result, ensure_ascii=False, indent=2)

# =========================================================
# 5. Build agent
# =========================================================

def build_agent():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
    )

    tools = [
        inspect_dataset,
        run_configured_analysis,
    ]

    allowed_model_text = ", ".join(ALLOWED_MODELS[TASK_TYPE])

    system_prompt = f"""
You are a LangChain-based agent baseline for analytical textual reporting.

Your task is to answer one user question at a time about a structured dataset.
You should use the available tools when analytical evidence is needed.
Do not invent analytical results.

Current experiment settings:
- Task type: {TASK_TYPE}
- X columns: {X_COLUMNS}
- Y column: {Y_COLUMN}
- Allowed models for this task: {allowed_model_text}
- Configured model for this experiment: {CONFIGURED_MODEL_NAME}

Rules:
1. Use only the configured model for this experiment.
2. Do not assume analytical results without calling tools.
3. Use inspect_dataset if you need to verify the dataset structure.
4. Use run_configured_analysis as the analysis tool for this experiment.
5. Base the final answer on the returned analytical results.
6. Do not use any hidden question bank, answer template, or predefined interpretive schema.
7. Keep the answer clear and grounded in the tool outputs and the provided background knowledge.
""".strip()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent

# =========================================================
# 6. Agent query function
# =========================================================

def ask_agent(agent, dataset_path: str, background_knowledge: str, question: str) -> str:
    prompt = f"""
Dataset path: {dataset_path}

Background knowledge:
{background_knowledge}

Question:
{question}

Please answer the question using the available tools when analytical evidence is needed.
""".strip()

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    messages = result["messages"]
    final_message = messages[-1]
    return getattr(final_message, "content", str(final_message))

# =========================================================
# 7. Main experiment loop
# =========================================================

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY in your environment.")

    agent = build_agent()

    outputs = {
        "dataset_path": DATASET_PATH,
        "background_knowledge": BACKGROUND_KNOWLEDGE,
        "task_type": TASK_TYPE,
        "configured_model_name": CONFIGURED_MODEL_NAME,
        "x_columns": X_COLUMNS,
        "y_column": Y_COLUMN,
        "allowed_models": ALLOWED_MODELS[TASK_TYPE],
        "questions_and_answers": []
    }

    for question in QUESTIONS:
        answer = ask_agent(agent, DATASET_PATH, BACKGROUND_KNOWLEDGE, question)
        outputs["questions_and_answers"].append(
            {
                "question": question,
                "answer": answer
            }
        )
        print("=" * 80)
        print(question)
        print("-" * 80)
        print(answer)
        print()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()