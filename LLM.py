import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import re
import os
import json
import streamlit as st

# load_dotenv()

# Get the API key
# api_key = os.getenv("GEMINI_API_KEY")
api_key = st.secrets["GEMINI_API_KEY"]
# Configure
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

def llm_response(df: pd.DataFrame, recommendations: list):
    schema_info = df.dtypes.to_dict()
    missing_info = df.isna().sum().to_dict()
    sample_data = df.sample(min(len(df), 20), random_state=42).to_dict()

    prompt = f"""
    I have a dataset with the following information:

    Column data types:
    {schema_info}

    Missing value counts:
    {missing_info}

    Sample rows:
    {sample_data}

    The data quality checker found the following issues and recommendations:
    {recommendations}

    Write ONLY Python code that cleans the entire DataFrame `df` according to the above recommendations. 
    Specific instructions:
    - Address the issues listed (e.g., standardize 'city', fix 0 prices, handle outliers, correct data types, etc.)
    - If a column requires standardization, you may use mapping/dictionary logic or simple replacements based on the sample.
    - For numeric issues (like outliers, 0 values), handle them using imputation, capping, or removal as appropriate.
    - Fill missing values (mean/median for numeric, mode for categorical).
    - Remove duplicates if any.
    - Standardize only when confident that values are duplicates/variants.
    - Never overwrite values that already match valid, common categories.   
    - If unsure whether to map a value, leave it unchanged.
    - For categorical columns, create mappings only where obvious (e.g., "navy blue" → "blue").
    - For location/city fields, do NOT map regions or provinces to cities (e.g., "Punjab" must stay as "Punjab").
    - Avoid excessive collapsing of categories; preserve unique but valid values.
    - Do NOT subset the DataFrame with head(), tail(), or sample().
    - Do not create new fake data.
    - The final line must be: `df_cleaned = df`

    Assume pandas is imported as pd and numpy as np.
    """

    response = model.generate_content(prompt)

    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response.text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return response.text.strip()

 

def check_data_quality(df):
    schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.head(3).to_dict(orient="records")
    null_counts = df.isnull().sum().to_dict()
    basic_stats = df.describe(include="all").to_dict()

    llm_prompt = f"""
    You are a data quality checker.
    Dataset schema: {schema_info}
    Sample rows: {sample_rows}
    Missing values: {null_counts}   
    Stats: {basic_stats}

    Analyze the dataset and tell:
    1. Does this dataset need cleaning? (yes/no)
    2. If yes, list specific issues found (e.g., missing values, wrong datatypes, duplicates).
    3. Identify issues like inconsistent categories, spelling variations, or invalid entries.
    4. Do NOT recommend changing values that are already valid and consistent.
    5. Recommend mappings only when you detect clear duplicates (e.g., "navy blue" and "dark blue" → "blue").
    6. If a value looks valid but unfamiliar (e.g., a rare city name), mark it as "unverified" instead of forcing a mapping.
    7. Suggest cleaning actions in plain English.
    
    Respond in JSON format ONLY:   
    {{
      "needs_cleaning": true/false,
      "issues": ["..."],
      "recommendations": ["..."]
    }}
    """

    response = model.generate_content(llm_prompt)          
    raw_text = response.text.strip()

    # Extract JSON safely
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())  #  Always return dict
        except json.JSONDecodeError:
            return {
                "needs_cleaning": False,
                "issues": ["LLM returned invalid JSON"],
                "recommendations": []
            }
    else:
        return {
            "needs_cleaning": False,
            "issues": ["LLM did not return JSON"],
            "recommendations": []
        }
    


def get_viz_suggestions(df):
    schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.head().to_dict(orient="records")

    prompt = f"""
    You are a data analyst. Analyze the dataset and suggest visualizations to explore it fully.
    Schema: {schema_info}
    Sample rows: {sample_rows}

    Instructions:
    1. Suggest plots for distribution of numeric columns (histogram).
    2. Suggest plots for relationships between numeric columns (scatter, line, pairplot).
    3. Suggest plots for categorical vs numeric (bar, box, violin).
    4. Include correlation heatmap for numeric columns.
    5. Only return JSON. Each plot must be an object with "type" and "x" (and "y" if needed).  
    6. Suggest between 5 and 10 plots.

    JSON example format:
    [
      {{"type": "histogram", "x": "column_name"}},
      {{"type": "scatter", "x": "column_x", "y": "column_y"}},
      {{"type": "bar", "x": "column", "y": "column"}},
      {{"type": "box", "x": "categorical_col", "y": "numeric_col"}},
      {{"type": "heatmap", "x": ["numeric_col1", "numeric_col2"]}},
      {{"type": "pairplot", "x": ["num_col1", "num_col2", "num_col3"]}}
    ]
    """

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Remove code fences if present
    raw_text = re.sub(r"^```(?:json|python)?\n?", "", raw_text)
    raw_text = re.sub(r"```$", "", raw_text).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return []


import re

def clean_sql(sql_text: str) -> str:
    """
    Extracts the SQL query from Gemini output.
    Handles cases with ```sql ... ``` or raw SQL.
    """
    sql_text = sql_text.strip()

    # Look for a code block like ```sql ... ```
    match = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)```", sql_text)
    if match:
        return match.group(1).strip()

    # If no code fences, just return text
    return sql_text.strip()
  

def nl_to_sql(prompt, schema):
    full_prompt = f"""
    You are an expert SQLite query generator.
    Table name: df_cleaned    
    Schema: {schema}
    User request: {prompt}
    Return only the raw SQL query. Do NOT include markdown, comments, or explanations.
    """

    response = model.generate_content(full_prompt)

    # Clean the response properly
    sql_query = clean_sql(response.text)

    return sql_query    
 

def explain_plot(vis_metadata, df_metadata):   
    prompt = f"""   
    You are a data analyst. Summarize the following plot analysis in a concise, structured json format:
    Plot type: {vis_metadata.get("type")}
    X-axis: {vis_metadata.get("x")}
    Y-axis: {vis_metadata.get("y")}
    Dataset info: {df_metadata if df_metadata else "not provided"}.

    Return a structured JSON with exactly these keys:
        - "what_it_shows": (list of 1-3 short bullet points about what the plot shows)   
        - "key_insights": (list of 2-5 concise insights about patterns, trends, or outliers)
        - "how_to_use": (list of 2-3 practical ways this plot can help in analysis/decision making)
    """    

    response = model.generate_content(prompt)

    text = response.text.strip()

    # Extract JSON block from response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"what_it_shows": [text], "key_insights": [], "how_to_use": []}

    json_str = match.group(0)

    try:
        explanation = json.loads(json_str)
    except json.JSONDecodeError:
        explanation = {"what_it_shows": [text], "key_insights": [], "how_to_use": []}
    
    return explanation   
