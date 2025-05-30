import pandas as pd
import sqlalchemy
import streamlit as st

engine = sqlalchemy.create_engine('postgresql://admin:admin@localhost/defectdb')
df = pd.read_sql("SELECT * FROM detections ORDER BY timestamp DESC", engine)

st.title("Defect Detection Dashboard")
st.dataframe(df)
