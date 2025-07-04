import streamlit as st
import os
import tempfile
import pandas as pd
from script import process_statement

st.set_page_config(page_title="PDF Bank Statement Parser", layout="wide")
st.title("PDF Bank Statement Parser")

poppler_bin = r"D:\Mridul.Intern\poppler-23.05.0\Library\bin"  # Change as needed

def save_uploaded_file(uploadedfile, save_dir):
    file_path = os.path.join(save_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = save_uploaded_file(uploaded_file, tmpdir)
        output_file = os.path.join(tmpdir, uploaded_file.name.replace(".pdf", ".xlsx"))
        with st.spinner("Processing the file"):
            df, acc_no, acc_name = process_statement(pdf_path, output_file, poppler_bin=poppler_bin)
        st.success("Processed")
        st.write("**Account No :**", acc_no)
        st.write("**Account Holder  :**", acc_name)
        st.dataframe(df)
        # st.do wnload_button("Download Excel", df.to_excel(index=False), file_name="parsed.xlsx")