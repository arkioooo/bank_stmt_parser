# Streamlit PDF Parser

This project is a Streamlit application that allows users to upload files or folders containing PDF bank statements. The application processes these PDFs for parsing, extracting relevant information.
! MAY NOT WORK FOR ALL BANKS YET !

## Project Structure

``` 
streamlit-pdf-parser
├── app.py - stages streamlit app
├── script.py - provides logic and code for app.py
├── requirements.txt
└── README.md

├── colat_bk_stmt.ipynb - script.py copy for processing folders
├── detect_desc.ipynb - bank statement description cleaning and entity detection
├── finetune_spacy_md.ipynb - [ongoing] finetune NER for description entity detection
├── rtwete.ipynb - testing file
├── test.ipynb - testing file
```

## Installation

To set up the project, follow these steps:

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit application, use the following command:

```
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.
