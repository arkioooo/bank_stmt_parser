# Streamlit PDF Parser

This project is a Streamlit application that allows users to upload files or folders containing PDF documents. The application processes these PDFs for parsing, extracting relevant information as specified.

## Project Structure

```
streamlit-pdf-parser
├── app.py
├── script.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, follow these steps:

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended):
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

## Usage

1. Upload a file or folder containing PDF documents.
2. The application will check for PDF files and process them accordingly.
3. View the parsed results directly in the application interface.

## Dependencies

The project requires the following Python packages:

- Streamlit
- PyMuPDF
- pdf2image
- pytesseract
- pandas
- fuzzywuzzy

Make sure to install these packages using the `requirements.txt` file provided.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.