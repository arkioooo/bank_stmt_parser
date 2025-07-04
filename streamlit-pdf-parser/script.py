import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re
import os
from fuzzywuzzy import fuzz

# classifications dict
CLASSIFICATIONS = {
    'ECS/NACH': ['ecs', 'nach', 'ach', 'ift'],
    'IMPS': ['imps'],
    'NEFT': ['neft'],
    'RTGS': ['rtgs'],
    'ESIC' : ['esic'],
    'Interest' : ['interest'],
    'Refund/Reversal' : ['refund', 'reversal', 'rev'],
    'Salary': ['salary', 'stipend'],
    'Tax': ['tax', 'duty', 'customs'],
    'Credit Card' : ['credit card','cc'],
    'Debit Card' : ['debit card', 'dc'],
    'Bank Instrument' : ['dd', 'commissioner'],
    'Cash Txn' : ['cash', 'withdrawal', 'deposit'],
    'Cheque Txn' : ['cheque', 'chq', 'clearing', 'clg'],
    'Company Expense' : ['expense', 'business', 'corporate', 'travel', 
                         'home', 'charges', 'employee', 'fund', 'reimbursement', 
                         'renumeration', 'leave'],
    'Forex' : ['forex', 'brn'],
    'Insurance' : ['insurance', 'premium'],
    'Rent' : ['rent'],
    'UPI' : ['upi'],
    'Other': []
}

desc_list = ['description', 'remark', 'transaction', 'detail', 'particulars']

def extract_account_info_from_text(text, pdf_path=None, bbox_acc_no=None, poppler_bin=None, page_num=0):
    """
    Extract account number and account holder name from PDF text content.
    If not found, use OCR on the provided bbox to extract account number.
    """
    account_number = None
    account_holder = None

    # regex for acc no (allowing for alphanumeric, e.g. SBIN0001234567)
    acc_no_patterns = [
        r'Account\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
        r'A/c\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
        r'Account\s*Number\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
        r'Acc\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})'
    ]

    for pattern in acc_no_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            account_number = match.group(1)
            break

    # OCR fallback if not found
    if not account_number and pdf_path and bbox_acc_no:
        ocr_text = ocr_text_from_bbox(pdf_path, bbox_acc_no, poppler_bin, page_num)
        # Try to extract account number from OCR text
        for pattern in acc_no_patterns:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                account_number = match.group(1)
                break
        # If still not found, try to find any long alphanumeric sequence
        if not account_number:
            generic_match = re.search(r'\b[A-Z0-9]{6,}\b', ocr_text)
            if generic_match:
                account_number = generic_match.group(0)

    # Account holder extraction (same as your logic)
    lines = text.split('\n')
    for i, line in enumerate(lines[:30]):
        line = line.strip()
        if not line:
            continue
        lcline = line.lower()
        # Skip lines with known non-name keywords or numbers
        if any(skip in lcline for skip in ['statement', 'account', 'period', 'bank', 'branch', 'customer', 'number', 'date', 'summary', 'address']):
            continue
        if any(char.isdigit() for char in line):
            continue
        # Skip lines with special chars
        if re.search(r'[^a-zA-Z\s\.]', line):
            continue
        # Heuristic: 2-4 words, each word capitalized or uppercase, not too long
        words = line.split()
        if 1 < len(words) <= 4 and all(w[0].isupper() or w.isupper() for w in words):
            account_holder = line
            break

    # look for patterns around "Joint Holder" or similar
    joint_holder_match = re.search(r'([A-Z\s]+)\s*Joint\s+Holder', text, re.IGNORECASE)
    if joint_holder_match and not account_holder:
        potential_name = joint_holder_match.group(1).strip()
        if len(potential_name) > 3:
            account_holder = potential_name

    return account_number, account_holder

def extract_text_from_pdf(pdf_path):
    """
    Extract all text content from PDF for parsing account information.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()
    
    doc.close()
    return full_text

def ocr_page_to_dataframe(pdf_path, page_num, poppler_bin=None):
    """
    Use OCR to extract tabular data from a PDF page when table detection fails.
    """
    try:
        # print(f"Using OCR for page {page_num + 1}...")
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_bin, first_page=page_num+1, last_page=page_num+1)
        
        if not images:
            return pd.DataFrame()
        
        img = images[0]
        
        # tesseract.exe path
        if hasattr(pytesseract, 'pytesseract'):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mridul.intern\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        
        text = pytesseract.image_to_string(img, config='--psm 6')
        return parse_ocr_text_to_dataframe(text)
        
    except Exception as e:
        # print(f"OCR failed for page {page_num + 1}: {e}")
        return pd.DataFrame()

def parse_ocr_text_to_dataframe(text):
    """
    Parse OCR text into a structured dataframe for bank statements.
    """
    lines = text.split('\n')
    rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if any(skip in line.lower() for skip in ['opening balance', 'closing balance', 'transaction total', 'statement', 'account', 'branch', 'address']):
            continue
        
        date_match = re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', line)
        if date_match:
            parts = line.split()
            if len(parts) >= 4: # min parts in transaction
                try:
                    date = date_match.group(1)
                    amounts = []
                    description_parts = []
                    
                    for part in parts:
                        clean_part = part.replace(',', '').replace('(', '').replace(')', '')
                        try:
                            amount = float(clean_part)
                            amounts.append(amount)
                        except ValueError:
                            if part != date:
                                description_parts.append(part)
                    
                    if amounts:
                        description = ' '.join(description_parts)
                        debit = None
                        credit = None
                        balance = None
                        
                        if len(amounts) >= 3:
                            balance = amounts[-1]  # last amount is usually balance
                            if len(amounts) == 3:
                                if amounts[0] != 0:
                                    debit = amounts[0]
                                if amounts[1] != 0:
                                    credit = amounts[1]
                        elif len(amounts) == 2:
                            balance = amounts[-1]
                            if 'cr' in line.lower() or 'credit' in line.lower():
                                credit = amounts[0]
                            else:
                                debit = amounts[0]
                        elif len(amounts) == 1:
                            balance = amounts[0]
                        
                        rows.append({
                            'date': date,
                            'description': description,
                            'debit': debit,
                            'credit': credit,
                            'balance': balance
                        })
                        
                except Exception as e:
                    continue
    
    if rows:
        df = pd.DataFrame(rows)
        # print(f"OCR extracted {len(df)} potential transactions")
        return df
    else:
        return pd.DataFrame()

def extract_tables(pdf_path, poppler_bin=None):
    """
    Extract tables from PDF and return combined dataframe.
    Uses OCR as fallback when table detection fails.
    """
    doc = fitz.open(pdf_path)
    dfs = []
    
    print(f"Processing {len(doc)} pages")
    
    for page_num, page in enumerate(doc):
        # print(f"Processing page {page_num + 1}...")
        tables = page.find_tables()
        page_has_data = False
        
        # fitz table detection
        for table_num, tbl in enumerate(tables):
            try:
                df = tbl.to_pandas()
                if not df.empty:
                    df = df.dropna(how='all')
                    string_cols = df.select_dtypes(include=['object']).columns
                    if len(string_cols) > 0:
                        mask = df[string_cols].astype(str).apply(lambda x: x.str.strip()).replace('', pd.NA).notna().any(axis=1)
                        df = df[mask]
                    df = df[df.isna().sum(axis=1) <= 2]
                    if not df.empty:
                        dfs.append(df)
                        page_has_data = True
            except Exception as e:
                print(f"Error processing table {table_num + 1} on page {page_num + 1}: {e}")
                continue
        
        # or try OCR as fallback
        if not page_has_data:
            print(f"No valid tables found on page {page_num + 1}, trying OCR")
            ocr_df = ocr_page_to_dataframe(pdf_path, page_num, poppler_bin)
            if not ocr_df.empty:
                ocr_df = ocr_df[ocr_df.isna().sum(axis=1) <= 2]
                if not ocr_df.empty:
                    dfs.append(ocr_df)
                    print(f"OCR added {len(ocr_df)} rows from page {page_num + 1}")
    
    doc.close()
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total combined rows: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

def ocr_text_from_bbox(pdf_path, bbox, poppler_bin=None, page_num=0):
    """
    Crop the page image using bbox and perform OCR on cropped image.
    bbox format: (left, upper, right, lower)
    """
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_bin)
        if page_num >= len(images):
            return ""
            
        img = images[page_num].crop()
        
        # set tesseract.exe path
        if hasattr(pytesseract, 'pytesseract'):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mridul.intern\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        
        text = pytesseract.image_to_string(img, config='--psm 6').strip()
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def classify_transaction(description):
    """
    Classify transactions based on keywords in description.
    """
    if pd.isna(description):
        return 'Other'
        
    description_lower = str(description).lower()
    
    for category, keywords in CLASSIFICATIONS.items():
        if any(keyword in description_lower for keyword in keywords):
            return category
    return 'Other'

def normalize(df):
    """
    Normalize and clean the extracted dataframe.
    """
    if df.empty:
        columns = ['serial_no', 'account_holders_name', 'date', 'month_year', 'description', 'debit', 'credit', 'balance', 'classification']
        return pd.DataFrame(columns=columns)

    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r'\s+', '_', regex=True)
                    .str.replace(r'\.', '', regex=True))

    # Find the first description-like column present
    desc_cols_found = [col for col in desc_list if col in df.columns]
    if desc_cols_found:
        desc_col = desc_cols_found[0]
        df.rename(columns={desc_col: 'description'}, inplace=True)
    else:
        df['description'] = pd.NA

    date_columns = [col for col in df.columns if 'date' in col]
    if date_columns:
        df['date'] = pd.to_datetime(df[date_columns[0]], dayfirst=True, errors='coerce').dt.strftime('%d-%m-%Y')
        df['month_year'] = pd.to_datetime(df[date_columns[0]], dayfirst=True, errors='coerce').dt.strftime('%m-%Y')
    else:
        df['date'] = None
        df['month_year'] = None

    df['serial_no'] = range(1, len(df) + 1)

    for col in ['debit', 'credit', 'balance']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').replace({'nan': None, 'NaN': None, '': None})
            df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce') if x is not None else pd.NA)
        else:
            df[col] = pd.NA

    # classification
    df['classification'] = df['description'].apply(classify_transaction)
    df['account_holders_name'] = None

    return df[['serial_no', 'account_holders_name', 'date', 'month_year', 'description', 'debit', 'credit', 'balance', 'classification']]

def process_statement(input_pdf, output_file, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None):
    """
    Process bank statement PDF and extract structured data.
    Uses text extraction as primary method, OCR as fallback.
    """
    print(f"Processing {input_pdf}")
    
    # use fitz
    pdf_text = extract_text_from_pdf(input_pdf)
    acc_no, acc_name = extract_account_info_from_text(pdf_text, pdf_path=input_pdf, bbox_acc_no=bbox_acc_no, poppler_bin=poppler_bin)
    print(f"Extracted from text - Account No: {acc_no}, Account Holder: {acc_name}")
    
    # or fallback to OCR
    if (not acc_no or not acc_name) and bbox_acc_no and bbox_acc_name:
        print("Falling back to OCR method")
        if not acc_no:
            acc_no = ocr_text_from_bbox(input_pdf, bbox_acc_no, poppler_bin)
        if not acc_name:
            acc_name = ocr_text_from_bbox(input_pdf, bbox_acc_name, poppler_bin)
        # print(f"OCR results - Account No: {acc_no}, Account Holder: {acc_name}")
    
    df_raw = extract_tables(input_pdf, poppler_bin)
    print(df_raw.tail())
    print(f"Extracted {len(df_raw)} raw rows from tables")
    
    df = normalize(df_raw)

    # Filter out rows with more than 2 missing values
    df = df[df.isna().sum(axis=1) <= 2]

    if acc_name:
        df['account_holders_name'] = acc_name
    if acc_no:
        df['acc_no'] = acc_no
    
    # save to Excel
    df.to_excel(output_file, index=False)
    print(f"Saved {len(df)} processed rows to {output_file}")
    
    return df, acc_no, acc_name

def process_folder(input_folder, output_folder, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None):
    """
    Process all PDFs in input_folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in {input_folder}")

    for pdf_file in pdf_files:
        input_pdf = os.path.join(input_folder, pdf_file)
        output_file = os.path.join(output_folder, os.path.splitext(pdf_file)[0] + ".xlsx")
        print(f"\n Processing {input_pdf}")
        try:
            df, account_number, account_holder = process_statement(input_pdf, output_file, bbox_acc_no, bbox_acc_name, poppler_bin)
            print(f"Processed {output_file}, ({len(df)}) rows")
        except Exception as e:
            print(f"Error processing {input_pdf}: {e}")
            continue

def longest_common_prefix(strings):
    if not strings:
        return ""
    s1 = min(strings)
    s2 = max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def combine_excels_if_similar(output_folder, threshold=80):
    """
    Combine Excel files in output_folder into a single Excel file if their filenames are at least
    `threshold` percent similar (fuzzy match). Do not save individuals if combined.
    """
    excel_files = [f for f in os.listdir(output_folder) if f.lower().endswith('.xlsx')]
    if not excel_files:
        print("No Excel files found to combine.")
        return

    groups = []
    used = set()
    for i, file1 in enumerate(excel_files):
        if file1 in used:
            continue
        group = [file1]
        used.add(file1)
        for file2 in excel_files[i+1:]:
            if file2 in used:
                continue
            score = fuzz.ratio(os.path.splitext(file1)[0], os.path.splitext(file2)[0])
            if score >= threshold:
                group.append(file2)
                used.add(file2)
        groups.append(group)

    for group in groups:
        if len(group) > 1:
            print(f"Combining files: {group}")
            dfs = []
            for fname in group:
                df = pd.read_excel(os.path.join(output_folder, fname))
                df['source_file'] = fname
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            # Remove individual files
            for fname in group:
                os.remove(os.path.join(output_folder, fname))
            # Name combined file based on common prefix or joined names
            base_names = [os.path.splitext(f)[0] for f in group]
            prefix = longest_common_prefix(base_names).rstrip("_- ")
            if not prefix or len(prefix) < 3:
                prefix = "_".join(base_names)
            combined_path = os.path.join(output_folder, f"{prefix}_combined.xlsx")
            combined_df.to_excel(combined_path, index=False)
            print(f"Combined Excel saved to {combined_path}")
        else:
            print(f"File {group[0]} has no similar files (>= {threshold}%) to combine. Keeping as is.")

# run locally
# if __name__ == "__main__":
#     input_folder = ""
#     output_folder = ""
#     poppler_bin = None

#     process_folder(input_folder, output_folder, poppler_bin)
#     combine_excels_if_similar(output_folder, threshold=80)      #threshold for fuzzy

# ocr run
# file_name = "kotak_parag"
# pdf_path = "data/old pdfs/kotak_parag.pdf"
# poppler_bin = r"D:\\Mridul.Intern\\poppler-23.05.0\\Library\\bin"

# df = extract_tables(pdf_path, poppler_bin=poppler_bin)
# if not df.empty:
#     df.to_csv("ocr_kotak_parag.csv", index=False)
#     print("Saved")
# else:
#     print("No data extracted.")