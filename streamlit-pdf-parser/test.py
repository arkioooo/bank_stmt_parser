import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re
import os
import subprocess
from pathlib import Path
from fuzzywuzzy import fuzz, process
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

# Classifications dict
CLASSIFICATIONS = {
    'ECS/NACH': ['ecs', 'nach', 'ach', 'ift'],
    'IMPS': ['imps'],
    'NEFT': ['neft'],
    'RTGS': ['rtgs'],
    'ESIC': ['esic'],
    'Interest': ['interest'],
    'Refund/Reversal': ['refund', 'reversal', 'rev'],
    'Salary': ['salary', 'stipend'],
    'Tax': ['tax', 'duty', 'customs'],
    'Credit Card': ['credit card', 'cc'],
    'Debit Card': ['debit card', 'dc'],
    'Bank Instrument': ['dd', 'commissioner'],
    'Cash Txn': ['cash', 'withdrawal', 'deposit'],
    'Cheque Txn': ['cheque', 'chq', 'clearing', 'clg'],
    'Company Expense': ['expense', 'business', 'corporate', 'travel',
                       'home', 'charges', 'employee', 'fund', 'reimbursement',
                       'renumeration', 'leave'],
    'Forex': ['forex', 'brn'],
    'Insurance': ['insurance', 'premium'],
    'Rent': ['rent'],
    'UPI': ['upi'],
    'Other': []
}

desc_list = ['description', 'remark', 'transaction', 'detail', 'particulars']

# ---------- FORMAT & OCR HELPERS ---------- #
TEXT_RATIO_TH = 0.01  # <1 % text → call it scanned

def is_page_scanned(page) -> bool:
    """Return True if page lacks a usable text layer."""
    text_area = sum(abs(fitz.Rect(b[:4])) for b in page.get_text("blocks"))
    return (text_area / abs(page.rect)) < TEXT_RATIO_TH

def ocr_pdf_page(src_pdf: str, page_num: int, dst_pdf: str, lang="eng"):
    """OCR a single page (0-based index) into dst_pdf in-place."""
    cmd = [
        "ocrmypdf", "--skip-text", "--pages", f"{page_num+1}",
        "--output-type", "pdf", "-l", lang, src_pdf, dst_pdf
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def ensure_page_searchable(pdf_path: str, page_num: int, cache_dir="__ocr_cache") -> str:
    """
    If the requested page lacks text, OCR that page only and return a
    temporary searchable-PDF path; otherwise return original path.
    """
    doc = fitz.open(pdf_path)
    needs_ocr = is_page_scanned(doc[page_num])
    doc.close()
    if not needs_ocr:
        return pdf_path  # already searchable

    Path(cache_dir).mkdir(exist_ok=True)
    ocr_path = Path(cache_dir) / f"{Path(pdf_path).stem}_p{page_num}.pdf"
    if not ocr_path.exists():
        ocr_pdf_page(pdf_path, page_num, str(ocr_path))
    return str(ocr_path)

# ---------- HEADER NORMALISER ---------- #
HEADER_CANON = {
    "date": ["date", "value date", "txn date", "transaction date"],
    "description": ["description", "narration", "particulars", "remarks", "detail", "transaction"],
    "debit": ["debit", "withdrawal", "dr"],
    "credit": ["credit", "deposit", "cr"],
    "balance": ["balance", "running balance", "bal", "closing balance"],
}

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using fuzzy match with fuzzywuzzy."""
    ren = {}
    all_aliases = sum(HEADER_CANON.values(), [])
    
    for col in df.columns:
        # Use fuzzywuzzy.process.extractOne - returns (match, score) tuple
        result = process.extractOne(
            col.lower(), 
            all_aliases,
            scorer=fuzz.partial_ratio
        )
        
        if result and result[1] >= 80:  # result[1] is the score
            best_match = result[0]
            # Find which canonical category this belongs to
            for canon, aliases in HEADER_CANON.items():
                if best_match in aliases:
                    ren[col] = canon
                    break
    
    return df.rename(columns=ren)

def extract_account_info_from_text(text, pdf_path=None, bbox_acc_no=None, poppler_bin=None, page_num=0):
    """
    Extract account number and account holder name from PDF text content.
    Try Pix2Struct first, then fallback to regex/OCR.
    """
    account_number = None
    account_holder = None
    
    # pix2struct try for acc no and name
    # try:
    #     poppler_bin = r"D:\\Mridul.Intern\\poppler-23.05.0\\Library\\bin"
    #     model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-large")
    #     processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-large")
    #     pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_bin)
    #     page = pages[0].convert("RGB")
    #     width, height = page.size
    #     top_half = page.crop((0, 0, width, height // 2))
        
    #     prompt_holder = "Get the name of the account holder. It does not end with 'bank', 'finance' and the like. If not found, return 'Not found'."
    #     inputs_holder = processor(images=top_half, text=prompt_holder, return_tensors="pt")
    #     pred_holder = model.generate(**inputs_holder, max_new_tokens=64)
    #     answer_holder = processor.batch_decode(pred_holder, skip_special_tokens=True)[0]
        
    #     if answer_holder and "not found" not in answer_holder.lower():
    #         account_holder = answer_holder.strip()
        
    #     prompt_accno = "Get the account number which is usually a long string of integers. If not found, return 'Not found'."
    #     inputs_accno = processor(images=top_half, text=prompt_accno, return_tensors="pt")
    #     pred_accno = model.generate(**inputs_accno, max_new_tokens=64)
    #     answer_accno = processor.batch_decode(pred_accno, skip_special_tokens=True)[0]
        
    #     if answer_accno and "not found" not in answer_accno.lower():
    #         account_number = answer_accno.strip()
            
    # except Exception as e:
    #     print(f"Pix2Struct extraction failed: {e}")
    
    if not account_number or not account_holder:
        # regex for acc no (allowing for alphanumeric, e.g. SBIN0001234567)
        acc_no_patterns = [
            r'Account\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
            r'A/c\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
            r'Account\s*Number\.?\s*[:\-]?\s*([A-Z0-9]{6,})',
            r'Acc\s*No\.?\s*[:\-]?\s*([A-Z0-9]{6,})'
        ]
        
        if not account_number:
            for pattern in acc_no_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    account_number = match.group(1)
                    break
        
        # OCR fallback if not found
        if not account_number and pdf_path and bbox_acc_no:
            ocr_text = ocr_text_from_bbox(pdf_path, bbox_acc_no, poppler_bin, page_num)
            for pattern in acc_no_patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    account_number = match.group(1)
                    break
            
            if not account_number:
                generic_match = re.search(r'\b[A-Z0-9]{6,}\b', ocr_text)
                if generic_match:
                    account_number = generic_match.group(0)
        
        if not account_holder:
            lines = text.split('\n')
            for i, line in enumerate(lines[:30]):
                line = line.strip()
                if not line:
                    continue
                
                lcline = line.lower()
                if any(skip in lcline for skip in ['statement', 'account', 'period', 'bank', 'branch', 'customer', 'number', 'date', 'summary', 'address']):
                    continue
                
                if any(char.isdigit() for char in line):
                    continue
                
                if re.search(r'[^a-zA-Z\s\.]', line):
                    continue
                
                words = line.split()
                if 1 < len(words) <= 4 and all(w[0].isupper() or w.isupper() for w in words):
                    account_holder = line
                    break
            
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
            if len(parts) >= 4:  # min parts in transaction
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
    Iterate pages; OCR where required; harvest tables; fallback to OCR text.
    Returns combined DataFrame (may be empty).
    """
    all_dfs = []
    orig_doc = fitz.open(pdf_path)

    for pnum in range(len(orig_doc)):
        # get a searchable path for THIS page
        page_pdf = ensure_page_searchable(pdf_path, pnum)
        doc = fitz.open(page_pdf)
        page = doc[0] if len(doc) == 1 else doc[pnum]

        # 1. table extraction
        tables = page.find_tables()
        got_data = False
        for tbl in tables:
            try:
                df = tbl.to_pandas().dropna(how="all")
                if not df.empty:
                    all_dfs.append(df)
                    got_data = True
            except Exception:
                continue

        # 2. OCR fallback if no table rows
        if not got_data:
            ocr_df = ocr_page_to_dataframe(page_pdf, pnum, poppler_bin)
            if not ocr_df.empty:
                all_dfs.append(ocr_df)

        doc.close()

    orig_doc.close()
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def ocr_text_from_bbox(pdf_path, bbox, poppler_bin=None, page_num=0):
    """
    Crop the page image using bbox and perform OCR on cropped image.
    bbox format: (left, upper, right, lower)
    """
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_bin)
        if page_num >= len(images):
            return ""
        
        img = images[page_num].crop(bbox)
        
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

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean the extracted dataframe with fuzzy header mapping.
    Deduplicates columns, maps headers, selects a single date column safely,
    parses dates, normalizes amounts, classifies transactions, and orders columns.
    """
    # 1. Drop duplicate‐named columns outright
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. If empty, return an empty DataFrame with the expected schema
    if df.empty:
        cols = [
            'serial_no',
            'account_holders_name',
            'date',
            'month_year',
            'description',
            'debit',
            'credit',
            'balance',
            'classification'
        ]
        return pd.DataFrame(columns=cols)

    # 3. Normalize column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r'\s+', '_', regex=True)
          .str.replace(r'\.', '', regex=True)
    )

    # 4. Fuzzy‐map headers
    df = map_headers(df)
    df = df.loc[:, ~df.columns.duplicated()]

    # 5. Ensure a description column exists
    if 'description' not in df.columns:
        fallback = next((c for c in desc_list if c in df.columns), None)
        if fallback:
            df.rename(columns={fallback: 'description'}, inplace=True)
        else:
            df['description'] = pd.NA

    # 6. Identify exactly one source_date column
    if 'date' in df.columns:
        source_date = 'date'
    else:
        # pick a single other "*date*" column if present
        others = [c for c in df.columns if 'date' in c and c != 'date']
        source_date = others[0] if len(others) == 1 else None

    # 7. Drop any additional date‐like columns to avoid ambiguity
    date_cols = [c for c in df.columns if 'date' in c]
    if source_date and len(date_cols) > 1:
        for c in date_cols:
            if c != source_date:
                df.drop(columns=c, inplace=True)
    
    # 8. Safely convert that single Series to datetime
    if source_date:
        # print("Columns at conversion time:", df.columns.tolist())
        # print("source_date =", repr(source_date))
        dt = pd.to_datetime(df[source_date], dayfirst=True, errors='coerce')
        df['date']       = dt.dt.strftime('%d-%m-%Y')
        df['month_year'] = dt.dt.strftime('%m-%Y')
    else:
        df['date']       = None
        df['month_year'] = None

    # 9. Assign serial numbers
    df['serial_no'] = range(1, len(df) + 1)

    # 10. Normalize numeric columns
    for col in ['debit', 'credit', 'balance']:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(',', '', regex=False)
                       .replace({'nan': None, 'NaN': None, '': None})
                       .apply(lambda x: pd.to_numeric(x, errors='coerce') if x is not None else pd.NA)
            )
        else:
            df[col] = pd.NA

    # 11. Classify transactions
    df['classification'] = df['description'].apply(classify_transaction)
    df['account_holders_name'] = None

    # 12. Return in final column order
    return df[
        [
            'serial_no',
            'account_holders_name',
            'date',
            'month_year',
            'description',
            'debit',
            'credit',
            'balance',
            'classification'
        ]
    ]

def process_statement(input_pdf, output_file, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None):
    """
    End-to-end pipeline: page-aware OCR, table extraction, header normalisation using fuzzywuzzy.
    """
    # print(f"Processing {input_pdf}")
    
    # Extract text for account info
    pdf_text = extract_text_from_pdf(input_pdf)
    acc_no, acc_name = extract_account_info_from_text(
        pdf_text, pdf_path=input_pdf, bbox_acc_no=bbox_acc_no,
        poppler_bin=poppler_bin)
    
    print(f"Extracted from text - Account No: {acc_no}, Account Holder: {acc_name}")
    
    # Fallback to OCR if needed
    if (not acc_no or not acc_name) and bbox_acc_no and bbox_acc_name:
        print("Falling back to OCR method")
        if not acc_no:
            acc_no = ocr_text_from_bbox(input_pdf, bbox_acc_no, poppler_bin)
        if not acc_name:
            acc_name = ocr_text_from_bbox(input_pdf, bbox_acc_name, poppler_bin)
        # print(f"OCR results - Account No: {acc_no}, Account Holder: {acc_name}")
    
    # ----- TABLE HARVEST ----- #
    df_raw = extract_tables(input_pdf, poppler_bin)
    
    # print(f"Extracted {len(df_raw)} raw rows from tables")
    
    # ----- HEADER / TYPE CLEANING ----- #
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df = normalize(df_raw)
    
    # Filter out rows with more than 2 missing values
    df = df[df.isna().sum(axis=1) <= 2]
    
    # Insert account metadata
    if acc_name:
        df['account_holders_name'] = acc_name
    if acc_no:
        df['acc_no'] = acc_no
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    # print(f"Saved {len(df)} processed rows to {output_file}")
    
    return df, acc_no, acc_name

def process_folder(input_folder, output_folder, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None):
    """
    Process all PDFs in input_folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    # print(f"Found {len(pdf_files)} PDF files in {input_folder}")
    
    for pdf_file in pdf_files:
        input_pdf = os.path.join(input_folder, pdf_file)
        output_file = os.path.join(output_folder, os.path.splitext(pdf_file)[0] + ".xlsx")
        
        # print(f"\n Processing {input_pdf}")
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
        # print("No Excel files found to combine.")
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
            # print(f"Combining files: {group}")
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
        # else:
        #     print(f"File {group[0]} has no similar files (>= {threshold}%) to combine. Keeping as is.")

# Main execution
# if __name__ == "__main__":
#     input_folder = "data/pdf"
#     output_folder = "retry"
#     poppler_bin = None
    
#     process_folder(input_folder, output_folder, poppler_bin)
#     combine_excels_if_similar(output_folder, threshold=80)  # threshold for fuzzy
    
    # OCR test run (uncomment to test specific files)
    # file_name = "kotak_parag"
    # pdf_path = "data/old pdfs/kotak_parag.pdf"
    # poppler_bin = r"D:\\Mridul.Intern\\poppler-23.05.0\\Library\\bin"
    # df = extract_tables(pdf_path, poppler_bin=poppler_bin)
    # if not df.empty:
    #     df.to_csv("ocr_kotak_parag.csv", index=False)
    #     print("Saved")
    # else:
    #     print("No data extracted.")
