# README

This document provides a reference for all the key functions in the PDF parser pipeline. It explains each function’s purpose, inputs, outputs, and usage.

## Table of Contents

- [FORMAT & OCR HELPERS](#format--ocr-helpers)  
  - `is_page_scanned`  
  - `ocr_pdf_page`  
  - `ensure_page_searchable`  
- [HEADER NORMALISER](#header-normaliser)  
  - `map_headers`  
- [ACCOUNT INFO EXTRACTION](#account-info-extraction)  
  - `extract_text_from_pdf`  
  - `extract_account_info_from_text`  
- [OCR-BASED TABLE EXTRACTION](#ocr-based-table-extraction)  
  - `ocr_page_to_dataframe`  
  - `parse_ocr_text_to_dataframe`  
- [TABLE EXTRACTION PIPELINE](#table-extraction-pipeline)  
  - `extract_tables`  
- [BOUNDING-BOX OCR](#bounding-box-ocr)  
  - `ocr_text_from_bbox`  
- [TRANSACTION CLASSIFICATION](#transaction-classification)  
  - `classify_transaction`  
- [DATA NORMALISATION](#data-normalisation)  
  - `normalize`  
- [END-TO-END PIPELINE](#end-to-end-pipeline)  
  - `process_statement`  
  - `process_folder`  
- [UTILITY FUNCTIONS](#utility-functions)  
  - `longest_common_prefix`  
  - `combine_excels_if_similar`  

## FORMAT & OCR HELPERS

### `is_page_scanned(page) → bool`  
Detects whether a PDF page lacks a usable text layer.  
- **Input**:  
  - `page` (`fitz.Page`)  
- **Output**: `True` if text area is below threshold, else `False`.  
- **Usage**: Called per page to decide if OCR is needed.

### `ocr_pdf_page(src_pdf, page_num, dst_pdf, lang="eng")`  
Runs OCR on a single page of a PDF using OCRmyPDF.  
- **Inputs**:  
  - `src_pdf` (str): Path to source PDF  
  - `page_num` (int): Zero-based page index  
  - `dst_pdf` (str): Path to output PDF  
  - `lang` (str): OCR language code  
- **Output**: None (writes `dst_pdf`).  
- **Usage**: Converts one page to a searchable PDF.

### `ensure_page_searchable(pdf_path, page_num, cache_dir="__ocr_cache") → str`  
Returns a path to a searchable PDF for the specified page, OCR’ing only if needed.  
- **Inputs**:  
  - `pdf_path` (str)  
  - `page_num` (int)  
  - `cache_dir` (str): Directory for cached OCR pages  
- **Output**: Path to either the original PDF or a per-page OCRed PDF.  
- **Usage**: Ensures page can be text-extracted before table detection.

## HEADER NORMALISER

### `map_headers(df) → pd.DataFrame`  
Renames DataFrame columns to canonical names using fuzzy matching (FuzzyWuzzy).  
- **Input**:  
  - `df` (`pandas.DataFrame`) with raw headers  
- **Output**: DataFrame with renamed columns (`date`, `description`, `debit`, `credit`, `balance`).  
- **Usage**: Standardises diverse bank statement column labels.

## ACCOUNT INFO EXTRACTION

### `extract_text_from_pdf(pdf_path) → str`  
Extracts all text from a PDF.  
- **Input**: `pdf_path` (str)  
- **Output**: Combined text of every page.  
- **Usage**: Provides input for regex or vision model account info extraction.

### `extract_account_info_from_text(text, pdf_path=None, bbox_acc_no=None, poppler_bin=None, page_num=0) → (str, str)`  
Extracts account number and holder name via Pix2Struct or regex/OCR fallback.  
- **Inputs**:  
  - `text` (str): Full PDF text  
  - `pdf_path` (str, optional): For OCR fallback  
  - `bbox_acc_no` (tuple, optional): Bounding box for account number OCR  
  - `poppler_bin` (str, optional): Path to Poppler binaries  
  - `page_num` (int): Page index for OCR fallback  
- **Output**: Tuple `(account_number, account_holder)`.  
- **Usage**: Identifies account metadata from statements.

## OCR-BASED TABLE EXTRACTION

### `ocr_page_to_dataframe(pdf_path, page_num, poppler_bin=None) → pd.DataFrame`  
Uses Tesseract to OCR a page image and parse tabular data via text heuristics.  
- **Inputs**:  
  - `pdf_path` (str)  
  - `page_num` (int)  
  - `poppler_bin` (str, optional)  
- **Output**: DataFrame of parsed rows (columns `date`, `description`, `debit`, `credit`, `balance`).  
- **Usage**: Fallback when table detection fails.

### `parse_ocr_text_to_dataframe(text) → pd.DataFrame`  
Converts raw OCR text lines into structured rows.  
- **Input**: `text` (str)  
- **Output**: DataFrame as above.  
- **Usage**: Helper for `ocr_page_to_dataframe`.

## TABLE EXTRACTION PIPELINE

### `extract_tables(pdf_path, poppler_bin=None) → pd.DataFrame`  
Iterates pages, applies per-page OCR, extracts tables, and falls back to OCR parsing.  
- **Inputs**:  
  - `pdf_path` (str)  
  - `poppler_bin` (str, optional)  
- **Output**: Combined DataFrame of all pages.  
- **Usage**: Core table-harvesting function.

## BOUNDING-BOX OCR

### `ocr_text_from_bbox(pdf_path, bbox, poppler_bin=None, page_num=0) → str`  
Crops a PDF page to the specified bounding box and OCRs it.  
- **Inputs**:  
  - `pdf_path` (str)  
  - `bbox` (tuple): `(left, upper, right, lower)`  
  - `poppler_bin` (str, optional)  
  - `page_num` (int)  
- **Output**: Extracted text string.  
- **Usage**: Targeted OCR for account number or name.

## TRANSACTION CLASSIFICATION

### `classify_transaction(description) → str`  
Assigns a transaction category based on keywords.  
- **Input**: `description` (str)  
- **Output**: Category string (e.g., `IMPS`, `NEFT`, `Rent`, etc.).  
- **Usage**: Enriches rows with standard classification.

## DATA NORMALISATION

### `normalize(df) → pd.DataFrame`  
Cleans, deduplicates, fuzzy-maps headers, parses dates, converts amounts, and orders columns.  
- **Input**: Raw DataFrame from `extract_tables`.  
- **Output**: Final, cleaned DataFrame with columns:  
  ```
  serial_no | account_holders_name | date | month_year |
  description | debit | credit | balance | classification
  ```
- **Usage**: Prepares data for export.

## END-TO-END PIPELINE

### `process_statement(input_pdf, output_file, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None) → (pd.DataFrame, str, str)`  
Executes the full pipeline on a single PDF.  
- **Inputs**:  
  - `input_pdf` (str)  
  - `output_file` (str): Path for Excel output  
  - `bbox_acc_no`, `bbox_acc_name` (tuple, optional)  
  - `poppler_bin` (str, optional)  
- **Outputs**:  
  - Cleaned DataFrame  
  - `account_number` (str)  
  - `account_holder` (str)  
- **Usage**: Main entry point for processing one statement.

### `process_folder(input_folder, output_folder, bbox_acc_no=None, bbox_acc_name=None, poppler_bin=None)`  
Processes all PDFs in a folder and writes individual Excel files.  
- **Inputs**:  
  - `input_folder`, `output_folder` (str)  
  - Other optional args as above  
- **Output**: None (writes files).  
- **Usage**: Batch-processing of a directory.

## UTILITY FUNCTIONS

### `longest_common_prefix(strings) → str`  
Computes the longest common prefix among a list of strings.  
- **Input**: List of strings  
- **Output**: Common prefix string  
- **Usage**: Naming combined Excel files.

### `combine_excels_if_similar(output_folder, threshold=80)`  
Merges Excel files whose filenames exceed a similarity threshold.  
- **Inputs**:  
  - `output_folder` (str)  
  - `threshold` (int, 0–100)  
- **Output**: Writes combined Excel files, removes originals.  
- **Usage**: Post-processing to consolidate similar reports.

**End of Reference**