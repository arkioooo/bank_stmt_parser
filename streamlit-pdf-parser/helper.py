# import pandas as pd
# import networkx as nx
# from fuzzywuzzy import fuzz
# from openpyxl import load_workbook
# import re

# # transaction types
# TRANSACTION_TYPES = {
#     'NEFT', 'RTGS', 'POS', 'ACH', 'IMPS', 'UPI',
#     'NACH', 'FT', 'DD', 'ECS', 'AEPS', 'SWIFT'
# }

# # pre-compile regex patterns
# ACCOUNT_NO_PATTERN = re.compile(r'([A-Z]{4,}\d{6,}\w*)')
# ALPHA_NUMERIC_WORD_PATTERN = re.compile(r'\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b')
# TRANSACTION_TYPE_PATTERN = re.compile(r'\b(?:' + '|'.join(TRANSACTION_TYPES) + r')\b', re.IGNORECASE)
# NON_ALPHA_SPACE_PATTERN = re.compile(r'[^A-Za-z\s]')
# MONTHS_REMOVE = re.compile(r'(?i)\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b')

# STOP_WORDS = {
#     'CR', 'DR', 'BY', 'TO', 'FROM', 'TRANSFER', 'PAYMENT',
#     'CREDIT', 'DEBIT', 'THROUGH', 'VIA', 'TRANSACTION', 'CHQ', 
#     'ADV', 'CHEQUE', 'DEPOSIT', 'OUTWARD', 'INWARD'
# }

# def should_process(text):
#     """Check if text contains any of the transaction types"""
#     if pd.isna(text) or not isinstance(text, str):
#         return False

#     return any(trans_type in text.upper() for trans_type in TRANSACTION_TYPES)

# def extract_after_account_no(text):
#     """
#     Detect account number and return everything after it.
#     """
#     match = ACCOUNT_NO_PATTERN.search(text)
#     if match:
#         return text[match.end():].strip()
#     return text

# def clean_text(text):
#     """Clean bank statement description text using regex"""
#     if not should_process(text):
#         return ""

#     # 1. Extract everything after the account number
#     processed_text = extract_after_account_no(text)

#     # 2. Remove any group that contains both letters and digits (alphanumeric words)
#     processed_text = ALPHA_NUMERIC_WORD_PATTERN.sub('', processed_text)

#     # 3. Remove transaction type keywords (case-insensitive, as whole words)
#     processed_text = TRANSACTION_TYPE_PATTERN.sub('', processed_text)

#     # 4. Remove abbreviated months and dates
#     processed_text = MONTHS_REMOVE.sub('', processed_text)

#     # 5. Remove special characters except alphabets and spaces, convert to uppercase, and collapse spaces
#     processed_text = NON_ALPHA_SPACE_PATTERN.sub(' ', processed_text)
#     processed_text = processed_text.upper()
#     processed_text = ' '.join(processed_text.split())

#     # 6. Remove common transaction-related words
#     words = [w for w in processed_text.split() if w not in STOP_WORDS]

#     return ' '.join(words)


# def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
#     """
#     Nodes are tuples: (acc_no, account_holders_name).
#     Edges carry amount, date, description.
#     """
#     G = nx.DiGraph()
#     for _, row in df.iterrows():
#         src = (
#             str(row.get('acc_no', '')).strip(),
#             str(row.get('account_holders_name', '')).strip()
#         )
#         # derive beneficiary_id similarly from description cleaning or NER if available
#         dst_name = clean_text(row.get('description', '')) or ''
#         # If you have a beneficiary name column, replace dst_name accordingly:
#         # dst_name = row.get('beneficiary_name', dst_name)
#         dst = (dst_name, dst_name)

#         amt  = float(row.get('debit', 0) or row.get('credit', 0) or 0)
#         date = pd.to_datetime(row['date'], dayfirst=True, errors='coerce').date()
#         desc = row.get('description', '')

#         G.add_edge(src, dst, amount=amt, date=date, description=desc)
#     return G

# def score_cycle_confidence(G: nx.DiGraph, cycle: list) -> int:
#     """
#     Confidence scoring:
#       - +30 if all amounts identical
#       - +20 if all acc_no exact-match around the cycle
#       - +50 if average name‐fuzzy similarity ≥90
#     Total max = 100
#     """
#     # Gather edge amounts and node identifiers
#     amounts, acc_nos, names = [], [], []
#     for u, v in zip(cycle, cycle[1:]+[cycle[0]]):
#         data = G.get_edge_data(u, v)
#         attr = next(iter(data.values())) if isinstance(data, dict) else data
#         amounts.append(attr['amount'])
#         acc_nos.append((u[0], v[0]))
#         names.append((u[1], v[1]))

#     # Amount consistency
#     amt_conf = 30 if len(set(amounts)) == 1 else 0

#     # Account‐number consistency: all pairs equal?
#     acc_conf = 20 if all(u==v for u,v in acc_nos) else 0

#     # Name fuzzy‐matching
#     sim_scores = [fuzz.token_sort_ratio(a, b) for a,b in names]
#     name_conf = 50 if (sum(sim_scores)/len(sim_scores)) >= 90 else 0

#     return amt_conf + acc_conf + name_conf

# def detect_circular_transactions(df: pd.DataFrame,
#                                  max_cycle_length: int = 6) -> pd.DataFrame:
#     G = build_transaction_graph(df)
#     results = []
#     for cycle in nx.simple_cycles(G):
#         if 2 < len(cycle) <= max_cycle_length:
#             conf = score_cycle_confidence(G, cycle)
#             u, v = cycle[0], cycle[1]
#             edge = G.get_edge_data(u, v)
#             attr = next(iter(edge.values())) if isinstance(edge, dict) else edge
#             # Flatten node identifiers for output
#             seq = [f"{no}/{nm}" for no,nm in cycle] + [f"{cycle[0][0]}/{cycle[0][1]}"]
#             results.append({
#                 'description':           attr['description'],
#                 'amount':                attr['amount'],
#                 'date':                  attr['date'],
#                 'beneficiary_sequence':  " → ".join(seq),
#                 'confidence':            conf
#             })
#     return pd.DataFrame(results)

# def append_circular_sheet(excel_path: str,
#                           circ_df: pd.DataFrame,
#                           sheet_name: str = 'circular_transactions'):
#     wb = load_workbook(excel_path)
#     if sheet_name in wb.sheetnames:
#         wb.remove(wb[sheet_name])
#     ws = wb.create_sheet(sheet_name)
#     # write headers
#     for ci, col in enumerate(circ_df.columns, start=1):
#         ws.cell(row=1, column=ci, value=col)
#     # write data
#     for ri, row in enumerate(circ_df.itertuples(index=False), start=2):
#         for ci, val in enumerate(row, start=1):
#             ws.cell(row=ri, column=ci, value=val)
#     wb.save(excel_path)

# # Usage Example
# if __name__ == '__main__':
#     excel_file = r'D:\Mridul.Intern\OneDrive - Vivriti Capital Private Limited\Desktop\proj\desc ner\retry\canara_aditya.xlsx'
#     df = pd.read_excel(excel_file)
#     circ_df = detect_circular_transactions(df, max_cycle_length=5)
#     append_circular_sheet(excel_file, circ_df)
#     print(f"Appended {len(circ_df)} circular transactions.")

# import re
# import pandas as pd
# import networkx as nx

# # Regex matching 4 letters + ≥7 digits (e.g. SBIN1234567)
# ACC_PATTERN = re.compile(r'\b([A-Z]{4}\d{7,})\b')

# def parse_two_accounts(desc: str):
#     """
#     Extract the first two account numbers from the description.
#     Returns (src_acc, dst_acc) or (None, None) if fewer than two found.
#     """
#     found = ACC_PATTERN.findall(desc or "")
#     return (found[0], found[1]) if len(found) >= 2 else (None, None)

# def detect_circular_transactions_csv(
#     input_csv: str,
#     output_csv: str,
#     acc_no_col: str = "acc_no",
#     max_cycle_length: int = 6
# ):
#     # 1. Load CSV
#     df = pd.read_csv(input_csv, dtype=str)
#     df['description'] = df['description'].fillna("")

#     # 2. Build directed graph using extracted account numbers
#     G = nx.DiGraph()
#     for _, row in df.iterrows():
#         # parse two accounts from description
#         src_acc, dst_acc = parse_two_accounts(row['description'])
#         if not src_acc or not dst_acc:
#             continue
#         desc = row['description']
#         G.add_edge(src_acc, dst_acc, description=desc)

#     # 3. Detect cycles
#     results = []
#     for cycle in nx.simple_cycles(G):
#         if 2 < len(cycle) <= max_cycle_length:
#             # first edge description
#             u, v = cycle[0], cycle[1]
#             desc = G[u][v][0]['description'] if isinstance(G[u][v], dict) else G[u][v]['description']
#             seq = " → ".join(cycle + [cycle[0]])
#             results.append({
#                 'description':          desc,
#                 'beneficiary_sequence': seq,
#                 'confidence':           100  # all-number match
#             })

#     # 4. Write output CSV
#     out_df = pd.DataFrame(results)
#     out_df.to_csv(output_csv, index=False)
#     print(f"Detected {len(results)} circular transactions → {output_csv}")


# detect_circular_transactions_csv(
#     input_csv        = 'data/desc/canara_desc.csv',
#     output_csv       = 'circular_transactions.csv',
#     acc_no_col       = 'acc_no',
#     max_cycle_length = 5
# )