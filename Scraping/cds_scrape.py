"""
--------CDS Document Downloader and OCR Processor--------
This script downloads documents from the CERN Document Server,
and applies OCR using Nougat. 

Before running this script, ensure you have Nougat installed and configured.
module load CMake/3.21.1-GCCcore-11.2.0
uv pip install --only-binary=:all: "pyarrow==14.0.2"
uv pip install \
  "sentencepiece==0.1.99" \
  "nougat-ocr==0.1.17" \
  "pypdfium2==4.30.0" \
  PyPDF2 requests beautifulsoup4 \
  "transformers==4.35.0" \
  "tokenizers==0.14.1" \
  "huggingface-hub==0.17.3" \
  "albumentations==1.3.1" \
  "opencv-python-headless<5.0.0.0"
uv pip install --force-reinstall "numpy<2"
uv sync
"""


import numpy as np
import os
import subprocess
import requests
import PyPDF2
import argparse
import re
from datetime import datetime
import csv
from typing import Optional, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_URL = "https://cds.cern.ch/search" 

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
}


def get_atlas_papers_json(n_docs: int = 10, start_index: int = 1):
    """
    Queries the CDS for ATLAS papers and returns JSON metadata.
    Args:
        n_docs: number of records to return
        start_index: index of the first record to return

    Returns:
        A list of CDS record dictionaries.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8",
        "Referer": "https://cds.cern.ch/",
    }
    #query params for CDS search API
    params = {
        "cc": "ATLAS Papers",
        "of": "recjson",        #JSON output
        "rg": n_docs,           #number of records per page
        "jrec": start_index,    #starting record index
        "ln": "en",             #language
    }

    resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)

    #explicit troubleshoot check for 403 error: fixed with browser-like user agent
    if resp.status_code == 403:
        print("Got 403 Forbidden from CDS. The server is blocking this request.")
        resp.raise_for_status() #raise error if request fails

    resp.raise_for_status()

    return resp.json()

def get_effective_pages(pdf_path, pages):
    """
    Finds the final 'The ATLAS Collaboration' page in the PDF 
    and trims the page range so Nougat only processes up to that page.
    Scans the pdf from the end to find the last occurance of the phrase, 
    marking the beginning of that page as the end of the document to process through Nougat.
    Args:
        pdf_path: path to the PDF file
        pages: original page range or None for full doc
    Returns:
        A new page range string for --pages or None for full doc.
    """
    #1) Locating the ATLAS Collaboration section
    if isinstance(pages, str) and pages.lower() == "none":
        return None  #full doc
    try: #try reading PDF to count pages & extract text
        reader = PyPDF2.PdfReader(pdf_path)
        n_pages = len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return pages  #return original pages on error
    
    search_str = "the atlas collaboration"
    collaboration_page = None 
    
    for i in range(n_pages-1, -1, -1): # from last page backwards
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            continue
        
        if search_str in text.lower():
            collaboration_page = i + 1  #1-based page number
            break
    if collaboration_page is None or collaboration_page <= 1:
        return pages  #no change needed
    
    #2) Using collaboration_page to trim the pages
    #full doc pages=None
    trim_end = collaboration_page - 1  #last page to OCR process
    
    if pages is None:
        if trim_end < 1:
            return None  #full doc if trim_end is weird
        return f"1-{trim_end}"
    #complex multi-range pages with commas requested: not handled, return original
    if "," in pages:
        return pages
    #simple range pages start-end
    if "-" in pages:
        try:
            start_str, end_str = pages.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
        except ValueError:
            return pages  #invalid format: return original
        if end >= collaboration_page:
            new_end = trim_end
            if new_end < start:
                return pages
            return f"{start}-{new_end}"
        return pages  #no change needed if user end before author list
    #single page
    return pages  #no change needed

def find_missing_pages_in_nougat_output(nougat_output_path):
    """
    Scans a Nougat output file to find any missing page markers
    [MISSING_PAGE_FAIL:X], where X is the page number.
    Args:
        nougat_output_path: path to the Nougat output file
    Returns:
        List of page numbers that Nougat reported as missing.
    """
    try:
        with open(nougat_output_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading Nougat output {nougat_output_path}: {e}")
        return []

    pattern = re.compile(r"\[MISSING_PAGE_FAIL:(\d+)\]")
    matches = pattern.findall(text)
    #convert matches to ints, make unique & sort
    missing_pages = sorted({int(m) for m in matches})
    return missing_pages

def log_missing_pages(
    recid,
    pdf_path,
    output_path,
    effective_pages,
    missing_pages: List[int],
):
    """
    Appends missing page info into a CSV log in BASE_DIR to analyse whcih
    pages/types nougat tends to miss.
    CSV stored in BASE_DIR/nougat_missing_pages_log.csv
    Args:
        recid: CDS record id
        pdf_path: local path to paper
        output_path: path to nougat output file
        missing_pages: list of page numbers that nougat flagged as missing
    """
    if not missing_pages:
        return
    
    log_path = os.path.join(BASE_DIR, "nougat_missing_pages_log.csv")
    is_new = not os.path.exists(log_path) #if we need to write the header row
    
    try:
        #opening in append mode for multiple entries
        with open(log_path, "a", newline='', encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            #if its a new file, write in column headers
            if is_new:
                writer.writerow(
                    ["timestamp_utc", "recid", "pdf_name", "output_name",
                        "effective_pages", "missing_pages"]
                )
            #append one row for this record
            writer.writerow(
                [
                    datetime.utcnow().isoformat(), #keeps logs comparable across UK/CERN
                    recid,
                    os.path.basename(pdf_path),
                    os.path.basename(output_path),
                    effective_pages or "",  #store empty string (instead of None)
                    ";".join(str(p) for p in missing_pages), #store list as semicolon separated string
                ]
            )   
    except Exception as e:
        print(f"Error logging missing pages to {log_path}: {e}")


def run_nougat_on_pdf(
    pdf_path,
    out_dir,
    pages,
    preview_lines: int = 50,
):
    """
    Run Nougat OCR on 1 pdf.
    1)trim ATLAS Collaboration list
    2)make output dir
    3)build nougat CLI command & run via subprocess 
    4)print stdout/stderr for debugging
    5)return path to the ouput mmd file
    Args:
        pdf_path: input PDF file
        out_dir: directory to write nougat outputs
        preview_lines: not used in this function
    Returns:
        output_path: path to mmd or tex file, None on failure
        effective_pages: the final page range string we passed to nougat, None on failure
    """
    #trim pages
    effective_pages = get_effective_pages(pdf_path, pages)

    #ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)
    #basename for output files
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    #nougat CLI
    cmd = ["nougat"]
    if effective_pages is not None:
        cmd += ["--pages", effective_pages]
    cmd += ["-o", out_dir, pdf_path]

    #run nougat
    result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False #we check non-zero return code below
)

    #----NOUGAT TROUBLESHOOTING BLOCK---
    log_stdout_path = os.path.join(out_dir, f"{base_name}.nougat.stdout.log")
    log_stderr_path = os.path.join(out_dir, f"{base_name}.nougat.stderr.log")
    try:
        with open(log_stdout_path, "w", encoding="utf-8") as f:
            f.write(result.stdout or "")
        with open(log_stderr_path, "w", encoding="utf-8") as f:
            f.write(result.stderr or "")
    except Exception as e:
        print(f"Could not write nougat logs: {e}")

    #if Nougat fails:
    if result.returncode != 0:
        print(f"Nougat failed (returncode={result.returncode}) for: {pdf_path}")
        if result.stderr:
            print("Nougat stderr first 40 lines:")
            for i, line in enumerate(result.stderr.splitlines()):
                if i >= 40:
                    print("... see full logs")
                    break
                print(line)
        #paths to full log
        print(f"Full logs:\n  {log_stdout_path}\n  {log_stderr_path}")
        return None, effective_pages

    #----------------------------

    mmd_path = os.path.join(out_dir, base_name + ".mmd")
    tex_path = os.path.join(out_dir, base_name + ".tex")

    if os.path.exists(mmd_path):
        return mmd_path, effective_pages
    if os.path.exists(tex_path):
        return tex_path, effective_pages
    
    #if we cant find output files treat same as fail
    return None, effective_pages

def preview_text_file(path, n_lines: int = 50):
    """
    Previews the first n lines of a Nougat output file to check for errors.
    Strips whitespace and prints each line.
    Args:
        path: path to the Nougat output file
        n_lines: number of lines to preview
    """
    print("-------TEXT PREVIEW:-------")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                print(line.rstrip())
    except Exception as e:
        print(f"---ERROR READING {path}: {e}---")

def download_atlas_pdfs(n_docs, start_index, base_dir=BASE_DIR):
    """
    Downloads ATLAS papers from CDS and returns metadata for running Nougat later.
    Args:
        n_docs: number of documents to download
        start_index: starting index for document retrieval
        base_dir: base directory for downloads and outputs
    Returns:
        A list of dictionaries:
            "recid": <record id>,
            "title": <title>,
            "record_dir": <output folder>,
            "pdf_path": <local file path>
    """
    #ensure base dir exist
    os.makedirs(base_dir, exist_ok=True)

    #fetch metadata from cds
    records = get_atlas_papers_json(n_docs=n_docs, start_index=start_index)
    print(f"Found {len(records)} records.")
    
    downloaded = [] #list of downloaded recs

    for rec in records:
        #extract record info
        recid = rec.get("recid") or rec.get("id") or rec.get("control_number")
        title = rec.get("title", {}).get("title", "NO TITLE") #use no title to avoid crash
        print(f"Processing record {recid}: {title}")

        #find the PDF link: selects the first one it sees
        pdf_url = None
        for file in rec.get("files", []):
            if file.get("eformat") == ".pdf":
                pdf_url = file.get("url")
                break

        if not pdf_url:
            print(f"No PDF found for record {recid}, skipping.")
            continue

        #prepares folder and filenames
        record_dir = os.path.join(base_dir, f"CDS_Record_{recid}")
        os.makedirs(record_dir, exist_ok=True)

        #using URL basename as filename for easy identification and checking
        pdf_name = os.path.basename(pdf_url)
        pdf_path = os.path.join(record_dir, pdf_name)

        #Downloads the PDF
        try:
            print(f"Downloading PDF from {pdf_url}")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
        }
            r = requests.get(pdf_url, headers=headers, timeout=60)
            r.raise_for_status()
            #write pdf to disk
            with open(pdf_path, "wb") as file:
                file.write(r.content)
            print(f"Saved: {pdf_path}")
            #metadata for later ocr step below
            downloaded.append(
                {
                    "recid": recid,
                    "title": title,
                    "record_dir": record_dir,
                    "pdf_path": pdf_path,
                }
            )
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
            continue

    return downloaded

def run_ocr_on_downloaded_pdfs(
    downloaded_records,
    pages,
    preview_lines: int = 50,
):
    """
    Takes the list returned by download_atlas_pdfs and runs Nougat on all PDFs
    Args:
        downloaded_records: list of dicts
        pages: page range or None for full doc
        preview_lines: how many lines to show from each Nougat output to check for errors
    """
    for rec in downloaded_records:
        pdf_path = rec["pdf_path"]
        record_dir = rec["record_dir"]
        recid = rec.get("recid", "UNKNOWN")

        nougat_output, effective_pages = run_nougat_on_pdf(
            pdf_path,
            record_dir,
            pages=pages,
            preview_lines=preview_lines,
        )
        #skip to next if nougat fails
        if not nougat_output:
            print("Nougat produced no output for:", pdf_path)
            continue

        #preview to spot obviuos failures ---CURRENTLY NOT CALLED---
        #preview_text_file(nougat_output, n_lines=preview_lines)

        #scanning for missing pages
        missing_pages = find_missing_pages_in_nougat_output(nougat_output)
        #append row to csv log if missing pages
        log_missing_pages(
            recid,
            pdf_path,
            nougat_output,
            effective_pages,
            missing_pages,
        )

        
        
def main():
    """
    Defines arguments, downloads PDFs, then runs OCR
    """
    parser = argparse.ArgumentParser(description="Download ATLAS papers from CDS and run Nougat OCR")
    parser.add_argument("--n_docs", type=int, default=5, help="Number of documents to download")
    parser.add_argument("--start_index", type=int, default=1, help="Starting index for document retrieval")
    parser.add_argument("--pages", type=str, default=None, help="Page range for Nougat OCR (eg. '1-3') or None for full doc")
    parser.add_argument("--preview_lines", type=int, default=50, help="Number of lines to preview from Nougat output")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base directory for downloads and outputs")

    args = parser.parse_args()

    print(f"Using base directory: {args.base_dir}")
    
    #1)download PDFs
    downloaded_records = download_atlas_pdfs(n_docs=args.n_docs, start_index=args.start_index, base_dir=args.base_dir)
    #stop early if nothing downloaded
    if not downloaded_records:
        print("No documents downloaded!")
        return
    #2)run Nougat
    run_ocr_on_downloaded_pdfs(downloaded_records, pages=args.pages, preview_lines=args.preview_lines)


if __name__ == "__main__":
    main()
  
