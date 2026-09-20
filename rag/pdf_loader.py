import re
import pymupdf

def clean_text(raw_text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\xa0', ' ')
    text = text.replace('\ufffd', "'")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_pdf(file_path):
    doc = pymupdf.open(file_path)
    pages_data = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text", sort=True)
        if not raw_text.strip():
            blocks = page.get_text("blocks", sort=True)
            block_texts = [b[4] for b in blocks if b[6] == 0]
            raw_text = "\n\n".join(block_texts)
        cleaned = clean_text(raw_text)
        if cleaned:
            pages_data.append({
                "page": page_num + 1,
                "text": cleaned
            })
    doc.close()
    return pages_data
