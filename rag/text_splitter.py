from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(pages_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if isinstance(pages_data, str):
        pages_data = [{"page": 1, "text": pages_data}]
    
    chunks = []
    chunk_counter = 1
    
    for item in pages_data:
        text = item["text"].strip()
        page = item.get("page", 1)
        if not text:
            continue
            
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunk_content = text[start:].strip()
                if chunk_content:
                    chunks.append({
                        "chunk_id": chunk_counter,
                        "text": chunk_content,
                        "page": page
                    })
                    chunk_counter += 1
                break
            
            break_point = text.rfind(" ", start, end)
            if break_point == -1 or break_point <= start:
                break_point = end
            
            chunk_content = text[start:break_point].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "text": chunk_content,
                    "page": page
                })
                chunk_counter += 1
            
            start = max(start + 1, break_point - chunk_overlap)
            next_space = text.find(" ", start)
            if next_space != -1 and next_space < break_point:
                start = next_space + 1
                
    return chunks
