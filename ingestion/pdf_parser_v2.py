import pdfplumber
import re
import os

def process_words_to_text(words: list) -> str:
    """
    Groups words into lines based on their vertical position.
    Derived from the character sorting logic in the original parser.
    """
    if not words:
        return ""
    
    # Sort words: top to bottom, then left to right
    words.sort(key=lambda w: (round(w['top']), w['x0']))
    
    lines = []
    current_line = [words[0]['text']]
    last_top = round(words[0]['top'])
    
    for i in range(1, len(words)):
        w = words[i]
        # If the vertical position is nearly the same, it's the same line
        if abs(round(w['top']) - last_top) <= 2: 
            current_line.append(w['text'])
        else: 
            lines.append(" ".join(current_line))
            current_line = [w['text']]
            last_top = round(w['top'])
            
    lines.append(" ".join(current_line))
    return "\n".join(lines)

def extract_text_v2(pdf_path: str) -> str:
    """
    Analyzes page layout to decide between 1-column or 2-column processing.
    """
    full_pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 1. Extract words with bounding boxes
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                continue

            # 2. Gutter Detection: Check if words cross the center of the page
            mid_x = page.width / 2
            # Words that span across the middle (likely 1-column headers or titles)
            crossing_words = [w for w in words if w['x0'] < mid_x - 10 and w['x1'] > mid_x + 10]
            
            # If very few words cross the middle, treat as 2-column layout
            is_two_column = len(crossing_words) < (len(words) * 0.05)

            if not is_two_column:
                # Process as standard single column
                page_text = process_words_to_text(words)
            else:
                # Split words into left and right buckets based on center x-coordinate
                left_col = [w for w in words if w['x1'] <= mid_x + 5]
                right_col = [w for w in words if w['x0'] >= mid_x - 5]
                
                # Combine left column text then right column text to preserve reading order
                page_text = process_words_to_text(left_col) + "\n" + process_words_to_text(right_col)
            
            full_pages_text.append(page_text)

    return "\n\n".join(full_pages_text)

def remove_credits_block(full_text: str) -> str:
    """Finds 'Abstract' and cuts off everything before it."""
    match = re.search(r'\nAbstract\n', full_text, re.IGNORECASE)
    if match:
        return full_text[match.start():].strip()
    return full_text

def remove_references_section(full_text: str) -> str:
    """Finds 'References' and cuts off everything after it."""
    match = re.search(r'\nReferences\b', full_text, re.IGNORECASE)
    if match:
        return full_text[:match.start()].strip()
    return full_text

if __name__ == "__main__":
    # 1. Filenames exactly as they appear in your data folder
    papers = [
        "Attention is all you need.pdf",
        "BERT paper.pdf" 
    ]
    
    # 2. Path correction: They are in 'data', not 'data/papers'
    input_folder = "data"
    output_folder = "data"

    for paper_name in papers:
        input_path = os.path.join(input_folder, paper_name)
        output_filename = paper_name.replace(".pdf", ".txt")
        output_path = os.path.join(output_folder, output_filename)

        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            continue

        print(f"🚀 Processing: {paper_name}...")
        
        try:
            # Run the extraction logic
            raw_text = extract_text_v2(input_path)
            
            # Clean the noise using your helper functions
            cleaned = remove_credits_block(raw_text)
            final_text = remove_references_section(cleaned)
            
            # Save to the data folder
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
                
            print(f"✅ Success! Created: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to process {paper_name}: {e}")