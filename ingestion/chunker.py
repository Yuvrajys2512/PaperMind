# ingestion/chunker.py

import tiktoken

def get_tokenizer():
    """
    Returns a tokenizer using the cl100k_base encoding.
    
    Why cl100k_base? It's the encoding used by most modern models
    including GPT-4 and the embedding models we'll use later.
    Using the same tokenizer as the embedding model means our
    token counts are accurate — no surprises when we hit model limits.
    
    Why not just count words? Because models think in tokens not words.
    "Uncharacteristically" is one word but multiple tokens.
    Counting words would give us inaccurate chunk sizes.
    """
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int, tokenizer) -> list:
    """
    Splits a text string into overlapping chunks of roughly chunk_size tokens.
    
    How it works:
    1. Tokenize the entire text into a list of token IDs (integers)
    2. Slide a window of chunk_size tokens across that list
    3. Each window becomes one chunk
    4. Each window starts (chunk_size - overlap) tokens after the previous one
    
    Why work with token IDs rather than words?
    Because the tokenizer splits at the token level, not the word level.
    Decoding back to text gives us clean readable strings.
    
    Args:
        text: the raw text to chunk
        chunk_size: number of tokens per chunk (256, 512, or 1024)
        overlap: number of tokens shared between consecutive chunks (50 or 100)
        tokenizer: the tiktoken tokenizer instance
    
    Returns:
        list of text strings, each roughly chunk_size tokens long
    """
    # Convert the entire text into a flat list of token IDs
    # e.g. "Hello world" → [15496, 995]
    tokens = tokenizer.encode(text)
    
    if len(tokens) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Take a slice of chunk_size tokens starting at 'start'
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # Decode the token IDs back into a readable string
        chunk_text_str = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text_str)
        
        # Move the window forward by (chunk_size - overlap) tokens
        # This means the next chunk starts overlap tokens before this one ended
        # creating the shared context we want between chunks
        step = chunk_size - overlap
        start += step
        
        # If the remaining tokens are less than half a chunk,
        # the last chunk already captured them with overlap — stop here
        if start >= len(tokens):
            break
    
    return chunks


def chunk_sections(sections: list, chunk_size: int, overlap: int) -> list:
    """
    Takes the list of labelled sections from the assembler and chunks each one.
    
    Why chunk section by section rather than the full text?
    Because chunking across section boundaries creates chunks that contain
    content from two different sections — half Introduction, half Methodology.
    That chunk would be retrieved for questions about either topic but would
    answer neither completely. Keeping sections separate means every chunk
    belongs clearly to one part of the paper.
    
    Returns a flat list of chunk dicts, each with full metadata attached.
    """
    tokenizer = get_tokenizer()
    all_chunks = []
    chunk_id = 1
    
    for section in sections:
        body = section["body"]
        
        if not body.strip():
            continue
        
        # Get the text chunks for this section
        text_chunks = chunk_text(body, chunk_size, overlap, tokenizer)
        
        for i, chunk_text_str in enumerate(text_chunks):
            token_count = len(tokenizer.encode(chunk_text_str))
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "section": section["heading"],
                "section_type": section["type"],
                "page_num": section["page_num"],
                "chunk_index": i,          # position within this section
                "total_chunks_in_section": len(text_chunks),
                "text": chunk_text_str,
                "token_count": token_count
            })
            chunk_id += 1
    
    return all_chunks