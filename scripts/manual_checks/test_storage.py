from api.storage import create_paper_record, update_paper_status, get_paper, list_papers

# Create a record
paper_id = create_paper_record("attention_is_all_you_need.pdf")
print(f"Created: {paper_id}")

# Read it back
record = get_paper(paper_id)
print(f"Status: {record['status']}")  # should be 'processing'

# Mark it ready
update_paper_status(paper_id, "ready")
record = get_paper(paper_id)
print(f"Status after update: {record['status']}")  # should be 'ready'

# List all
all_papers = list_papers()
print(f"Total papers: {len(all_papers)}")

print("All good!")