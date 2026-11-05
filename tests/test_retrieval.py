import pytest
from src.document_ingestion import extract_text,chunk_text
from src.retriever import LocalRetriever
def test_txt_parse_chunks_and_retrieval(tmp_path):
    text=extract_text("guide.txt",b"Do not share your OTP. Verify the employer independently.")
    assert "OTP" in text and chunk_text(text)
    r=LocalRetriever(tmp_path); r.add_document("guide.txt",b"Never share an OTP or password with a recruiter.")
    hits=r.search("OTP password",3); assert hits and any(h.get("filename")=="guide.txt" for h in hits)
def test_malformed_and_missing_kb(tmp_path):
    r=LocalRetriever(tmp_path); assert r.search("unrelated query",3)==[]
    with pytest.raises(ValueError): extract_text("bad.docx",b"x")
    with pytest.raises(ValueError): extract_text("bad.txt",b"x"*1024,max_bytes=10)
def test_empty_txt_is_not_indexed(tmp_path):
    r=LocalRetriever(tmp_path)
    with pytest.raises(ValueError): r.add_document("empty.txt",b" ")

def test_rebuild_removes_stale_extracted_content(tmp_path):
    r=LocalRetriever(tmp_path)
    r.add_document("retired.txt",b"Zorbaxultra phrase from a document that will be removed.")
    saved=tmp_path/"retired.txt"
    assert any(row.get("filename")=="retired.txt" for row in r.rows)
    saved.unlink()
    rows=r.rebuild()
    assert all(row.get("filename")!="retired.txt" for row in rows)
    assert all("Zorbaxultra" not in row["excerpt"] for row in rows)
    reloaded=LocalRetriever(tmp_path)
    assert all("Zorbaxultra" not in row["excerpt"] for row in reloaded.rows)
    assert reloaded.search("Zorbaxultra",4)==[]

def test_delete_document_removes_source_and_persisted_index_reference(tmp_path):
    r=LocalRetriever(tmp_path)
    r.add_document("remove-me.md",b"Quenchedragon phrase should disappear after deletion.")
    assert r.delete_document("remove-me.md") is True
    assert not (tmp_path/"remove-me.md").exists()
    assert all(row.get("filename")!="remove-me.md" for row in r.rows)
    assert LocalRetriever(tmp_path).search("Quenchedragon",4)==[]
    assert LocalRetriever(tmp_path).delete_document("remove-me.md") is False

def test_document_deletion_rejects_path_traversal(tmp_path):
    r=LocalRetriever(tmp_path)
    with pytest.raises(ValueError): r.delete_document("../outside.txt")

def test_startup_detects_removed_source_and_rebuilds_stale_index(tmp_path):
    r=LocalRetriever(tmp_path)
    r.add_document("gone.txt",b"Velvetquartz evidence existed only in this temporary document.")
    (tmp_path/"gone.txt").unlink()
    reloaded=LocalRetriever(tmp_path)
    assert all(row.get("filename")!="gone.txt" for row in reloaded.rows)
    assert reloaded.search("Velvetquartz",4)==[]
    assert "knowledge documents changed" in reloaded.diagnostic
