from app.rag.chunking import build_job_chunks, clean_text


def test_clean_and_semantic_sections():
    raw = "<h2>Job Responsibilities</h2><p>Build RAG pipelines.</p><h2>Requirements</h2><p>Strong Python skills.</p>"
    chunks = build_job_chunks(title="RAG Engineer", company="Example", location="Singapore", description=raw)
    assert clean_text(raw).startswith("Job Responsibilities")
    assert {chunk.section_type for chunk in chunks} == {"responsibilities", "required_qualifications"}
    assert all("Job: RAG Engineer" in chunk.content for chunk in chunks)


def test_long_content_is_split_under_limit_with_context():
    chunks = build_job_chunks(title="Engineer", company=None, location=None, description="Requirements\n\n" + "Python " * 1000)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 2000 for chunk in chunks)


def test_irrelevant_company_and_benefit_sections_are_filtered():
    description = (
        "About the Company\n\n"
        + "A very long company background that should not become retrieval context. " * 80
        + "\n\nResponsibilities\n\nBuild retrieval systems.\n\n"
        + "Required Qualifications\n\nPython and SQL.\n\n"
        + "Benefits\n\nFree snacks and events."
    )

    chunks = build_job_chunks(title="RAG Engineer", company="Example", location="Singapore", description=description)

    sections = {chunk.section_type for chunk in chunks}
    content = "\n".join(chunk.content for chunk in chunks)
    assert sections == {"responsibilities", "required_qualifications"}
    assert "company background" not in content.lower()
    assert "free snacks" not in content.lower()


def test_list_items_are_grouped_as_structured_paragraphs():
    description = "Required Qualifications\n\n- Python\n- RAG\n- Vector databases"

    chunks = build_job_chunks(title="RAG Engineer", company="Example", location="Singapore", description=description)

    assert len(chunks) == 1
    assert chunks[0].section_type == "required_qualifications"
    assert all(term in chunks[0].content for term in ("Python", "RAG", "Vector databases"))
