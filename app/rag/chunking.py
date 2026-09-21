from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


SECTION_PATTERNS = {
    "overview": re.compile(
        r"role overview|job overview|job summary|position summary|about the role|the role|overview|summary|职位概述|岗位简介|职位简介",
        re.I,
    ),
    "responsibilities": re.compile(r"responsibilit|what you(?:'|’)ll do|job duties|岗位职责|工作职责", re.I),
    "required_qualifications": re.compile(
        r"requirement|qualification|what we(?:'|’)re looking|must have|who you are|what you bring|basic qualification|minimum qualification|任职要求|岗位要求|任职资格",
        re.I,
    ),
    "preferred_qualifications": re.compile(r"nice to have|preferred|bonus|加分|优先", re.I),
    "skills": re.compile(r"skills|technical stack|technologies|tech stack|技能|技术栈|技术能力", re.I),
    "experience": re.compile(r"experience|professional background|工作经验|经验要求", re.I),
    "education": re.compile(r"education|academic|degree|学历|教育背景", re.I),
    "projects_research": re.compile(r"projects?|research|研究|项目经验", re.I),
    "work_arrangement": re.compile(r"location|work arrangement|remote|hybrid|visa|relocation|地点|远程|签证", re.I),
    "benefits": re.compile(r"benefit|what we offer|perks|福利|待遇", re.I),
    "company": re.compile(r"about (?:us|the company)|company description|our mission|our values|公司介绍|公司背景|使命|价值观", re.I),
    "compensation": re.compile(r"compensation|salary|薪资|薪酬", re.I),
    "application": re.compile(r"how to apply|application process|申请方式|招聘流程", re.I),
    "legal": re.compile(r"equal opportunity|eeo|diversity|inclusion|法律声明|平等就业", re.I),
}

RELEVANT_SECTION_TYPES = frozenset(
    {
        "overview",
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
        "skills",
        "experience",
        "education",
        "projects_research",
        "work_arrangement",
        "general",
    }
)
IGNORED_SECTION_TYPES = frozenset({"company", "benefits", "compensation", "application", "legal"})


@dataclass(frozen=True)
class ChunkData:
    section_type: str
    content: str
    content_hash: str
    token_count: int


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_for_heading(text: str) -> str | None:
    if text.strip().endswith((".", "!", "?", "。", "！", "？")):
        return None
    compact = re.sub(r"[^\w\s'’]", "", text).strip()
    if not compact or len(compact) > 90:
        return None
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.search(compact):
            return section
    return None


def split_sections(text: str, *, filter_irrelevant: bool = True) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("general", [])]
    current_paragraph: list[str] = []

    def flush_paragraph() -> None:
        if current_paragraph:
            sections[-1][1].append("\n".join(current_paragraph))
            current_paragraph.clear()

    for raw_line in text.splitlines():
        raw = raw_line.strip()
        is_list_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", raw))
        line = raw.strip(" #*-•")
        if not line:
            flush_paragraph()
            continue
        heading_section = section_for_heading(line)
        if heading_section:
            flush_paragraph()
            sections.append((heading_section, []))
            continue
        if is_list_item:
            flush_paragraph()
        current_paragraph.append(line)
        if is_list_item:
            flush_paragraph()
    flush_paragraph()
    parsed = [(section, "\n\n".join(parts)) for section, parts in sections if parts]
    if not filter_irrelevant:
        return parsed

    has_relevant_heading = any(section in RELEVANT_SECTION_TYPES - {"general"} for section, _ in parsed)
    filtered: list[tuple[str, str]] = []
    for section, body in parsed:
        if section in IGNORED_SECTION_TYPES:
            continue
        # Keep unheaded text as a fallback. If explicit sections exist, a
        # very long leading block is more likely to be company background.
        if section == "general" and has_relevant_heading and len(body) > 1200:
            continue
        filtered.append((section, body))
    return filtered


def split_long_text(text: str, max_chars: int = 1800, overlap_chars: int = 180) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars])
                start += max_chars - overlap_chars
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:].lstrip()
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_job_chunks(
    *,
    title: str,
    company: str | None,
    location: str | None,
    description: str | None,
    job_type: str | None = None,
    is_remote: bool | None = None,
) -> list[ChunkData]:
    text = clean_text(description)
    if not text:
        return []
    work_arrangement = (
        "remote"
        if is_remote is True
        else "onsite or unspecified"
        if is_remote is False
        else "unspecified"
    )
    context = (
        f"Job: {title}\nCompany: {company or 'Unknown'}\n"
        f"Location: {location or 'Unknown'}\n"
        f"Job type: {job_type or 'Unspecified'}\n"
        f"Work arrangement: {work_arrangement}"
    )
    output: list[ChunkData] = []
    for section, section_text in split_sections(text, filter_irrelevant=True):
        for body in split_long_text(section_text):
            content = f"{context}\nSection: {section}\n\n{body}".strip()
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            output.append(ChunkData(section, content, digest, max(1, len(content) // 4)))
    return output
