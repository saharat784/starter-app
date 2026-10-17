"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re
from student import settings


def _extract_blocks(text: str) -> list[str]:
    """Parse text into logical blocks (preserving tables and code blocks as intact units)."""
    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    current_block: list[str] = []
    in_code_block = False
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Check code block fences
        if stripped.startswith("```"):
            if not in_code_block:
                if current_block and not in_table:
                    blocks.append("".join(current_block))
                    current_block = []
                in_code_block = True
                current_block.append(line)
            else:
                in_code_block = False
                current_block.append(line)
                blocks.append("".join(current_block))
                current_block = []
            continue

        if in_code_block:
            current_block.append(line)
            continue

        # Check table lines
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            current_block.append(line)
            continue
        elif in_table:
            in_table = False
            blocks.append("".join(current_block))
            current_block = []

        # Empty line separates normal paragraphs
        if not stripped:
            if current_block:
                blocks.append("".join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("".join(current_block))

    return [b.strip() for b in blocks if b.strip()]


def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    """Sliding window split for very long blocks that cannot be split by paragraphs."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += max(1, max_size - overlap)
    return chunks


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)
    """
    if not text or not text.strip():
        return [doc_id]

    chunk_size = getattr(settings, "CHUNK_SIZE", 900)
    chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 150)
    split_by_heading = getattr(settings, "SPLIT_BY_HEADING", True)
    include_heading = getattr(settings, "INCLUDE_HEADING", True)

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
    sections = []
    breadcrumbs = []

    matches = list(heading_pattern.finditer(text))

    if split_by_heading and matches:
        first_pos = matches[0].start()
        if first_pos > 0:
            preamble = text[:first_pos].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_title = match.group(2).strip()

            while len(breadcrumbs) >= level:
                breadcrumbs.pop()
            breadcrumbs.append(heading_title)
            current_path = " > ".join(breadcrumbs)

            start_content = match.end()
            end_content = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start_content:end_content].strip()

            sections.append((current_path, sec_text))
    else:
        sections.append(("", text.strip()))

    chunks: list[str] = []

    for path, sec_text in sections:
        header_prefix = f"# {path}\n\n" if (include_heading and path) else ""

        if len(header_prefix) + len(sec_text) <= chunk_size:
            chunk = (header_prefix + sec_text).strip()
            if chunk:
                chunks.append(chunk)
            continue

        blocks = _extract_blocks(sec_text)
        if not blocks:
            chunk = (header_prefix + sec_text).strip()
            if chunk:
                chunks.append(chunk)
            continue

        curr_pieces = []
        curr_len = len(header_prefix)

        for block in blocks:
            effective_max = max(100, chunk_size - len(header_prefix))
            if len(block) > effective_max:
                if curr_pieces:
                    combined = header_prefix + "\n\n".join(curr_pieces)
                    chunks.append(combined.strip())
                    curr_pieces = []
                    curr_len = len(header_prefix)
                sub_chunks = _split_long_text(block, effective_max, chunk_overlap)
                for sc in sub_chunks:
                    chunks.append((header_prefix + sc).strip())
                continue

            if curr_len + len(block) + 2 > chunk_size and curr_pieces:
                combined = header_prefix + "\n\n".join(curr_pieces)
                chunks.append(combined.strip())
                if chunk_overlap > 0 and len(curr_pieces[-1]) <= chunk_overlap:
                    curr_pieces = [curr_pieces[-1], block]
                    curr_len = len(header_prefix) + len(curr_pieces[0]) + len(block) + 2
                else:
                    curr_pieces = [block]
                    curr_len = len(header_prefix) + len(block) + 2
            else:
                curr_pieces.append(block)
                curr_len += len(block) + 2

        if curr_pieces:
            combined = header_prefix + "\n\n".join(curr_pieces)
            chunks.append(combined.strip())

    cleaned = [c for c in chunks if c.strip()]
    return cleaned if cleaned else [text.strip()]
