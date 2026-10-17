"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
from student import settings  # noqa: F401 — MAX_CONTEXT_CHARS


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message
    """
    context_blocks = []
    total_chars = 0
    max_chars = getattr(settings, "MAX_CONTEXT_CHARS", 6000)

    if hits:
        for chunk, score in hits:
            doc_id = getattr(chunk, "doc_id", "doc")
            chunk_text = getattr(chunk, "text", str(chunk)).strip()
            block = f"--- เอกสารอ้างอิง ({doc_id}) ---\n{chunk_text}"
            if total_chars + len(block) > max_chars and context_blocks:
                break
            context_blocks.append(block)
            total_chars += len(block) + 2

    context_str = "\n\n".join(context_blocks)
    options_text = "\n".join([f"{key}. {text}" for key, text in options])

    system_msg = (
        "คุณเป็นผู้เชี่ยวชาญในการตอบข้อสอบปรนัยที่มีความรอบคอบ ละเอียดถี่ถ้วน และแม่นยำสูงสุด 100%\n"
        "หน้าที่ของคุณคือวิเคราะห์คำถามและเลือกคำตอบที่ถูกต้องที่สุดเพียงข้อเดียวตามข้อมูลในเอกสารอ้างอิง\n"
        "กฎเหล็กในการตอบ:\n"
        "1. ยึดข้อมูลจาก 'เอกสารอ้างอิง' เป็นหลักอย่างเคร่งครัดที่สุด หากข้อความในเอกสารกล่าวถึง ให้ถือเป็นความจริงสูงสุด\n"
        "2. หากข้อมูลไม่ปรากฏในเอกสารอ้างอิง ให้ใช้หลักการทางวิชาการและทฤษฎีคอมพิวเตอร์ที่ถูกต้องที่สุด\n"
        "3. การวิเคราะห์คำถามและประโยคปฏิเสธ (สำคัญมาก):\n"
        "   - สังเกตว่าคำถามถามหาข้อที่ถูกหรือข้อที่ผิด เช่น 'ไม่ถูกต้อง', 'ไม่ใช่', 'ยกเว้น', 'ข้อใดผิด'\n"
        "   - หากถามว่า 'ข้อใดผิด' หรือ 'ยกเว้น': ตัวเลือกที่สอดคล้องกับเอกสารคือตัวเลือกที่ต้องตัดทิ้ง ตัวเลือกที่ขัดแย้ง/ไม่จริงคือคำตอบที่ต้องเลือก!\n"
        "4. หากเป็นโจทย์คำนวณ ตัวเลข หรือตารางข้อมูล:\n"
        "   - อ่านแถวและคอลัมน์ให้ตรงเงื่อนไข (ยอดขั้นต่ำ, เพดานส่วนลด, อัตราส่วนลด)\n"
        "   - ตรวจสอบเพดานส่วนลดสูงสุด (Limit/Cap): หากคำนวณตามเปอร์เซ็นต์แล้วเกินเพดาน ต้องตอบเท่ากับเพดานสูงสุดเท่านั้น เช่น เพดาน 50 บาท คำนวณได้ 100 บาท ต้องตอบ 50 บาท\n"
        "5. โครงสร้างการคิดในฟิลด์ 'reason':\n"
        "   - ขั้นที่ 1: ระบุเป้าหมายของคำถาม (เช่น โจทย์ถามหาข้อที่ผิด หรือข้อที่ถูกต้อง)\n"
        "   - ขั้นที่ 2: ตรวจสอบตัวเลือก ก, ข, ค, ง, จ ทีละข้อเทียบกับเอกสารอ้างอิง\n"
        "   - ขั้นที่ 3: ฟันธงตัวเลือกที่ถูกต้องตรงตามเป้าหมายของคำถามเพียงข้อเดียว\n"
        "6. ในฟิลด์ 'answer': ระบุเฉพาะตัวอักษร key ตัวเลือกเดี่ยว (เช่น ก, ข, ค, ง, จ) ให้ตรงกับบทสรุปใน reason"
    )

    user_parts = []
    if context_str:
        user_parts.append(f"เอกสารอ้างอิง:\n{context_str}")
    user_parts.append(f"คำถาม:\n{question}")
    user_parts.append(f"ตัวเลือก:\n{options_text}")
    user_parts.append("โปรดตอบในรูปแบบ JSON ที่มี reason และ answer (ระบุเฉพาะตัวอักษร key เช่น ก, ข, ค, ง, จ):")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message
    """
    context_blocks = []
    total_chars = 0
    max_chars = getattr(settings, "MAX_CONTEXT_CHARS", 6000)

    if hits:
        for chunk, score in hits:
            doc_id = getattr(chunk, "doc_id", "doc")
            chunk_text = getattr(chunk, "text", str(chunk)).strip()
            block = f"--- เอกสารอ้างอิงและคู่มือ ({doc_id}) ---\n{chunk_text}"
            if total_chars + len(block) > max_chars and context_blocks:
                break
            context_blocks.append(block)
            total_chars += len(block) + 2

    context_str = "\n\n".join(context_blocks)

    system_msg = (
        "คุณเป็นโปรแกรมเมอร์ Python ผู้เชี่ยวชาญระดับสูง (Senior Python Engineer)\n"
        "หน้าที่ของคุณคือเขียนฟังก์ชัน Python ตาม requirement ที่กำหนดให้ถูกต้องสมบูรณ์ 100% ผ่านทุก unit test cases\n"
        "ข้อกำหนดเคร่งครัด:\n"
        "1. ต้องใช้ signature และชื่อฟังก์ชันตามที่กำหนดเป๊ะ ๆ ห้ามเปลี่ยนชื่อหรือพารามิเตอร์เด็ดขาด\n"
        "2. ใช้เฉพาะ Python Standard Library เท่านั้น (หากใช้ re หรือ math ให้ import ที่บรรทัดแรกสุดของโค้ด) ห้าม import ไลบรารีภายนอก\n"
        "3. ห้ามใช้คำสั่ง print() หรือรับ input() เด็ดขาด ให้รับค่าผ่านพารามิเตอร์และส่งผลลัพธ์กลับด้วย return เท่านั้น\n"
        "4. กฎเหล็กข้อห้าม (FORBIDDEN FUNCTIONS / METHODS / VARIABLE NAMES) — สำคัญสูงสุด:\n"
        "   - ตรวจสอบ requirement ทันทีว่ามีคำว่า 'ห้ามใช้' หรือไม่\n"
        "   - ระบบตรวจจับผ่าน Abstract Syntax Tree (AST): หากโจทย์ห้ามใช้คำใด ห้ามมีคำนั้นปรากฏเป็น 'ชื่อฟังก์ชัน', 'ชื่อเมธอด', หรือ 'ชื่อตัวแปร' ใด ๆ ในโค้ดโดยเด็ดขาด! (หากตรวจพบชื่อที่ถูกห้ามจะถูกปรับเป็น 0 คะแนนทันทีทั้งข้อ)\n"
        "   - ตัวอย่างข้อควรระวังเรื่องชื่อตัวแปรและเมธอด:\n"
        "     * หากห้าม sum: ห้ามเขียน `sum = ...` เด็ดขาด! ให้ใช้ `total = 0` แล้ววน loop บวกสะสมค่า\n"
        "     * หากห้าม min/max: ห้ามเขียน `min = ...` หรือ `max = ...` เด็ดขาด! ให้ใช้ `lowest`, `highest`, `best` แล้ววน loop เปรียบเทียบ\n"
        "     * หากห้าม count: ห้ามเขียน `count = ...` และห้ามเรียก `.count()` เด็ดขาด! ให้ใช้ `cnt = 0` หรือ `freq = 0` แล้ววน loop นับ `if x == target: cnt += 1`\n"
        "     * หากห้าม sort/sorted: ห้ามเรียก `.sort()` หรือ `sorted()` เด็ดขาด! ให้วน loop เรียงลำดับด้วย bubble sort หรือ insertion sort\n"
        "     * หากห้าม reverse/reversed: ห้ามเรียก `.reverse()` หรือ `reversed()` เด็ดขาด! ให้ใช้ slicing `[::-1]` หรือวน loop ถอยหลัง\n"
        "     * หากห้าม abs: ห้ามเรียก `abs(...)` เด็ดขาด! ให้ใช้ `(x if x >= 0 else -x)`\n"
        "     * หากห้าม len: ห้ามเรียก `len(...)` เด็ดขาด! ให้ใช้วน loop นับความยาว `n = 0; for _ in s: n += 1`\n"
        "5. หากมีเอกสารอ้างอิงและคู่มือประกอบโจทย์:\n"
        "   - อ่านกฎเกณฑ์ สูตรคำนวณ และเงื่อนไขในตารางอย่างละเอียดถี่ถ้วน\n"
        "   - แปลงรหัสข้อความเป็นตัวพิมพ์ใหญ่ด้วย .upper() ก่อนตรวจสอบเสมอ\n"
        "   - สำหรับงานตรวจสอบรูปแบบ (format validation): ให้ import re ไว้บรรทัดแรก แล้วใช้ re.fullmatch(...) โดยใช้ [A-Z0-9] แทน \\w (เพราะ \\w รวมขีดล่าง _)\n"
        "   - สำหรับการค้นหาในตารางหรือ dictionary: ต้องตรวจสอบ `if key not in table: return 0.0` (หรือค่า default ตามโจทย์) เป็นเงื่อนไขแรกเสมอ เพื่อป้องกัน KeyError\n"
        "   - สำหรับการคำนวณส่วนลดหรืออัตราค่าบริการ: แยกกรณีมูลค่าคงที่กับเปอร์เซ็นต์ และจำกัดไม่เกินเพดานสูงสุดตามตาราง\n"
        "6. การจัดการ Data Types และ Edge Cases อย่างรัดกุม:\n"
        "   - คืนค่าให้ตรงกับ Type ใน Signature เสมอ (เช่น คืน float, int, bool, list, str ให้ถูกต้อง)\n"
        "   - ดักกรณีข้อมูลนำเข้าเป็นค่าว่าง (empty list [], empty string '', None), ค่าเป็น 0, ค่าติดลบ, และการหารด้วยศูนย์\n"
        "7. ตอบเฉพาะโค้ด Python ที่สมบูรณ์ในบล็อก ```python ... ``` เพียงบล็อกเดียวเท่านั้น โดยไม่ต้องใส่คำอธิบายอื่นนอกบล็อกโค้ด"
    )

    user_parts = []
    if context_str:
        user_parts.append(f"เอกสารอ้างอิงและคู่มือ:\n{context_str}")
    user_parts.append(f"โจทย์ความต้องการ:\n{requirement}")

    sig_hint = signature or (f"def {entry_point}(*args, **kwargs):" if entry_point else None)
    if sig_hint:
        user_parts.append(f"Signature ของฟังก์ชันที่ต้องเขียน:\n```python\n{sig_hint}\n```")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
