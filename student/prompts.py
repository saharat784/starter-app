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
        "คุณเป็นผู้เชี่ยวชาญในการตอบข้อสอบปรนัยที่มีความรอบคอบและแม่นยำสูงสุด\n"
        "หน้าที่ของคุณคือเลือกคำตอบที่ถูกต้องที่สุดเพียงข้อเดียวตามข้อมูลในเอกสารอ้างอิงที่กำหนด\n"
        "กฎสำคัญ:\n"
        "1. ยึดข้อมูลจาก 'เอกสารอ้างอิง' เป็นหลักอย่างเคร่งครัดที่สุด แม้จะขัดแย้งกับความรู้ทั่วไป\n"
        "2. หากไม่มีเอกสารอ้างอิงหรือข้อมูลไม่ปรากฏ ให้ใช้ความรู้เชิงวิชาการที่ถูกต้องที่สุด\n"
        "3. หากมีตาราง ให้ค้นหาแถวของคีย์ที่เกี่ยวข้องและอ่านคอลัมน์ให้ตรงกันอย่างระมัดระวัง (เช่น ยอดซื้อขั้นต่ำ, ส่วนลดสูงสุดต่อบิล)\n"
        "4. สำหรับโจทย์คำนวณส่วนลด:\n"
        "   - ตรวจสอบยอดซื้อขั้นต่ำ\n"
        "   - คำนวณส่วนลดตามสูตร (เช่น เปอร์เซ็นต์ หรือ ลดเป็นเงินสด)\n"
        "   - ตรวจสอบว่าคูปองนั้นมี 'ส่วนลดสูงสุดต่อบิล' (เพดานส่วนลด) หรือไม่ หากคำนวณได้มากกว่าเพดาน จะต้องตอบเท่ากับเพดานสูงสุดเท่านั้น (เช่น หากคำนวณได้ 100 บาท แต่ส่วนลดสูงสุดต่อบิลคือ 50 บาท คำตอบที่ถูกต้องคือ 50 บาท ห้ามตอบ 100 บาทเด็ดขาด)\n"
        "5. ระวังคำถามเชิงปฏิเสธ เช่น 'ไม่', 'ยกเว้น', 'ผิด'\n"
        "6. ในฟิลด์ 'reason' ให้แสดงการตรวจสอบข้อมูลทีละขั้นตอน (ค่าในตาราง -> การคำนวณ -> การจำกัดเพดานสูงสุดต่อบิล) แล้วสรุป key ตัวอักษรเดี่ยวใน 'answer'\n"
        "7. ต้องตอบในรูปแบบ JSON ตาม schema"
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
        "2. การ import ไลบรารี: ใช้เฉพาะ Python Standard Library เท่านั้น หากมีการใช้โมดูล เช่น re หรือ math ต้องเขียนคำสั่ง `import re` หรือ `import math` ไว้ที่บรรทัดบนสุดนอกฟังก์ชันเสมอ (ห้ามเรียก re. หรือ math. ใด ๆ โดยไม่ได้ import เด็ดขาด)\n"
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
        "   - สำหรับงานตรวจรูปแบบ (format validation): บรรทัดแรกต้องมี `import re` แล้วแปลง code = str(code).upper() ก่อนใช้ re.fullmatch(...) โดยใช้ [A-Z0-9] แทน \\w (เพราะ \\w รวมขีดล่าง _)\n"
        "   - สำหรับงานคำนวณส่วนลดตามตาราง: ไม่ต้อง import re ให้เขียนตรรกะตรงไปตรงมาดังนี้:\n"
        "     1) บรรทัดแรกของฟังก์ชัน: แปลงรหัส code = str(code).upper() เสมอ เพื่อรองรับทั้งตัวพิมพ์เล็กและตัวพิมพ์ใหญ่\n"
        "     2) สร้างตาราง dictionary ของคูปองที่มีข้อมูล: ประเภทส่วนลด, มูลค่าส่วนลด, ยอดซื้อขั้นต่ำ (min_total), และส่วนลดสูงสุดต่อบิล (max_discount)\n"
        "     3) ตรวจสอบ `if code not in table: return 0.0` เป็นเงื่อนไขแรกเสมอ เพื่อป้องกัน KeyError และรหัสที่ไม่มีในตารางต้องได้ 0.0\n"
        "     4) ตรวจสอบเงื่อนไขยอดซื้อขั้นต่ำ: หาก total < coupon['min_total'] ต้อง return 0.0 ทันที\n"
        "     5) คำนวณส่วนลด: หากเป็นเงินสด discount = min(coupon['value'], total); หากเป็นเปอร์เซ็นต์ discount = total * (coupon['value'] / 100) และหากมีเพดาน discount = min(discount, coupon['max_discount'])\n"
        "     6) คืนค่าผลลัพธ์เป็น float(discount) เสมอ\n"
        "6. การจัดการ Data Types และ Edge Cases อย่างรัดกุม:\n"
        "   - คืนค่าให้ตรงกับ Type ใน Signature เสมอ (เช่น float, int, bool, list, str ให้ถูกต้อง)\n"
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
