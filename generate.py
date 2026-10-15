"""Synthetic bilingual (Arabic + English) UAE business documents with ground truth.

Three document types common in UAE onboarding, KYC, procurement and leasing workflows:
  * tax_invoice      - VAT invoice (5% UAE VAT, 15-digit TRN)
  * trade_licence    - commercial licence (licence number, activity, issue/expiry)
  * tenancy_contract - Ejari-style tenancy contract (annual rent, cheques, dates)

Documents are plain text, the way they arrive after OCR. Layouts vary (label order,
Arabic-only or English-only labels, Arabic-Indic digits, OCR spacing noise) and a share
of documents carry deliberate business errors (wrong VAT, expired licence, rent that does
not split into the cheques) so the validation layer can be measured.
All names, numbers and companies are invented.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

COMPANIES = ["Al Noor Trading LLC", "Gulf Horizon Logistics FZE", "Desert Pearl General Trading", "Marina Tech Solutions LLC",
             "Emirates Fresh Foods LLC", "Falcon Line Transport LLC", "Oasis Medical Supplies FZ-LLC", "Blue Dhow Marine Services",
             "Sahara Build Contracting LLC", "Palm Retail Group LLC", "Zayed Port Shipping LLC", "Creek Side Electronics LLC"]
PEOPLE = ["Ahmed Al Mansouri", "Fatima Al Hashimi", "Rahul Menon", "Maria Santos", "Omar Khalid", "Aisha Rahman",
          "John Peterson", "Priya Nair", "Yousef Al Ali", "Leena George", "Hassan Qureshi", "Sara Al Suwaidi"]
ACTIVITIES = ["General Trading", "Logistics Services", "IT Consultancy", "Foodstuff Trading", "Marine Services",
              "Medical Equipment Trading", "Building Contracting", "Electronics Trading"]
AUTHORITIES = ["Dubai Economy and Tourism", "Abu Dhabi Department of Economic Development", "Sharjah Economic Development Department",
               "Jebel Ali Free Zone Authority", "Dubai Multi Commodities Centre"]
AREAS = ["Business Bay", "Al Barsha", "Jumeirah Village Circle", "Deira", "Dubai Marina", "Al Nahda", "Mirdif", "JLT"]
ITEMS = [("Laptop", 3200), ("Office chair", 450), ("Printer toner", 180), ("Pallet of water", 95), ("Network switch", 1250),
         ("Forklift service", 2200), ("Cleaning service (monthly)", 1800), ("Diesel (100 L)", 310), ("Safety helmets (box)", 260)]

# Label wording seen in the wild that the rule dictionary does not know; the LLM fallback covers these.
UNSEEN_LABELS = {
    "invoice_no": "Inv. Ref", "invoice_date": "Date of Supply", "supplier_trn": "TRN", "total": "Amount Due (AED)",
    "licence_no": "License Number", "expiry_date": "Valid Until", "activity": "Business Activity",
    "contract_no": "Registration No.", "annual_rent": "Yearly Rent (AED)", "cheques": "Payment Cheques",
}

LABELS = {
    "tax_invoice": {
        "title": ("TAX INVOICE", "فاتورة ضريبية"),
        "invoice_no": ("Invoice No", "رقم الفاتورة"),
        "invoice_date": ("Invoice Date", "تاريخ الفاتورة"),
        "supplier": ("Supplier", "المورد"),
        "supplier_trn": ("Supplier TRN", "الرقم الضريبي للمورد"),
        "customer": ("Customer", "العميل"),
        "subtotal": ("Subtotal (AED)", "المجموع قبل الضريبة"),
        "vat": ("VAT 5% (AED)", "ضريبة القيمة المضافة ٥٪"),
        "total": ("Total (AED)", "الإجمالي"),
    },
    "trade_licence": {
        "title": ("COMMERCIAL LICENCE", "رخصة تجارية"),
        "licence_no": ("Licence No", "رقم الرخصة"),
        "company": ("Trade Name", "الاسم التجاري"),
        "activity": ("Activity", "النشاط"),
        "authority": ("Issuing Authority", "جهة الإصدار"),
        "issue_date": ("Issue Date", "تاريخ الإصدار"),
        "expiry_date": ("Expiry Date", "تاريخ الانتهاء"),
        "manager": ("Manager", "المدير"),
    },
    "tenancy_contract": {
        "title": ("TENANCY CONTRACT", "عقد إيجار"),
        "contract_no": ("Ejari No", "رقم إيجاري"),
        "landlord": ("Landlord", "المؤجر"),
        "tenant": ("Tenant", "المستأجر"),
        "property": ("Property", "العقار"),
        "start_date": ("Contract Start", "تاريخ البداية"),
        "end_date": ("Contract End", "تاريخ النهاية"),
        "annual_rent": ("Annual Rent (AED)", "الإيجار السنوي"),
        "cheques": ("No. of Cheques", "عدد الشيكات"),
    },
}


@dataclass
class Document:
    doc_id: str
    doc_type: str
    text: str
    fields: dict
    errors: list = field(default_factory=list)   # injected business errors (ground truth for validation)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _d(x: date) -> str:
    return x.strftime("%d/%m/%Y")


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _fields(rng: random.Random, doc_type: str, today: date) -> tuple[dict, list]:
    errors: list[str] = []
    if doc_type == "tax_invoice":
        lines = rng.sample(ITEMS, rng.randint(1, 4))
        subtotal = round(sum(p * rng.randint(1, 12) for _, p in lines), 2)
        vat = round(subtotal * 0.05, 2)
        if rng.random() < 0.15:                                  # wrong VAT amount
            vat = round(vat * rng.choice([0.5, 1.4, 2.0]), 2)
            errors.append("vat_mismatch")
        f = {"invoice_no": f"INV-{rng.randint(2024, 2026)}-{rng.randint(10000, 99999)}",
             "invoice_date": _d(today - timedelta(days=rng.randint(0, 120))),
             "supplier": rng.choice(COMPANIES), "supplier_trn": "1" + "".join(str(rng.randint(0, 9)) for _ in range(14)),
             "customer": rng.choice(COMPANIES), "subtotal": _money(subtotal), "vat": _money(vat),
             "total": _money(subtotal + vat)}
        if rng.random() < 0.08:                                  # TRN with a missing digit
            f["supplier_trn"] = f["supplier_trn"][:-1]
            errors.append("invalid_trn")
        return f, errors
    if doc_type == "trade_licence":
        issue = today - timedelta(days=rng.randint(30, 700))
        expiry = issue + timedelta(days=365)
        if expiry < today and "expired" not in errors:
            errors.append("expired_licence")
        return {"licence_no": str(rng.randint(100000, 1999999)), "company": rng.choice(COMPANIES),
                "activity": rng.choice(ACTIVITIES), "authority": rng.choice(AUTHORITIES), "issue_date": _d(issue),
                "expiry_date": _d(expiry), "manager": rng.choice(PEOPLE)}, errors
    start = today - timedelta(days=rng.randint(0, 300))
    cheques = rng.choice([1, 2, 4, 6, 12])
    rent = rng.choice([45000, 55000, 62000, 72000, 85000, 98000, 120000, 145000, 180000])
    if rng.random() < 0.12:                                      # rent that does not divide into whole-dirham cheques
        rent += rng.choice([1, 3, 7])
        if cheques == 1:
            cheques = 4
    if rent % cheques:
        errors.append("rent_not_divisible")
    return {"contract_no": f"{rng.randint(2024, 2026)}{rng.randint(10 ** 9, 10 ** 10 - 1)}",
            "landlord": rng.choice(PEOPLE), "tenant": rng.choice(PEOPLE + COMPANIES),
            "property": f"Unit {rng.randint(101, 3405)}, {rng.choice(AREAS)}, Dubai",
            "start_date": _d(start), "end_date": _d(start + timedelta(days=364)), "annual_rent": _money(rent),
            "cheques": str(cheques)}, errors


def _render(rng: random.Random, doc_type: str, f: dict) -> str:
    labels = LABELS[doc_type]
    style = rng.choices(["bilingual", "english", "arabic"], weights=[0.5, 0.3, 0.2])[0]
    arabic_digits = style == "arabic" and rng.random() < 0.6
    title_en, title_ar = labels["title"]
    out = [f"{title_en}  {title_ar}" if style == "bilingual" else (title_en if style == "english" else title_ar)]
    keys = list(f)
    if rng.random() < 0.4:                                       # different issuers order fields differently
        head, tail = keys[:2], keys[2:]
        rng.shuffle(tail)
        keys = head + tail
    unseen = rng.random() < 0.18                                # issuer uses its own wording for some labels
    for k in keys:
        en, ar = labels[k]
        lab = {"bilingual": f"{en} / {ar}", "english": en, "arabic": ar}[style]
        if unseen and k in UNSEEN_LABELS and rng.random() < 0.6:
            lab = UNSEEN_LABELS[k]
        val = f[k]
        if arabic_digits:
            val = val.translate(AR_DIGITS)
        sep = rng.choice([": ", " : ", ":  ", " - "])
        out.append(f"{lab}{sep}{val}")
    if doc_type == "tax_invoice":
        out.append(rng.choice(["Thank you for your business.", "شكراً لتعاملكم معنا", "Payment due within 30 days."]))
    if doc_type == "tenancy_contract":
        out.append(rng.choice(["Subject to Dubai Land Department regulations.", "يخضع هذا العقد لأنظمة دائرة الأراضي والأملاك"]))
    text = "\n".join(out)
    if rng.random() < 0.06:                                      # OCR misreads a character inside one value
        lines = text.splitlines()
        i = rng.randrange(1, len(lines))
        lines[i] = lines[i][:-3] + lines[i][-3:].replace("0", "O").replace("1", "l").replace("5", "S")
        text = "\n".join(lines)
    if rng.random() < 0.25:                                      # OCR noise: doubled spaces and a stray line break
        text = text.replace(" ", "  ", rng.randint(1, 3))
    return text


def generate(n: int = 600, seed: int = 7, today: date = date(2026, 10, 1)) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        t = rng.choice(list(LABELS))
        f, errors = _fields(rng, t, today)
        docs.append(Document(f"DOC-{i:05d}", t, _render(rng, t, f), f, errors))
    return docs
