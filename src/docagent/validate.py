"""Business-rule checks a finance, KYC or leasing team would otherwise do by hand."""
from __future__ import annotations

from datetime import date, datetime


def _num(x) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _date(x) -> date | None:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(x).strip(), fmt).date()
        except (TypeError, ValueError):
            continue
    return None


def validate(doc_type: str, f: dict, today: date) -> list[str]:
    issues: list[str] = []
    if doc_type == "tax_invoice":
        sub, vat, total = _num(f.get("subtotal")), _num(f.get("vat")), _num(f.get("total"))
        if sub is not None and vat is not None and abs(vat - round(sub * 0.05, 2)) > 0.05:
            issues.append("vat_mismatch")
        if sub is not None and vat is not None and total is not None and abs(sub + vat - total) > 0.05:
            issues.append("total_mismatch")
        trn = f.get("supplier_trn")
        if trn is not None and not (str(trn).isdigit() and len(str(trn)) == 15):
            issues.append("invalid_trn")
    elif doc_type == "trade_licence":
        exp = _date(f.get("expiry_date"))
        if exp is not None and exp < today:
            issues.append("expired_licence")
    elif doc_type == "tenancy_contract":
        rent, cheques = _num(f.get("annual_rent")), _num(f.get("cheques"))
        if rent is not None and cheques:
            if abs(rent / cheques - round(rent / cheques)) > 1e-9:
                issues.append("rent_not_divisible")
        s, e = _date(f.get("start_date")), _date(f.get("end_date"))
        if s and e and not (330 <= (e - s).days <= 400):
            issues.append("unusual_term")
    return issues
