from datetime import date

from docagent.extract import RuleExtractor, normalize
from docagent.validate import validate


def test_normalize_converts_arabic_indic_digits():
    assert normalize("الإجمالي : ١٢٬٥٠٠٫٠٠") == "الإجمالي : 12,500.00"


def test_rules_read_arabic_only_invoice():
    text = "فاتورة ضريبية\nرقم الفاتورة : INV-2026-12345\nالمجموع قبل الضريبة : ١٠٠٠٫٠٠\nضريبة القيمة المضافة ٥٪ : ٥٠٫٠٠"
    f = RuleExtractor().extract(text, "tax_invoice")
    assert f["invoice_no"] == "INV-2026-12345"
    assert f["subtotal"] == "1000.00" and f["vat"] == "50.00"


def test_vat_and_trn_checks():
    ok = {"subtotal": "1,000.00", "vat": "50.00", "total": "1,050.00", "supplier_trn": "100123456789012"}
    assert validate("tax_invoice", ok, date(2026, 1, 1)) == []
    bad = dict(ok, vat="80.00", supplier_trn="12345")
    assert {"vat_mismatch", "invalid_trn"} <= set(validate("tax_invoice", bad, date(2026, 1, 1)))


def test_expired_licence_and_rent_split():
    assert validate("trade_licence", {"expiry_date": "01/01/2026"}, date(2026, 6, 1)) == ["expired_licence"]
    assert "rent_not_divisible" in validate("tenancy_contract", {"annual_rent": "62,003.00", "cheques": "4"}, date(2026, 1, 1))
