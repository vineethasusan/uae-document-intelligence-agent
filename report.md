# UAE Document Intelligence Agent: evaluation report

Test set: 540 unseen documents (invoices, trade licences, tenancy contracts) in Arabic, English or both.

| Metric | Result |
|---|---|
| Document type classification | 100.0% |
| Field-level extraction accuracy | 99.2% |
| Documents with every field correct | 93.5% |
| Business-rule checks: precision / recall | 98.9% / 99.5% |
| Q&A: right document in top 3 | 94.0% |
| Q&A: exact answer (offline engine) | 94.0% |
| Fields filled by rules / semantic fallback | 96.1% / 3.4% |
| Documents routed to LLM or human review (offline) | 6.3% |
| Accuracy of documents processed without review | 99.8% |
| Processing time per document | 2.0 ms |

## Issues flagged

- vat_mismatch: 24
- rent_not_divisible: 61
- expired_licence: 90
- needs_review: 34
- invalid_trn: 12

## Example outputs

**DOC-00360** (tax_invoice)
```json
{
  "fields": {
    "invoice_no": "INV-2025-58206",
    "invoice_date": "21/09/2026",
    "supplier": "Sahara Build Contracting LLC",
    "supplier_trn": "136123578355061",
    "customer": "Sahara Build Contracting LLC",
    "subtotal": "44,685.00",
    "vat": "2,234.25",
    "total": "46,919.25"
  },
  "issues": []
}
```

**DOC-00361** (tax_invoice)
```json
{
  "fields": {
    "invoice_no": "INV-2025-46392",
    "invoice_date": "24/08/2026",
    "supplier": "Gulf Horizon Logistics FZE",
    "supplier_trn": "158678896087889",
    "customer": "Falcon Line Transport LLC",
    "subtotal": "4,050.00",
    "vat": "283.50",
    "total": "4,333.50"
  },
  "issues": [
    "vat_mismatch"
  ]
}
```

**DOC-00362** (trade_licence)
```json
{
  "fields": {
    "licence_no": "1493394",
    "company": "Blue Dhow Marine Services",
    "activity": "Logistics Services",
    "authority": "Sharjah Economic Development Department",
    "issue_date": "29/07/2026",
    "expiry_date": "29/07/2027",
    "manager": "Hassan Qureshi"
  },
  "issues": []
}
```
