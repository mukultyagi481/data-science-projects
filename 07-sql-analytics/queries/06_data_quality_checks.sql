-- Question: before anyone reports these numbers, what in the data could make
-- them wrong? Every result above assumes invoice totals reconcile with line
-- items and that keys join cleanly. This checks both.
-- Technique: one row per check, so the output reads as a pass/fail report.
WITH invoice_vs_lines AS (
    SELECT
        i.InvoiceId,
        ROUND(i.Total, 2)                              AS invoice_total,
        ROUND(SUM(il.UnitPrice * il.Quantity), 2)      AS line_total
    FROM Invoice i
    JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
    GROUP BY i.InvoiceId, i.Total
)
SELECT 'invoices whose total does not match its line items' AS check_name,
       COUNT(*) AS failing_rows
FROM invoice_vs_lines
WHERE ABS(invoice_total - line_total) > 0.01

UNION ALL
SELECT 'invoices with no line items',
       COUNT(*)
FROM Invoice i
WHERE NOT EXISTS (SELECT 1 FROM InvoiceLine il WHERE il.InvoiceId = i.InvoiceId)

UNION ALL
SELECT 'invoice lines pointing at a missing track',
       COUNT(*)
FROM InvoiceLine il
LEFT JOIN Track t ON t.TrackId = il.TrackId
WHERE t.TrackId IS NULL

UNION ALL
SELECT 'tracks with no genre assigned',
       COUNT(*)
FROM Track
WHERE GenreId IS NULL

UNION ALL
SELECT 'customers with no country recorded',
       COUNT(*)
FROM Customer
WHERE Country IS NULL OR TRIM(Country) = ''

UNION ALL
SELECT 'invoices dated in the future',
       COUNT(*)
FROM Invoice
WHERE DATE(InvoiceDate) > DATE('now');
