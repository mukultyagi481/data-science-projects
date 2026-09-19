-- Question: how long do customers normally wait between orders? Without this,
-- "churned" is guesswork -- you cannot flag a lapsed customer until you know
-- what a normal gap looks like.
-- Technique: LAG over each customer's order history to measure gaps, then
-- compare each customer's most recent gap against their own median behaviour.
WITH ordered AS (
    SELECT
        CustomerId,
        DATE(InvoiceDate) AS order_date,
        LAG(DATE(InvoiceDate)) OVER (PARTITION BY CustomerId ORDER BY InvoiceDate)
            AS prev_order_date
    FROM Invoice
),
gaps AS (
    SELECT
        CustomerId,
        order_date,
        CAST(JULIANDAY(order_date) - JULIANDAY(prev_order_date) AS INTEGER) AS gap_days
    FROM ordered
    WHERE prev_order_date IS NOT NULL
),
per_customer AS (
    SELECT
        CustomerId,
        COUNT(*)                AS gaps_observed,
        ROUND(AVG(gap_days), 1) AS avg_gap_days,
        MAX(gap_days)           AS longest_gap_days,
        MAX(order_date)         AS last_order_date
    FROM gaps
    GROUP BY CustomerId
),
latest AS (
    SELECT MAX(DATE(InvoiceDate)) AS data_through FROM Invoice
)
SELECT
    p.CustomerId,
    p.gaps_observed,
    p.avg_gap_days,
    p.longest_gap_days,
    p.last_order_date,
    CAST(JULIANDAY(l.data_through) - JULIANDAY(p.last_order_date) AS INTEGER)
        AS days_since_last_order,
    ROUND((JULIANDAY(l.data_through) - JULIANDAY(p.last_order_date))
              / p.avg_gap_days, 2) AS gaps_overdue,
    CASE
        WHEN (JULIANDAY(l.data_through) - JULIANDAY(p.last_order_date))
                 > 2 * p.avg_gap_days THEN 'lapsed'
        WHEN (JULIANDAY(l.data_through) - JULIANDAY(p.last_order_date))
                 > p.avg_gap_days THEN 'at risk'
        ELSE 'active'
    END AS status
FROM per_customer p
CROSS JOIN latest l
WHERE p.gaps_observed >= 3
ORDER BY gaps_overdue DESC
LIMIT 25;
