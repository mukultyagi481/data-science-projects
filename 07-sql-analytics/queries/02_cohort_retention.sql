-- Question: do customers acquired in different quarters keep buying at
-- different rates? Acquisition channels change over time, so cohort behaviour
-- matters more than a single blended retention number.
-- Technique: first-purchase cohort via MIN() window, then months-since-signup
-- buckets, expressed as a share of the cohort's original size.
WITH first_purchase AS (
    SELECT
        CustomerId,
        MIN(DATE(InvoiceDate)) AS first_date
    FROM Invoice
    GROUP BY CustomerId
),
cohorts AS (
    SELECT
        f.CustomerId,
        strftime('%Y', f.first_date) || '-Q' ||
            ((CAST(strftime('%m', f.first_date) AS INTEGER) + 2) / 3) AS cohort,
        f.first_date,
        DATE(i.InvoiceDate) AS purchase_date,
        i.Total
    FROM first_purchase f
    JOIN Invoice i ON i.CustomerId = f.CustomerId
),
activity AS (
    SELECT
        cohort,
        CustomerId,
        CAST((JULIANDAY(purchase_date) - JULIANDAY(first_date)) / 91.0 AS INTEGER)
            AS quarters_since_first,
        Total
    FROM cohorts
),
cohort_size AS (
    SELECT cohort, COUNT(DISTINCT CustomerId) AS customers
    FROM activity
    WHERE quarters_since_first = 0
    GROUP BY cohort
)
SELECT
    a.cohort,
    s.customers                                            AS cohort_size,
    a.quarters_since_first                                 AS quarter_offset,
    COUNT(DISTINCT a.CustomerId)                           AS active_customers,
    ROUND(100.0 * COUNT(DISTINCT a.CustomerId) / s.customers, 1) AS pct_of_cohort,
    ROUND(SUM(a.Total), 2)                                 AS revenue
FROM activity a
JOIN cohort_size s ON s.cohort = a.cohort
GROUP BY a.cohort, a.quarters_since_first, s.customers
HAVING a.quarters_since_first <= 8
ORDER BY a.cohort, a.quarters_since_first;
