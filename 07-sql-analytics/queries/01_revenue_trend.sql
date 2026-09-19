-- Question: how is revenue trending, and is the latest month unusual?
-- Technique: monthly aggregation, LAG for month-over-month and year-over-year,
-- and a trailing three-month average to separate signal from single-month noise.
WITH monthly AS (
    SELECT
        strftime('%Y-%m', i.InvoiceDate)      AS month,
        ROUND(SUM(i.Total), 2)                AS revenue,
        COUNT(*)                              AS orders,
        COUNT(DISTINCT i.CustomerId)          AS buyers
    FROM Invoice i
    GROUP BY 1
),
with_lags AS (
    SELECT
        month, revenue, orders, buyers,
        LAG(revenue, 1)  OVER (ORDER BY month) AS prev_month,
        LAG(revenue, 12) OVER (ORDER BY month) AS same_month_last_year,
        ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
            AS trailing_3m_avg
    FROM monthly
)
SELECT
    month,
    revenue,
    orders,
    buyers,
    ROUND(revenue / orders, 2)                                    AS avg_order_value,
    trailing_3m_avg,
    ROUND(100.0 * (revenue - prev_month) / prev_month, 1)         AS mom_pct,
    ROUND(100.0 * (revenue - same_month_last_year)
                / same_month_last_year, 1)                        AS yoy_pct
FROM with_lags
ORDER BY month;
