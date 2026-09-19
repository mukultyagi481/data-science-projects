-- Question: how concentrated is revenue among customers? This decides whether
-- retention work should target a small high-value group or the broad base.
-- Technique: NTILE to build value deciles, then a cumulative window sum to
-- read off what share of revenue the top deciles account for.
WITH customer_value AS (
    SELECT
        c.CustomerId,
        c.Country,
        COUNT(DISTINCT i.InvoiceId)                    AS orders,
        ROUND(SUM(i.Total), 2)                         AS lifetime_revenue,
        MIN(DATE(i.InvoiceDate))                       AS first_order,
        MAX(DATE(i.InvoiceDate))                       AS last_order
    FROM Customer c
    JOIN Invoice i ON i.CustomerId = c.CustomerId
    GROUP BY c.CustomerId, c.Country
),
deciles AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY lifetime_revenue DESC) AS value_decile
    FROM customer_value
),
by_decile AS (
    SELECT
        value_decile,
        COUNT(*)                        AS customers,
        ROUND(SUM(lifetime_revenue), 2) AS revenue,
        ROUND(AVG(orders), 1)           AS avg_orders,
        ROUND(AVG(lifetime_revenue), 2) AS avg_revenue
    FROM deciles
    GROUP BY value_decile
)
SELECT
    value_decile,
    customers,
    revenue,
    avg_orders,
    avg_revenue,
    ROUND(100.0 * SUM(revenue) OVER (ORDER BY value_decile)
              / SUM(revenue) OVER (), 1) AS cumulative_pct_of_revenue
FROM by_decile
ORDER BY value_decile;
