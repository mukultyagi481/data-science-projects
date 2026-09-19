-- Question: which genres should we promote in each of our largest markets?
-- A global top-genre list hides the fact that taste is local.
-- Technique: revenue rolled up to country and genre, then RANK() within each
-- country partition, restricted to markets big enough to act on.
WITH line_revenue AS (
    SELECT
        c.Country,
        g.Name                              AS genre,
        il.UnitPrice * il.Quantity          AS revenue
    FROM InvoiceLine il
    JOIN Invoice  i ON i.InvoiceId = il.InvoiceId
    JOIN Customer c ON c.CustomerId = i.CustomerId
    JOIN Track    t ON t.TrackId = il.TrackId
    LEFT JOIN Genre g ON g.GenreId = t.GenreId
),
country_totals AS (
    SELECT Country, SUM(revenue) AS country_revenue
    FROM line_revenue
    GROUP BY Country
    HAVING SUM(revenue) >= 50          -- ignore markets too small to draw conclusions from
),
ranked AS (
    SELECT
        lr.Country,
        lr.genre,
        ROUND(SUM(lr.revenue), 2)                             AS genre_revenue,
        ROUND(100.0 * SUM(lr.revenue) / ct.country_revenue, 1) AS pct_of_country,
        RANK() OVER (PARTITION BY lr.Country
                     ORDER BY SUM(lr.revenue) DESC)            AS genre_rank
    FROM line_revenue lr
    JOIN country_totals ct ON ct.Country = lr.Country
    GROUP BY lr.Country, lr.genre, ct.country_revenue
)
SELECT Country, genre_rank, genre, genre_revenue, pct_of_country
FROM ranked
WHERE genre_rank <= 3
ORDER BY Country, genre_rank;
