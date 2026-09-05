-- What changed, why, and what it cost.
-- Compares each carrier's share of volume and its cost per package in the
-- second half against the first, so a mix shift and a rate shift are separated
-- rather than blended into one movement in total spend.
WITH halves AS (
    SELECT carrier,
           CASE WHEN month <= 6 THEN 'H1' ELSE 'H2' END AS period,
           COUNT(*)          AS shipments,
           SUM(expected_total) AS spend
    FROM shipments
    GROUP BY 1, 2
),
totals AS (
    SELECT period, SUM(shipments) AS total_shipments FROM halves GROUP BY 1
),
shares AS (
    SELECT h.carrier, h.period,
           h.shipments,
           h.shipments * 1.0 / t.total_shipments AS volume_share,
           h.spend / h.shipments                 AS cost_per_package
    FROM halves h JOIN totals t USING (period)
)
SELECT
    a.carrier,
    ROUND(100.0 * a.volume_share, 2)                       AS share_h1_pct,
    ROUND(100.0 * b.volume_share, 2)                       AS share_h2_pct,
    ROUND(100.0 * (b.volume_share - a.volume_share), 2)    AS share_change_pp,
    ROUND(a.cost_per_package, 2)                           AS cpp_h1,
    ROUND(b.cost_per_package, 2)                           AS cpp_h2,
    ROUND(100.0 * (b.cost_per_package / a.cost_per_package - 1), 2) AS cpp_change_pct,
    -- Cost of the mix shift alone, holding the first-half rate constant.
    ROUND((b.volume_share - a.volume_share) * b.shipments
          / NULLIF(b.volume_share, 0) * a.cost_per_package, 0) AS mix_shift_impact
FROM shares a
JOIN shares b ON a.carrier = b.carrier AND a.period = 'H1' AND b.period = 'H2'
ORDER BY ABS(b.volume_share - a.volume_share) DESC;
