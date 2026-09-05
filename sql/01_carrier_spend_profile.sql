-- Spend profile by carrier and zone band.
-- The first question on any parcel account: what does a package cost, and how
-- much of that cost is transportation rather than surcharge.
SELECT
    carrier,
    CASE WHEN zone <= 3 THEN 'Zone 2-3'
         WHEN zone <= 5 THEN 'Zone 4-5'
         ELSE 'Zone 6-8' END              AS zone_band,
    COUNT(*)                              AS shipments,
    ROUND(SUM(expected_total), 0)         AS spend,
    ROUND(AVG(expected_total), 2)         AS cost_per_package,
    ROUND(SUM(expected_total) / SUM(billable_lb), 3) AS cost_per_billable_lb,
    ROUND(AVG(billable_lb), 1)            AS avg_billable_lb,
    ROUND(SUM(expected_accessorials) / SUM(expected_total), 3) AS accessorial_share,
    ROUND(AVG(CASE WHEN delivered_late THEN 0.0 ELSE 1.0 END), 3) AS on_time_rate
FROM shipments
GROUP BY 1, 2
ORDER BY spend DESC;
