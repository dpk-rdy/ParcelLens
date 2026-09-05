-- Where to point an analyst's week: clients ranked by recoverable dollars per
-- invoice line, so effort goes where each dispute is worth the most.
SELECT
    client,
    carrier,
    COUNT(*)                                        AS exception_lines,
    ROUND(SUM(recovery_usd), 0)                     AS recovery_usd,
    ROUND(SUM(recovery_usd) / COUNT(*), 2)          AS recovery_per_line,
    MODE(exception_type)                            AS most_common_exception
FROM audited
WHERE exception_type <> 'none'
GROUP BY 1, 2
HAVING SUM(recovery_usd) > 0
ORDER BY recovery_usd DESC
LIMIT 15;
