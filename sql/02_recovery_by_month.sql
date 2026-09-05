-- Audit recovery by month and exception type, with the leakage rate.
-- Recovery in isolation is a vanity number; against spend it is a control metric.
SELECT
    a.month,
    ROUND(SUM(a.billed_total), 0)                                AS invoiced,
    ROUND(SUM(a.recovery_usd), 0)                                AS billing_recovery,
    ROUND(SUM(a.gsr_refund_usd), 0)                              AS service_refunds,
    ROUND(100.0 * SUM(a.recovery_usd + a.gsr_refund_usd)
          / NULLIF(SUM(a.billed_total), 0), 2)                   AS leakage_pct,
    SUM(CASE WHEN a.exception_type <> 'none' THEN 1 ELSE 0 END)  AS exception_lines,
    ROUND(100.0 * AVG(CASE WHEN a.exception_type <> 'none' THEN 1.0 ELSE 0.0 END), 2)
                                                                 AS exception_rate_pct
FROM audited a
GROUP BY 1
ORDER BY 1;
