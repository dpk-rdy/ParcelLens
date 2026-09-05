-- Dimensional weight exposure by client and box size.
-- Ranks the box sizes doing the most damage, which is where a packaging change
-- has to start.
SELECT
    client,
    box_id,
    COUNT(*)                                                  AS shipment_count,
    ROUND(AVG(item_cube_in3 / box_cube_in3), 3)               AS cube_utilization,
    ROUND(AVG(manifest_weight_lb), 1)                         AS avg_scale_lb,
    ROUND(AVG(billable_lb), 1)                                AS avg_billable_lb,
    ROUND(AVG(billable_lb - CEIL(manifest_weight_lb)), 2)     AS avg_dim_penalty_lb,
    ROUND(100.0 * AVG(CASE WHEN billable_lb > CEIL(manifest_weight_lb)
                           THEN 1.0 ELSE 0.0 END), 1)         AS dim_billed_pct,
    ROUND(SUM(expected_total), 0)                             AS spend
FROM shipments
GROUP BY 1, 2
HAVING COUNT(*) > 500
-- Rank by total pounds of DIM penalty, not the per-package average:
-- a small penalty on high volume outranks a large one on a rare box.
ORDER BY AVG(billable_lb - CEIL(manifest_weight_lb)) * COUNT(*) DESC
LIMIT 20;
