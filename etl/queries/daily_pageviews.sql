-- File: etl/queries/daily_pageviews.sql
-- Description:
--   Aggregate daily page‑view counts for forecasting.

SELECT
  DATE(TIMESTAMP_MICROS(event_timestamp)) AS ds,    -- date column
  COUNTIF(event_name = 'page_view')       AS y      -- target value
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
GROUP BY ds
ORDER BY ds;
