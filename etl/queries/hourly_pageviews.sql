-- File: etl/queries/hourly_pageviews.sql
-- Description:
--   Aggregate page_view counts by HOUR for a higher-resolution time series.

SELECT
  -- Format as 'YYYY-MM-DD HH:00:00'
  FORMAT_TIMESTAMP(
    '%Y-%m-%d %H:00:00',
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(event_timestamp), HOUR)
  ) AS ds,
  COUNTIF(event_name = 'page_view') AS y
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
GROUP BY ds
ORDER BY ds;
