-- File: etl/queries/user_features.sql
-- Description:
--   Build a user‑level feature table for purchase‑conversion modelling.
--   Period: 1 Jan 2021 → 15 Jan 2021  (14‑day window).

SELECT
  user_pseudo_id,                                             -- PRIMARY KEY
  -- Behaviour counts
  COUNTIF(event_name = 'page_view')      AS num_pageviews,
  COUNTIF(event_name = 'add_to_cart')    AS num_addtocart,
  COUNTIF(event_name = 'session_start')  AS num_sessions,
  -- Engagement time (ms) summed over events with param key 'engagement_time_msec'
  SUM(
    (SELECT value.int_value
     FROM UNNEST(event_params)
     WHERE key = 'engagement_time_msec')
  ) AS total_engagement_ms,
  -- Distinct days visited (= user stickiness)
  COUNT(DISTINCT event_date)             AS active_days,
  -- Recency (days since last event in window, smaller = more recent)
  1 + DATE_DIFF('2021-01-16', MAX(PARSE_DATE('%Y%m%d', event_date)), DAY) AS recency_days,
  -- Static attributes
  MAX(device.category)                   AS device_type,
  MAX(traffic_source.source)             AS traffic_source,
  MAX(geo.country)                       AS country,
  -- LABEL: did user purchase in window?
  COUNTIF(event_name = 'purchase') > 0   AS converted
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210115'
GROUP BY user_pseudo_id;
