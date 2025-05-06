-- File: etl/queries/user_features.sql
-- Description: Constructs a user-level feature table for conversion prediction.
-- Source: bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
-- Date range: 2021-01-01 to 2021-01-15

SELECT
  user_pseudo_id,
  -- Count of page views
  COUNTIF(event_name = 'page_view') AS num_pageviews,
  -- Count of add-to-cart events
  COUNTIF(event_name = 'add_to_cart') AS num_addtocart,
  -- Count of session starts
  COUNTIF(event_name = 'session_start') AS num_sessions,
  -- Label: whether purchase occurred
  COUNTIF(event_name = 'purchase') > 0 AS converted,
  -- Most frequent device category
  MAX(device.category) AS device_type,
  -- Most frequent traffic source
  MAX(traffic_source.source) AS traffic_source,
  -- Most frequent country
  MAX(geo.country) AS country
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210115'
GROUP BY user_pseudo_id;