WITH raw_data AS (
    SELECT
        *
    FROM
        {{ source('soil_moisture_data', 'soil_moisture') }}
),

cleaned_data AS (
    SELECT
        N::INT as nitrogen,
        P::INT as phosphorous,
        K::INT as potassium,
        ROUND(TEMPERATURE::FLOAT, 2) as temperature,
        ROUND(HUMIDITY::FLOAT, 2) as humidity,
        ROUND(PH::FLOAT, 2) as ph,
        ROUND(SOIL_MOISTURE::FLOAT, 2) as soil_moisture,
        ROUND(RAINFALL::FLOAT, 2) as rainfall
    FROM
        raw_data
)

SELECT
    *
FROM
    cleaned_data