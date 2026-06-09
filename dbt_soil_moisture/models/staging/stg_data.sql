WITH raw_data AS (
    SELECT
        *
    FROM
        {{ source('soil_moisture_data', 'soil_moisture') }}
),

cleaned_data AS (
    SELECT
        time,
        ATMOSPHERIC_TEMP::FLOAT as atmospheric_temp,
        SOIL_TEMP::FLOAT as soil_temp,
        HUMIDITY::FLOAT as humidity,
        RAINFALL::FLOAT as rainfall,
        LAG(SOIL_MOISTURE::FLOAT) OVER (ORDER BY time) as previous_soil_moisture,
        SOIL_MOISTURE::FLOAT as soil_moisture,
        LEAD(SOIL_MOISTURE::FLOAT) OVER (ORDER BY time) as future_soil_moisture
    FROM
        raw_data
    WHERE
        ATMOSPHERIC_TEMP IS NOT NULL
        AND SOIL_TEMP IS NOT NULL
        AND HUMIDITY IS NOT NULL
        AND SOIL_MOISTURE IS NOT NULL
    ORDER BY
        time
)

SELECT
    *
FROM
    cleaned_data
WHERE
    previous_soil_moisture IS NOT NULL
    AND future_soil_moisture IS NOT NULL
