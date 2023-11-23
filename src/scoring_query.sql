WITH post_voice AS (
        SELECT
            *,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_1m,
                    total_post_sum_1m
                ),
                0
            ) AS avg_post_voice_1m,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_2m,
                    total_post_sum_2m
                ),
                0
            ) AS avg_post_voice_2m,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_3m,
                    total_post_sum_3m
                ),
                0
            ) AS avg_post_voice_3m,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_4m,
                    total_post_sum_4m
                ),
                0
            ) AS avg_post_voice_4m,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_5m,
                    total_post_sum_5m
                ),
                0
            ) AS avg_post_voice_5m,
            IFNULL(
                SAFE_DIVIDE(
                    social_vol_6m,
                    total_post_sum_6m
                ),
                0
            ) AS avg_post_voice_6m
        FROM
            'ikala-ai-compute.ds_intern.creation_profolio_test_data_vol3'
    ),
    min_max AS (
        SELECT
            *,
            MAX(avg_post_voice_1m) OVER() AS max_avg_post_voice_1m,
            MIN(avg_post_voice_1m) OVER() AS min_avg_post_voice_1m,
            MAX(avg_post_voice_2m) OVER() AS max_avg_post_voice_2m,
            MIN(avg_post_voice_2m) OVER() AS min_avg_post_voice_2m,
            MAX(avg_post_voice_3m) OVER() AS max_avg_post_voice_3m,
            MIN(avg_post_voice_3m) OVER() AS min_avg_post_voice_3m,
            MAX(avg_post_voice_4m) OVER() AS max_avg_post_voice_4m,
            MIN(avg_post_voice_4m) OVER() AS min_avg_post_voice_4m,
            MAX(avg_post_voice_5m) OVER() AS max_avg_post_voice_5m,
            MIN(avg_post_voice_5m) OVER() AS min_avg_post_voice_5m,
            MAX(avg_post_voice_6m) OVER() AS max_avg_post_voice_6m,
            MIN(avg_post_voice_6m) OVER() AS min_avg_post_voice_6m
        FROM
            post_voice
    ),
    scaled AS (
        SELECT
            *,
            SAFE_DIVIDE(
                avg_post_voice_1m - min_avg_post_voice_1m,
                max_avg_post_voice_1m - min_avg_post_voice_1m
            ) AS scaled_avg_post_voice_1m,
            SAFE_DIVIDE(
                avg_post_voice_2m - min_avg_post_voice_2m,
                max_avg_post_voice_2m - min_avg_post_voice_2m
            ) AS scaled_avg_post_voice_2m,
            SAFE_DIVIDE(
                avg_post_voice_3m - min_avg_post_voice_3m,
                max_avg_post_voice_3m - min_avg_post_voice_3m
            ) AS scaled_avg_post_voice_3m,
            SAFE_DIVIDE(
                avg_post_voice_4m - min_avg_post_voice_4m,
                max_avg_post_voice_4m - min_avg_post_voice_4m
            ) AS scaled_avg_post_voice_4m,
            SAFE_DIVIDE(
                avg_post_voice_5m - min_avg_post_voice_5m,
                max_avg_post_voice_5m - min_avg_post_voice_5m
            ) AS scaled_avg_post_voice_5m,
            SAFE_DIVIDE(
                avg_post_voice_6m - min_avg_post_voice_6m,
                max_avg_post_voice_6m - min_avg_post_voice_6m
            ) AS scaled_avg_post_voice_6m
        FROM min_max
    )
SELECT
    *,
    SQRT(
        POW(
            scaled_avg_post_voice_1m - mean,
            2
        ) + POW(
            scaled_avg_post_voice_2m - mean,
            2
        ) + POW(
            scaled_avg_post_voice_3m - mean,
            2
        ) + POW(
            scaled_avg_post_voice_4m - mean,
            2
        ) + POW(
            scaled_avg_post_voice_5m - mean,
            2
        ) + POW(
            scaled_avg_post_voice_6m - mean,
            2
        )
    ) AS voice_stdev
FROM (
        SELECT
            *,
            SAFE_DIVIDE(
                scaled_avg_post_voice_1m + scaled_avg_post_voice_2m + scaled_avg_post_voice_3m + scaled_avg_post_voice_4m + scaled_avg_post_voice_5m + scaled_avg_post_voice_6m,
                6
            ) AS mean
        FROM scaled
    )