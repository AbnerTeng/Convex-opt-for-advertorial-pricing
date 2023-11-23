python -m src.main --row 20 --type plain --budget 700 \
    --voice 8000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "5 7 5 7") \
    --spec_kols $(printf "%s" "None") \
    --w1 0.167 --w2 0.833 --candidate 100

python -m src.plot --file all_20_plain.npy --save False