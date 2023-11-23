python -m src.main --row 20 --type l2 --budget 700 \
    --voice 8000 --lamb 0.5 --alpha 0 --post_cnt $(printf "%s" "5 7 5 7") \
    --spec_kols $(printf "%s" "1 2 7 8 9") \
    --w1 0.167 --w2 0.833 --candidate 100


python -m src.plot --file all_20_l2.npy --save False