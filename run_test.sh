echo "For pytest"
python -m src.main --row 10 --type plain --budget 1000 \
    --voice 5000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "1 1 1 1") \
     --spec_kols $(printf "%s" "1 2 3 4") \
     --w1 0.167 --w2 0.833 --candidate 100

