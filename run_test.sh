# echo "Optimization"
# python -m src.main --row 20 --type plain --budget 700 \
#     --voice 8000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "10 5 5 0") \
#     --spec_kols $(printf "%s" "7 10") \
#     --w1 0.167 --w2 0.833 --candidate 100


echo "For pytest"
python -m src.main --row 10 --type plain --budget 1000 \
    --voice 5000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "1 1 1 1") \
     --spec_kols $(printf "%s" "1 2 3 4") \
     --w1 0.167 --w2 0.833 --candidate 100
# echo "plot heatmap"
# python -m src.plot --file "all_20_plain.npy" --save True

