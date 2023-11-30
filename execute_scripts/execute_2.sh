python -m src.main --do_preproc False --dat_name preproc_data_v4 --download True --row 20 --type l2 --budget 700 \
    --voice 8000 --lamb 0.5 --alpha 0 --post_cnt $(printf "%s" "0 0 0 0") \
    --spec_kols $(printf "%s" "None") \
    --w1 0.167 --w2 0.833 --candidate 100

python -m src.plot --file all_20_l2.npy --save False