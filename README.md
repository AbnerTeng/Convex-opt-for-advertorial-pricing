# Creation Portfolio

## Build Virtual Environment

```plaintext
python -m venv your_venv_name
source your_venv_name/bin/activate
pip install -r requirements.txt
```

## Module Structure

```plaintext
root/
  |──src/
  |   |── __init__.py
  |   |── main.py
  |   |── opt_test_v3.py
  |   |── plot.py
  |   |── preproc.py
  |   |── property_scoring.py
  |   |── utils.py
  |   └── property_scoring.sql  
  |── data/
  |   |── param_mat/
  |   |── plotly_fig/
  |   |── opt_data_v2.csv
  |   |── preproc_data_v3.csv
  |   |── scoring_data_v3.csv
  |   |── use_data.csv 
  |   └── voice_data.csv/
  |── tests/
  |   |── __init__.py
  |   |── test_opt_test_v3.py
  |   |── test_preproc.py
  |   └── test_property_scoring.py
  |── .gitignore
  |── README.md
  |── .coveragerc
  |── pytest.ini
  |── requirements.txt
  └── run.sh (optional)
```

## Files

### preproc.py

> Data preprocessing file with below features

1. Data cleaning
2. Simple EDA
3. Impute missing values
4. Platform splitting

#### Execute command - `preproc.py`

```plaintext
python -m src.preproc --download <download or not>
```

Where

- `--download` is optional argument to download data to local.

### `utils.py`

> General utility functions for optimizatiom

### `opt_test_v3.py`

> Setting up functions and procedures for optimization

Be aware of the attribute `self.label_data` under `class Optimization`, it's only a data frame with column name `platform`.

### `property_scoring.py`

> Rank the KOL by two properties

1. Robustness of interact / view ratio
2. The experience of promotion

### `main.py`

> Main file to execute optimization procedure

#### Execute command - `main.py`

```plaintext

To execute the optimization procedure, I use shell script instead of python script to pass arguments.

```plaintext
chmod +x run.sh
./run.sh
```

Things inside `run.sh` are like

```bash
# echo "Optimization"
# python -m src.main --row 20 --type plain --budget 700 \
#     --voice 8000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "10 5 5 0") \
#     --spec_kols $(printf "%s" "7 10") \
#     --w1 0.167 --w2 0.833


echo "For pytest"
python -m src.main --row 10 --type plain --budget 1000 \
    --voice 5000 --lamb 0 --alpha 0 --post_cnt $(printf "%s" "1 1 1 1") \
     --spec_kols $(printf "%s" "1 2 3 4") \
     --w1 0.167 --w2 0.833


# echo "plot heatmap"
# python -m src.plot --file "all_20_plain.npy" --save True
```

- `--rows` is number of rows to use
- `--type` is type of objective function to use
- `--budget` is budget constraint of firm (thousand dollars)
- `--voice` is target voice of firm
- `--lamb` is lambda value for regularization methods
- `--alpha` is alpha value for ElasticNet
- `--post_cnt` constraints the minimum post count of specific post type
- `--spec_kol` constraints the minimum number of specific kol
- `--w1` the adjusted parameters for high weight constraints 
- `--w2` the adjusted parameters for low weight constraints
- `--candidates` number of candidates to choose

#### Hyper parameters

- `DEFAULT_K`: Degree of voice descent
- `self.weight`: weight of constraint functions

### `plot.py`

> Plotting heatmap of post_count, voice/price ratio and line char for voice_decreasing rate

#### Execute command - `plot.py`

```plaintext
python -m src.plot --file <file_name> --save <save fig or not>
```

Where

- `--file` is file name of optimization result matrix
- `--save` is optional argument to save figure or not

## Demo

Regarding the sophisticated optimization procedure, Here is a demo of some possible scenarios.

To execute the demo file, type:

```bash
chmod +x ./demo.sh
./demo.sh
```

Then you will see something like below:

```plaintext
Select demo scenario:
1. Scenario with plain objective function and budget + voice constraints
2. Scenario with regularized objective function and budget + voice constraints
3. Scenario with regularized objective function and budget + voice constraints + post constraints
4. Scenario with regularized objective function and budget + voice constraints + post constraints + KOL constraints
5. Scenario same as 4 but with unsatisfied constraints that need to adjust the weights.

Enter your choice:
```

Then you can choose one of the scenarios to execute, just type the number and press `Enter`.
