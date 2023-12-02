<p align="center">
  <img src="image/README/1701262792359.png" width="200"/>
</p>

# Creation Portfolio

Creation Portfolio is a project for optimizing the KOL selection problem. This project is based on [scipy](https://www.scipy.org/) minimize method.

Creation Portfolio is primarily designed to provide effective, robust and explainable optimization architecture for firms who want to select promotion KOLs on social media platforms for their products.

## ⚙️ Build Virtual Environment

We recommend you to build a virtual environment to run this project.

```plaintext
python -m venv your_venv_name
source your_venv_name/bin/activate
pip install -r requirements.txt
```

## ⚒️ Module Structure

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
  |── clean.sh
  |── demo.sh
  |── requirements.txt
  └── run_test.sh (optional)
```

## 🔔 Key Features

### Optimization Procedures

The main goal of this project is to **minimizing** the cost of promotion, so we design a simple objective function to achieve this goal.

$$
\text{Obj function} = \text{minimize  }\mathbf{Cost}^\mathbf{T} \mathbf{Price}
$$

Where $\mathbf{Cost}$ is the cost of each post and $\mathbf{Price}$ is the price of each post.

Due to the issue of **over concentration** of post, we add penalty term to the objective function to split the post count of each KOL.

L1 Regularization (Ridge Regression)

$$
\text{Obj function} = \min\mathbf{Cost}^\mathbf{T} \mathbf{Price} + \lambda \cdot \mathbf{Count}^{|\cdot|}
$$

L2 Regularization (Lasso Regression)

$$
\text{Obj function} = \min\mathbf{Cost}^\mathbf{T} \mathbf{Price} + \lambda \cdot \mathbf{Count}^{2}
$$

ElasticNet

$$
\text{Obj function} = \min\mathbf{Cost}^\mathbf{T} \mathbf{Price} + \lambda \cdot \left(\alpha \cdot \mathbf{Count}^{|\cdot|} + \frac{(1-\alpha)}{2} \cdot   \mathbf{Count}^{2}\right)
$$

### Voice Decreasing Rate

To fit the real world situation, we've designed a voice decreasing rate function to simulate the voice decreasing rate of KOLs. In this project, we use the following function to simulate the voice decreasing rate.

$$
\text{Voice Decreasing Rate} = \frac{1 - e^{-\frac{n}{k}}}{1 - e^{-\frac{1}{k}}}
$$

  Where $n$ is the number of posts and $k$ is the degree of voice decreasing.

### Other Features

Working... writing document is so hard 😭

## To Execute

To execute the optimization procedure, I use shell script instead of python script to pass arguments.
Regarding the sophisticated optimization procedure, we suggest everyone to first exxecute the `demo.sh` to see the possible scenarios.

> Noticed that you need to change the execution permission of all `.sh` files by typing `chmod +x *.sh` in your terminal.

```plaintext
chmod +x demo.sh
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

Now, you can choose one of the scenarios to execute, just type the number and press `Enter`. Then you will see the result of optimization procedure like below:

```plaintext
Output Cost: 665.0 (Thousand dollars)
Output Voice: 6031.192183401445
   platform live_count post_count short_count vid_count live_price post_price short_price vid_price live_voice  post_voice short_voice    vid_voice
0        fb        0.0        0.0         0.0       0.0       25.0       46.0       187.0     250.0        0.0    5.917391         0.0  1194.974967
1        fb        0.0        1.0         0.0       0.0       11.0       10.0       144.0     113.0        0.0   120.45432         0.0   741.681559
2        fb        0.0        1.0         0.0       0.0       11.0        8.0       215.0     105.0        0.0  132.596772         0.0          0.0
3        fb        0.0        0.0         0.0       0.0        5.0        2.0       284.0      93.0        0.0   35.593277         0.0     0.999999
4        fb        0.0        0.0         0.0       0.0       99.0      120.0       202.0     368.0        0.0   281.29993         0.0          0.0
5        fb        0.0        0.0         0.0       0.0       17.0       21.0       203.0     151.0        0.0   78.353186         0.0    38.949981
============== Optimize Done, output information ===============
Selected rows: 20 rows
Objective function types: l2
Budget Constraint: 700 (Thousand Dollars)
Target Voice: 6000
Regularization lambda: 0.5
Minimum count: Live: 5, Post: 7, Short: 5, Vid: 7
Choosed KOLs: KOL ['1', '2', '7', '8', '9']
```

## To Test

To test the optimization procedure, we use `pytest` to test the functions in `src/` folder. To execute the test, you can simply execute the shell script `run_test.sh` in the root directory.

```plaintext
chmod +x run_test.sh
./run_test.sh
```
