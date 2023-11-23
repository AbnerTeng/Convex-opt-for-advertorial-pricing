#!/bin/bash

echo "######################################################################"
echo "#                  Demo script for Creation Portfolio                #"
echo "######################################################################"

echo "In this demo, we will show some possible scenarios for optimizing the promoting post portfolio from simple to complex."
echo "We will show the result of the portfolio in the form of heatmap."

echo "######################################################################"
echo "#                   Demo 1: Simplest scenario                        #"
echo "######################################################################"

read -p "
Select demo scenario: 
1. Scenario with plain objective function and budget + voice constraints
2. Scenario with regularized objective function and budget + voice constraints
3. Scenario with regularized objective function and budget + voice constraints + post constraints
4. Scenario with regularized objective function and budget + voice constraints + post constraints + KOL constraints
5. Scenario same as 4 but with unsatisfied constraints that need to adjust the weights.

Enter your choice: " scenario

echo

echo "######################################################################"
echo "#                Change mode for execute scripts                     #"
echo "######################################################################"
chmod +x execute_scripts/*.sh
echo 

if [ "$scenario" == 1 ]; then
    echo "Demo 1: Scenario with plain objective function and budget + voice constraints"
    echo 
    echo "In this demo, we will show the result of the portfolio with plain objective function and budget + voice constraints."
    echo "The portfolio will be optimized with the following parameters:"
    echo "    - Number of KOLs choosed: 20"
    echo "    - Obj function: plain"
    echo "    - Budget: 700"
    echo "    - Voice: 8000"
    echo "    - Post constraints: 0 0 0 0"
    echo "    - KOL constraints: None"
    echo "    - Weights: 0.167 0.833"
    echo "    - Candidate: 100"
    echo "The result will be saved in the file: all_20_plain.npy"
    ## echo "The heatmap will be saved in the file: heatmap_10_plain.png"
    execute_scripts/execute_1.sh


elif [ "$scenario" == 2 ]; then
    echo "Demo 2: Scenario with regularized objective function (Ridge) and budget + voice constraints"
    echo "In this demo, we will show the result of the portfolio with regularized objective function (Ridge) and budget + voice constraints."
    echo "The portfolio will be optimized with the following parameters:"
    echo "    - Number of KOLs choosed: 20"
    echo "    - Obj function: L2 (Ridge)"
    echo "    - Budget: 700"
    echo "    - Voice: 8000"
    echo "    - Post constraints: 0 0 0 0"
    echo "    - KOL constraints: None"
    echo "    - Weights: 0.167 0.833"
    echo "    - Candidate: 100"
    echo "The result will be saved in the file: all_20_l2.npy"
    ## echo "The heatmap will be saved in the file: heatmap_10_reg.png"
    execute_scripts/execute_2.sh

elif [ "$scenario" == 3 ]; then
    echo "Demo 3: Scenario with plain objective function and budget + voice constraints + post constraints"
    echo "In this demo, we will show the result of the portfolio with plain objective function and budget + voice constraints + post constraints."
    echo "The portfolio will be optimized with the following parameters:"
    echo "    - Number of KOLs choosed: 20"
    echo "    - Obj function: plain"
    echo "    - Budget: 700"
    echo "    - Voice: 8000"
    echo "    - Post constraints: 5 7 5 7"
    echo "    - KOL constraints: None"
    echo "    - Weights: 0.167 0.833"
    echo "    - Candidate: 100"
    echo "The result will be saved in the file: all_20_plain.npy"
    ## echo "The heatmap will be saved in the file: heatmap_10_reg_post.png"
    execute_scripts/execute_3.sh

elif [ "$scenario" == 4 ]; then
    echo "Demo 4: Scenario with regularized objective function (Ridge) and budget + voice constraints + post constraints + KOL constraints"
    echo "In this demo, we will show the result of the portfolio with regularized objective function (Ridge) and budget + voice constraints + post constraints + KOL constraints."
    echo "The portfolio will be optimized with the following parameters:"
    echo "    - Number of KOLs choosed: 20"
    echo "    - Obj function: L2 (Ridge)"
    echo "    - Budget: 700"
    echo "    - Voice: 6000"
    echo "    - Post constraints: 5 7 5 7"
    echo "    - KOL constraints: 1 2 7 8 9"
    echo "    - Weights: 0.167 0.833"
    echo "    - Candidate: 100"
    echo "The result will be saved in the file: all_20_plain.npy"
    ## echo "The heatmap will be saved in the file: heatmap_10_reg_post_kol.png"
    execute_scripts/execute_4.sh

elif [ "$scenario" == 5 ]; then
    echo "Demo 5: Scenario same as 4 but with unsatisfied constraints that need to adjust the weights. In this case, we increase the budget from 6000 to 8000."
    echo "In this demo, we will show the result of the portfolio with regularized objective function (Ridge) and budget + voice constraints + post constraints + KOL constraints."
    echo "The portfolio will be optimized with the following parameters:"
    echo "    - Number of posts: 20"
    echo "    - Obj function: L2 (Ridge)"
    echo "    - Budget: 700"
    echo "    - Voice: 8000"
    echo "    - Post constraints: 5 7 5 7"
    echo "    - KOL constraints: 1 2 7 8 9"
    echo "    - Weights: 0.167 0.833"
    echo "    - Candidate: 100"
    echo "The result will be saved in the file: all_20_plain.npy"
    ## echo "The heatmap will be saved in the file: heatmap_10_reg_post_kol.png"
    execute_scripts/execute_5.sh

else
    echo "Invalid input. Please try again."
fi