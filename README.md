final project for CS5228



## Result Log

| modelf1                                                      | F1 on valid[fill nan with mean] | F1 on valid[drop nan] | date  |
| ------------------------------------------------------------ | ------------------------------- | --------------------- | ----- |
| lr+svm+random forest                                         | 84.64                           | 85.14                 | 05-06 |
| random forest(depth=14)*1                                    | 86.11                           | 86.34                 | 05-06 |
| random forest(depth=14)*4                                    | 85.87                           | 86.07                 | 05-06 |
| random forest(depth=12)*1                                    | 86.11                           | 86.07                 | 05-06 |
| random forest(depth=12)*4                                    | 85.87                           | 85.81                 | 05-06 |
| xgboost(n_estimator=220,  max_depth=3)                       | 87.59                           | 86.34                 | 05-06 |
| xgboost(n_estimator=210,  max_depth=3) + xgboost(n_estimator=220,  max_depth=3) + xgboost(n_estimator=230,  max_depth=3) | 87.71                           | 86.34                 | 05-06 |

