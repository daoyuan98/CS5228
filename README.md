final project for CS5228



## Result Log

Train:Valid = 19/20 * train dataset && 1/20 * train dataset

NT = not test



| model                           | acc on valid | acc on leaderboard | date  |
| ------------------------------- | ------------ | ------------------ | ----- |
| lr+svm+random forest            | 80.81        | 80.02              | 03-12 |
| random forest(depth=14)*1       | 85.99        | 86.13              | 03-13 |
| random forest(depth=14)*4       | 85.99        | NT                 | 03-13 |
| random forest(depth=12)*1       | 86.0~86.2    |                    | 03-13 |
| random forest(depth=12)*4[fill] | 86.32        | NT                 | 03-13 |
| random forest(depth=12)*4[drop] | 86.38        | NT                 | 03-13 |

