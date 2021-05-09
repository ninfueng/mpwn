# Python section

## Requirements:
```
- python == 3.8.2
- numpy == 1.19.1
- torch == 1.6.0
- torchvision == 0.7.0 
- tensorboard == 2.2.0
- loguru == 0.5.0
- hyperopt == 0.2.3
- tqdm == 4.46.1
- pandas == 1.0.3
- scipy == 1.5.4
- seaborn == 0.10.1
- matplotlib == 3.2.2
- scipy == 1.5.2 
- prefetch_generator == 1.0.1
```

## To use Fashion MNIST:

Run a preferred model. The flag `-l1` to `-l5` indicates type of layer that can be in set of `{f, t ,b}`. For example:
```bash
python main_grid_search.py -l1 f -l2 f -l3 f -l4 f -l5 f
```

To perform a grid search for all possible models.
```bash
bash grid-search.sh
```

To perform Bayesian optimization search.
```bash
python main_baysian.py
```

