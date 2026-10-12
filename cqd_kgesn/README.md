Implementation based on https://github.com/EdinburghNLP/adaptive-cqd

## Instructions

```bash
PYTHONPATH=. python cli/cqda-cli.py --checkpoint_path "../models/nell-betae/" --do_test --data_path ../data/betae/NELL-betae --negative_sample_size 1 --hidden_dim 1000 --gamma 12.0 --batch_size 100 --test_batch_size 1 --learning_rate 0.0001 --cpu_num 0 --max_steps 101000 --warm_up_steps 9999999 --save_checkpoint_steps 50000 --valid_steps 5 --log_steps 500 --test_log_steps 1000 --geo cqd --print_on_screen --optimizer adam --cqd-softmax-normalize-scores --cqd-type discrete --cqd-t-norm min --cqd-k 15000 --cqd-test-k 15000 --tasks 2i --seed 0 --beta_mode "(1600,2)" --box_mode "(none,0.02)" --prefix None --evaluate_union DNF --cuda
```

Do various combinations of:

```
--cqd-t-norm min
--cqd-t-norm prod
--cqd-softmax-normalize-scores
--cqd-minmax-normalize-scores
--cqd-sigmoid-normalize-scores
```

For validation set `--do_valid` instead of `--do_test`. 
Also for different tasks set the corresponding template, e.g. `--tasks 2i`