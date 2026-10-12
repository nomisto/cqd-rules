Testscript for the CQD^{Rule}

The start command is ```python all.py {dataset} {querytype} {k} {maxplus} {tnorm} {split}```

Valid values are
```
dataset = ["FB15k-237", "FB15k", "nell"]
querytype = see paper, e.g. 2i, 2u, 2p, 2in, ...
k = int, number of k for beam search
maxplus = bool (0/1), if maxplus should be applied (should be mostly set to 1)
tnorm = ["prod", "godel"]
split = ["test", "valid"]
```

e.g. `python all.py FB15k-237 2i 15000 1 prod test`