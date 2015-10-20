Attention-based encoder-decoder model for machine translation

## Training
Change the hard-coded paths to data in `nmt.py` then either run,

```
./run.sh 
```

or

```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```

Based on Kyunghyun Cho's tutorial code with some small modifications.
