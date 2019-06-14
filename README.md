# chainer-centernet

This is Centernet implementation on chainer(and chainercv). (original https://github.com/xingyizhou/CenterNet)

# How to install

We using to Pipenv, please install pipenv.

```sh
pipenv sync
```

# demo

## download

please download [here](https://drive.google.com/file/d/1TGRqdgyrWwJC0PXG91GD7e6t6zw9GLLv/view?usp=sharing).  
And put here `models/hg_256_pascalvoc.npz`.

## run

```sh
cd chainer-centernet
PYTHONPATH="." python scripts/visualize.py
```

# Train

TBD
