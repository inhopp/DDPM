# DDPM
Denoising Diffusion Probabilistic Models from scratch (pytorch)

> [Seminar 발표 자료]([Diffusion 1.pptx](https://github.com/inhopp/inhopp/files/13997580/Diffusion.1.pptx))


<br>

| Real | Generated |
|:-:|:-:|
| <img width="111" alt="화면 캡처 2024-01-20 193953" src="https://github.com/inhopp/inhopp/assets/96368476/2019be3d-0abe-4b80-8e85-c59652e4e70e"> | <img width="126" alt="download" src="https://github.com/inhopp/inhopp/assets/96368476/c2c281e6-b7e5-4b51-bd39-e7e5c7594632"> |


<br>


## Repository Directory 

``` python 
├── DDPM
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── utils.py
        ├── diffusion.py
        ├── requirments.txt
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `option.py` : Environment setting
- `model.py` : Define block and construct Model
- `train.py` : Train DDPM & Sampling outputs
- `utils.py` : ex) beta Scheduler, ...
- `diffusion.py` : Diffusion Forward/Backward Process

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/DDPM.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 30) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 256)
```



#### Main Reference
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=b02eb802