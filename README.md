# Time-of-arrival estimation and phase unwrapping of HRTF with integer linear programming

Source code of the papers [_Time-of-arrival estimation and phase unwrapping of head-related transfer functions with integer linear programming_]() (AES Convention 156) and [_Arbitrarily Sampled Signal Reconstruction Using Relative Difference Features_]() (GSP Workshop 2024).


## Getting started

You'll need to have Python >=3.11 installed on your system. 
```bash
pip install -r requirements.txt
```

## Preprocessing

Given a single HRTF sofa file, `preprocess.py` will compute its TOAs using 36 different configurations we stated in the paper and store the results into a folder.
    
```bash
python preprocess.py input.sofa output_folder --toa-weight 0.1 --oversampling 10
```
Here, `--toa-weight` controls the value of $\lambda$ and $w_{\delta, i}$ of exponential weighting in the paper.
The output npz files has the format of `{edgeslist/ilp/l2}_toa_{True/False}_cross_{True/False}_{angle/dot/none}.npz`.
They corresponds to the terms in the paper as follows:

| Name | Name in ther paper |
|:------:|:--------------------:|
| edgeslist | EDGY |
| ilp | SIMP |
| l2 | LS |
| toa_True | w/ Min. |
| toa_False | w/o Min. |
| cross_True | w/ Cross |
| cross_False | w/o Cross |
| angle | EXP |
| dot | CORR |
| none | NONE |


## Experiments

### ITD/aligned-HRIRs reconstruction

Please create a folder named `processed` and put the preprocessed directories in it.
Then, execute the notebook [visualise](visualise.ipynb) to see the results.