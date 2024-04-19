# Time-of-arrival estimation and phase unwrapping of HRTF with integer linear programming

Source code of the papers [_Time-of-arrival estimation and phase unwrapping of head-related transfer functions with integer linear programming_]() (AES Convention 156) and [_Arbitrarily sampled signal reconstruction using relative difference features_](pdf/gsp_paper.pdf) (GSP Workshop 2024).


## Getting started

Please install the required packages by running the following command.
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
Then, execute the notebook [eval](eval.ipynb) to evaluate the estimated TOAs.

### Noise robustness

Please execute the notebook [noise-robust](noise-robust.ipynb) to see the results.
The notebook includes the code to generate Figure 2 in the paper.
Changing the sofa file to the one you want to test will generate the results for the given HRTF.

### Phase unwrapping

Please execute the notebook [phase-unwrapping](phase-unwrapping.ipynb) to see the results.
The notebook includes the code to generate Figure 4 and 5 in the paper.


## Citation

Coming soon.