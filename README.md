All the materials available in this document are to reproduce the results published in the following article:

> P. G. Jurado, X. Liang and S. Chatterjee, "Deterministic Transform Based Weight Matrices for Neural Networks," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 4528-4532, doi: 10.1109/ICASSP43922.2022.9747256.

The code permits to build four different neural network architectures (ELM, RVFL, dRVFL and SSFN).
The code is organized as follows:

- main.py: Govern to construct a specified neural network along a defined number of monte-carlo iterations.
- ELM.py and ELM_rand: Build ELM architecture, either using DTs or Randomized matrices.
- dRVFL.py and dRVFL_rand: Build dRVFL architecture, either using DTs or Randomized matrices. If "l_max" (layers) equals 1, RVFL is built. 
- SSFN.py and SSFN: Build SSFN architecture, either using DTs or Randomized matrices.
- MyOptimizers.py: Contain least-square and ADDM-least-square alogirthms to solve optimization problems for the output matrices.
- MyFunctions.py: Define different functions used along the all code files.
- make_dataset_helper.py: Make dataset for the experiments.
- DeterministicTransforms.py: Build different DeterministicTransforms not included in python.
- layer_iteration.py: Contain functions for the proposed learning scheme. Computes the different implemented deterministic transforms and also the two different methods to chose the transform. 

In "mat_files" folder, you find the used datasets in our experiments. 
This folder must be placed in the same directory as the codes.   


### Basic Usage
For example, in order to implement SSFN on Vowel dataset based on the parameters TABLE Ⅱ shows, execute the following command.   
```python main.py --data vowel --neural_net SSFN --rand_net No --n_DT 11 --MC_it 20 --alpha 2 --mu 1e3 --lam 1e1 --k_max 100 --n1 1000 --l_max 20 --method 1 --thr_var 1e-7 --eta_l 0.1 --eta_n 0.005 --delta 50```

It is also possible to execute the above command using the default argument like as follows.   
```python main.py --data vowel --lambda_ls 100 --myu 1000 --learning_rate 0.000001```

Not all the input parameters are used in the all the networks. Here are the parameters that are needed for each architecture, therefore the others can be ignored when executing:
- ELM: data, neural_net, rand_net, n_DT, MC_it, alpha, mu, k_max, n1 (when Rand), method, thr_var
- dRVFL: data, neural_net, rand_net, n_DT, MC_it, alpha, mu, k_max, n1 (when Rand), l_max, method, thr_var
- RVFL: same as dRVFL, but l_max must be "1"
- SSFN: data, neural_net, rand_net, n_DT, MC_it, alpha, mu, lam, k_max, n1 (when Rand), l_max, method, thr_var, eta_l, eta_n (when Rand), delta (when Rand)


### Options 
You can check out the options of the input parameters and call the function by using:   
```python main.py --help```


### Experiments Reproducibility
The tuned parameters used to obtain the results presented in the article for deterministic transforms case, are available in the document "Parameters_tuned_DT.pdf".


########################################################################################################################
########################################################################################################################
%
%   Contact:    Pol Grau Jurado (polgj@kth.se), Saikat Chatterjee (sach@kth.se)
%
% 	September 2021
