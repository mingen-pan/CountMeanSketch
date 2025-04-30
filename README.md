# Optimized Count Mean Sketch with Randomized Response (OCMS+RR)

## Introduction

This repo stores the code related to the paper *Improving Count-Mean Sketch as the Leading Locally Differentially Private Frequency Estimator for Large Dictionaries*, which is published in the 2025 IEEE 38th Computer Security Foundations Symposium (CSF).

OCMS+RR is the only leading algorithm (so far) for reducing the worst-case MSE and l_1 / l_2 losses when processing datasets with very large dictionaries. A dictionary represents all the possible values of the objects in a dataset.


## Code Structure

`ocms_rr.py`: contains the server and client implementation of OCMS+RR.

`evaluation`: contains all the codes to reproduce the evaluation section of the paper

`evaluation/sketch_experiment.py`: the entry point of the evaluation code.

`cms_reconstruction_example.py`: The example of Appendix A in the paper.

## Example

The code snippet below builds the server and cliengt of CMS+RR optimized for the worst-case MSE, and processes a dataset.
```python
from ocms_rr import build_ocms_rr_optimized_for_worst_case_mse

original_objects = [0] * 4000 + [25] * 3000 + [50] * 2000 + [75] * 1000
dict_size = 1000000
epsilon = 3

server, client = build_ocms_rr_optimized_for_worst_case_mse(epsilon, dict_size)
for original_object in original_objects:
    perturbed_value, hash_param = client.perturb(original_object)
    server.receive(perturbed_value, hash_param)
estimated_frequencies = server.batch_query(list(range(100)))
print(estimated_frequencies[[0, 25, 50, 75, 99]])
```

One could refer to the `__main__` section in `ocms_rr.py` for more examples. To run the the `__main__` section, you can do

(assume you are in the same directory as `ocms_rr.py`)

```shell
python3 ocms_rr.py
```


## Evaluation

To reproduce the evaluation results, one could run

```shell
# enter eval directory if you are not there
cd eval
python3 sketch_experiment.py
```
