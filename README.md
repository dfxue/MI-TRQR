# MI-TRQR
This repository provides the official implementation of NlPS 2025 paper [MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks](https://openreview.net/forum?id=NRqGpUAjV9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2025%2FConference%2FAuthors%23your-submissions)).

```bibtex
@inproceedings{xue2025mitrqr,
  title={MI-TRQR: Mutual Information-Based Temporal Redundancy Quantification and Reduction for Energy-Efficient Spiking Neural Networks},
  author={Xue, Dengfeng and Li, Wenjuan and Lu, Yifan and Yuan, Chunfeng and Liu, Yufan and Liu, Wei and Yao, Man and Yang, Li and Li, Guoqi and Li, Bing and Maybank, Stephen and Hu, Weiming and Li, Zhetao},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

# Code and Prerequisites
This code is based on the [PSN](https://github.com/fangwei123456/Parallel-Spiking-Neuron) codebase.

The added code implements the MI-TRQR module, which is a plug-and-play module to quantify and reduce temporal redundancy, at the cost of zero parameters. 
The implementation need the [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) package.

Here we provide a sample of quantify the redundancy between two binary spiking features: S1 and S2.

```python
import torch
from torchmetrics.functional.clustering import mutual_info_score

# Note: Both S1 and S2 must be int type.
S1=S1.int()
S2=S2.int()
mi_value = mutual_info_score(S1, S2) # output: tensor(0.5004)
```
