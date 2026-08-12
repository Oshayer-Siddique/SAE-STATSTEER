# steered_inference

This is the codebase used to test the performance of LLMs after steering using the stat_steer method.

All the steering vectors used while generating inferences are contained inside the ```vectors``` folder.

The ```prompts``` folder contains all the incomplete prompts from four domains which the LLMs had to complete.
 
Politic prompts were adapted from the ```twinviews-13k``` dataset:
```
@inproceedings{fulayRelationshipTruthPolitical2024,
  author       = {Fulay, Suyash and Brannon, William and Mohanty, Shrestha and Overney, Cassandra and Poole-Dayan, Elinor and Roy, Deb and Kabbara, Jad},
  title        = {On the Relationship between Truth and Political Bias in Language Models},
  booktitle    = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP '24)},
  year         = {2024},
  month        = nov,
  publisher    = {Association for Computational Linguistics},
  note         = {arXiv:2409.05283},
  abstract     = {Language model alignment research often attempts to ensure that models are not only helpful and harmless, but also truthful and unbiased. However, optimizing these objectives simultaneously can obscure how improving one aspect might impact the others. In this work, we focus on analyzing the relationship between two concepts essential in both language model alignment and political science: \textit{truthfulness} and \textit{political bias}. We train reward models on various popular truthfulness datasets and subsequently evaluate their political bias. Our findings reveal that optimizing reward models for truthfulness on these datasets tends to result in a left-leaning political bias. We also find that existing open-source reward models (i.e. those trained on standard human preference datasets) already show a similar bias and that the bias is larger for larger models. These results raise important questions about both the datasets used to represent truthfulness and what language models capture about the relationship between truth and politics.}
}
```
Logic prompts were adapted from ```Logicbench``` dataset:
```
@article{parmar2024towards,
  title={Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models},
  author={Parmar, Mihir and Patel, Nisarg and Varshney, Neeraj and Nakamura, Mutsumi and Luo, Man and Mashetty, Santosh and Mitra, Arindam and Baral, Chitta},
  journal={arXiv preprint arXiv:2404.15522},
  year={2024}
}
```
Moral and Sentiment prompts were synthetically generated.
