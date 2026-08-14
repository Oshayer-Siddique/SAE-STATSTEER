# SAE-STATSTEER

[![arXiv](https://img.shields.io/badge/arXiv-2508.00079-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2607.19364)
[![PDF](https://img.shields.io/badge/Paper%20PDF-EF3939?style=flat&logo=readme&logoColor=white&color=gray&labelColor=ec1c24)](https://arxiv.org/pdf/2607.19364)

SAE-StatSteer introduces a statistics-based method for the generation of steering vectors for LLMs. We test our technique on four domains: Morality, Logic, Sentiment, and Political polarity.

This repository contains all the code used for the paper SAE-StatSteer.

## What is LLM steering

When LLMs generate answers, they pass values from one hidden layer to the next. We can artificially add some vector to these values that are being passed. This changes the models behaviour.

This method of guiding the model's behaviour is called LLM steering.

## What are SAE (Sparse Autoencoders)

One way to generate a steering vector that modifies the model's behaviour in the desired way is to find out the concept encoded in each neuron of a hidden layer. The issue is that neurons are polysemantic, i.e. each neuron encodes more then one concept.

SAE are models that can take a hidden layer and encode it into a sparse matrix, where each value is monosemantic i.e. encodes a simple concept. This makes it much easier to generate a steering vector that does what we want it to do.

More information can be found at [neuronpedia.org](https://www.neuronpedia.org/).

## Our contribution

To select relevant features for forming the steering vector, we first filter features through six reliability conditions, then ranks the survivors by an unweighted Borda consensus over three statistics, an F-test, KSG mutual information, and Cohen's d, and finally combine the selected SAE decoder rows using Cohen's-d weights. 

> We used Google TPUs for the generation of Sparse Autoencoders, and had to use JAX instead of Pytorch.
> We installed Gemma-2-2b, Gemma-2-9b, and Gemma-3-4b and ran them on coommercially available GPUs during the evaluation phase of our experiment.

