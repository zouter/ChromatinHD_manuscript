# Ideas

## Features of activation

Can we learn what feautures of active transcription are shared across genes?

There are several ways to do this:

### Purely accessibility

Train a model that predicts differences in gene expression based on **the fragments**, including the ones that are present without gene expression, and those that change betweenw cells. The parameters are shared across genes
We can do train-test across genes within the same dataset, but also across datasets!

Perhaps this information could be used to create a "tandom" predictive model starting from enformer

### Sequence

Train a model that predicts differences in gene expression based on the **sequence around the fragments**.
It is an extension of the previous model, but with an enformer-esque approach. However, we should try the previous model first, as that will be used as baseline for the sequence model.

You could even parameterize the sequence model using the gene expression of the particular cell, to make it cell-type specific

## Improved co-predictivity, chromatin modules on steroids

There are countless improvements possible:

- Can we phase the genome purely based on accessibility? If two fragments are present in the same cell, we could hypothesize that they will be from the same chromosome => you can phase the regions that are open. Of course, it's not going to be perfect, but there should be some linkage disequilibrium! The phased genome could then be used to better learn co-predictivity from the same chromosome
- Can we create a better way to determine co-predictivity across windows where we don't have to do the arbitrary 1kb cutoff? There should be a way to discard shared fragments, right?