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

## Beyond Enformer

A key question is how we could use all of these models to improve sequence-to-expression prediction. One key thing here is that the multiome data may give insights into how accessibility affects gene expression, so if we would be able to accurately predict accessibility from sequence, it might help predict gene expression.

We could try a model stacking approach: predict accessibility from sequence, then predict expression from accessibility. This could be expanded with a direct model.

## Deconvolving other Tn5 techniques

Other Tn5 techniques are super scalable as well to profile local protein binding, particularly given that they can be "easily" combined with a single experiment using different barcodes.

However, the Tn5 insertion propensity will be correlation with the local accessibility. Meaning that we do not actually see where a protein binds, but rather a smoothened variant of protein binding (probably correlated with DNA proximity?) combined with the insertion site probability.

I thin this is by definition a convolution, i.e. P(X) = P(TF) * P(cut site), and we want P(TF)

Could we deconvolve to get P(TF)? This way we would be able to more accurately get where the protein binds, in single cells perhaps?