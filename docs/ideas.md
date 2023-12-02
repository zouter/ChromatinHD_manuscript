# Ideas

## Do we have less power?

It may be that using peaks is important as to maintain power. Perhaps our models have less power.
Of course, this is balanced with an increase in power outside/inside of peaks
Still, could we measure this somehow? Perhaps using synthetic data where we put in a real peak structure and check whether we need more cells to detect it as differential?

## Is regularization really required?

We use some Bayesian prior to regularize the diff model. But is this really necessary? So far, we only saw this qualitatively.

We have test cells available, so testing this should be easy.

## Easy regulatory network

Would it be possible to get an "easy" regulatory network from the data? Something like weighting different regions based on how predictive they are on a per-cell basis?

## Features of activation

Can we learn what feautures of active transcription are shared across genes?

There are several ways to do this:

### Purely accessibility

Train a model that predicts differences in gene expression based on **the fragments**, including the ones that are present without gene expression, and those that change between cells. The parameters are shared across genes
We can do train-test across genes within the same dataset, but also across datasets!

Perhaps this information could be used to create a "tandem" predictive model starting from enformer

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

I think this is by definition a convolution, i.e. P(X) = P(TF) * P(cut site), and we want P(TF)

Could we deconvolve to get P(TF)? This way we would be able to more accurately get where the protein binds, in single cells perhaps?

## Fragment size advanced

Is the ratio of TF footprint vs submono really the only thing that counts? Isn't there just a linear relationship between fragment length

Perhaps this relationship even continues into mono fragments => ambiguous meaning of these fragments. Could this ambiguous meaning be detected e.g. by also seeing smaller fragments if the gene is slightly lower expressed?

With larger fragment sizes, are the fragments "randomly" distributed or is there still a kind of footprinting signal?

## Beyond single-nucleus

Some mRNAs may be invisible, or diluted, in the nucleus, even though the gene is expressed. Could we include in some way single-cell data and integrate it with the single-nucleus to get better (differential) transcriptome measurements?

## Diff 2.0

Can we get a likelihood of not only individual cut sites, but also of all cut sites together, forming a fragment or different fragments within the same cell? Can we as such detect e.g. loss of co-accessibility due to a variant?

This would unify both co-accessibility and fragment size within one likelihood model.

This likelihood model could be used any time you want to *predict* accessibility, such as for sequence, variants, cell types, time, transcriptome, embedding, etc.

## Time

Can we create pred and diff but for time?

For pred, you would use a deltatime, that could be random, and select different sets of cells for prediction (-> fragments) and output (-> transcriptome).

Alternatively, we can look at the forcasting literature to see how they do this. For example a typical LSTM architecture could actually work, as long as we are able to embed all information from a particular time step.

Different questions we could try to answer:

- What is the distribution of information content of fragments. Likely the one at the time point itself are most informative
- Can we built a across-genes model, to learn whether these features are shared across genes. What is the earliest you could start predicting that a gene is going to be upregulated
- Does a non-peak approach detect more than a peak approach. We will need to implement a peak loader in that case.
