# Generative adversarial networks

## Feedback about tips & tricks

* Normalize the inputs
* Modified loss function
* ...

## Feedback about GAN approach
i.e. how we let the GANS interact, optimization, which kind of GAN set-up (conditional, divergence, ...).

## Feedback about the networks

Convolutional, layers, dropout, ...

### Dropout
* Use 0.2 dropout probability
* Only on hidden layers to start with

-> Use dropout on visible layers
-> Halve weights at test-time (when dropout 0.5): takes geometric mean? Or does fraction depend on dropout probability


### Batchnorm
* Have used L2

-> Try (more robust?) median - quantile transformation?

## Questions/notes

Also about TensorFlow itself
 * Maybe multistart is less needed when dropout is applied.
