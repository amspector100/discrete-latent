# Discrete-Latent

## Tests

To run unit-tests, run 'python -m unittest' in the parent directory (discrete-latent).

If you test a class you're building, please add that test to the test folder (it will save a lot of time)! It shouldn't take any time since we'll have to test the stuff we build either way.

## Structure

The big-picture structure is as follows:
- encoders.py contains encoder classes
- decoders.py contains decoder classes
- latents.py contains classes which handle the transfer of 
information between the encoder/decoder (e.g. the VQVAE class)
- vae.py contains the master class which wraps the encoder,
decoder, and latent classes
- training.py contains functions for training the vae
- data.py handles loading/processing data

There are also a variety of utilities scattered around (kmeans, weight dropout, gumbel softmax sampling).

## To-Do

- Play with norm parameter ('l2' or 'softmax') in gumbel softmax