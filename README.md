# string-matching-rnn
- This project develops an AI algorithm to
match 2 strings

- We use Tensorflow to build a Bidirectional RNN model

## Note:
`model-1.keras`: 
- 2 Input -> 2 Embedding -> Concatenate -> Bi-lstm 1 -> Bi-lstm 2 -> Dense -> Output

- Epochs: 20, Batch size: 2048
- Dense's activation: Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam

## Installation
- Need Conda installed
- Run `conda env create -f environment.yml` to create the environment