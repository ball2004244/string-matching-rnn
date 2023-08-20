# string-matching-rnn
- This project develops an AI algorithm to
match 2 strings

- We use Tensorflow to build a Bidirectional RNN model

## Note:
`model-1.keras`: 
- 2 Input -> 2 Embedding -> Concatenate -> Bi-lstm 1 -> Bi-lstm 2 -> Dense -> Output

- Epochs: 10, Batch size: 256
- Dense's activation: Sigmoid
- Loss: Binary Crossentropy

`model-2.keras`:
- 2 Input -> 2 Embedding -> Concatenate -> Bi-lstm 1 -> Bi-lstm 2 -> Dense -> Output

- Epochs: 10, Batch size: 32
- Dense's activation: Softmax
- Loss: Binary Crossentropy
