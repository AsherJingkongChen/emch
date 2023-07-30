# Guide to the benchmark

## Sec 1. Environment
Before we start, let's check out the status of my machine:

### OS Version
```
Mac OS X 10.15.7 (Catalina)
Darwin Kernel Version: 19.6.0
```

### Hardware Information
```
Hardware Model: MacBookPro9,2
Installed Memory Size: 8GB
CPU name: Dual-Core Intel Core i5
CPU frequency: 2.5GHz
Physical CPU cores: 2
Logical CPU cores: 4
```

In this benchmark, I operate ML inference on Intel CPU only.

## Sec 2. Targets
I'd like to see how fast can a Sentence Trasformer model be.
There is only one task: Semantic search on [BelR/scifact dataset](https://huggingface.co/datasets/BeIR/scifact).
1. get all embeddings of queries (text) and corpus (title).
2. compute cosine similarity for all pairs of queries and corpus.
3. get the id of corpus with the highest similarity.
4. if the id is contained in qrels (the answer), it hits.
5. the score is `(hits / query total count)`

I run the task with some similar approaches, and compare the total computation time, to find the fastest approach.

## Sec 3. Comparision

## Sec 4. Instruction to reproduce

### Execution
Mac OS:
```
time python3 -m src.python.ort
time python3 -m src.python.pytorch
cargo b -r && time ./target/release/ort
```

## Sec 5. Implementation details
