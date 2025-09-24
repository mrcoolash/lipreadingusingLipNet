

# 📘 Lipreading using LipNet

## 1. Overview

This project implements **LipNet**, a deep learning model for end-to-end lipreading. It takes **video frames of a speaker’s mouth** as input and outputs the corresponding **spoken sentence** as text.

Pipeline includes:

* **Data preprocessing** (video frame extraction, resizing, normalization).
* **Label encoding** (mapping spoken sentences to character sequences).
* **LipNet architecture** (3D Convolutions → Spatiotemporal feature extraction → Bidirectional GRUs → CTC Loss for sequence alignment).
* **Training loop** with accuracy tracking.
* **Inference** for decoding predictions into text.

---

## 2. Data Pipeline

### 2.1 Input

* **Raw video clips**: typically RGB videos of a person speaking.
* Each clip → sequence of frames (`T x H x W x C`).
* Ground-truth transcription: spoken sentence (characters).

### 2.2 Preprocessing

* Frames extracted and resized (usually `50x100` per frame).
* Normalization applied (`[0,1]` scale).
* Sentences tokenized to character indices.

**Output**:

* `(Batch, Time, Height, Width, Channels)` tensor for videos.
* `(Batch, SequenceLength)` tensor for labels.

---

## 3. Model Architecture (LipNet)

### 3.1 Input Shape

```
Video Input: (Batch, Time=75, Height=50, Width=100, Channels=3)
```

### 3.2 Layers

1. **3D Convolution Blocks**

   * Three stacked Conv3D layers with BatchNorm + Dropout + MaxPooling.
   * Capture **spatiotemporal features** across consecutive frames.

2. **Flatten + TimeDistributed**

   * Frames compressed into feature vectors per timestep.

3. **Bidirectional GRUs (2 layers)**

   * Extract **temporal dependencies** across frames.
   * Forward + backward GRUs capture context from entire sequence.

4. **Fully Connected (Dense)**

   * TimeDistributed dense layer over GRU outputs.
   * Softmax over character vocabulary.

5. **CTC Loss Layer**

   * Used instead of cross-entropy to align variable-length sequences without explicit frame-to-label mapping.

### 3.3 Output

* Predicted character probabilities per timestep.
* After decoding → final predicted sentence.

---

## 4. Training Pipeline

* **Loss Function**: CTC Loss (`keras.backend.ctc_batch_cost`).
* **Optimizer**: Adam.
* **Metrics**: Character Error Rate (CER), Word Error Rate (WER), Accuracy.
* **Batch Input**:

  * `inputs`: video tensor.
  * `labels`: ground truth sequences.
  * `input_length`, `label_length`: required for CTC alignment.

**Output**:

* Trained LipNet model weights.
* Logs of training loss and validation accuracy.

---

## 5. Inference (Prediction)

### Input

* Single video clip of lip movements.

### Process

1. Preprocess frames (resize, normalize).
2. Pass through trained LipNet.
3. Decode using **best path decoding** or **beam search decoding**.

### Output

* Predicted text transcription of spoken words.

---

## 6. Code Structure (High-Level)

```plaintext
LipreadingusingLipNet.ipynb
│
├── Data Loading & Preprocessing
│   ├── Video frame extraction
│   ├── Label encoding (character map)
│
├── Model Definition
│   ├── Conv3D blocks
│   ├── GRU layers
│   ├── Dense + Softmax
│   └── CTC loss function
│
├── Training
│   ├── Data generator
│   ├── Model compile + fit
│
├── Evaluation
│   ├── CER, WER metrics
│   └── Validation accuracy
│
└── Inference
    ├── Load saved model
    ├── Preprocess new video
    └── Decode predicted sequence
```

---

## 7. Key Inputs & Outputs

| Stage              | Input Shape / Type                    | Output Shape / Type                         |
| ------------------ | ------------------------------------- | ------------------------------------------- |
| Preprocessing      | Raw video (`T x H x W x 3`)           | Tensor (`Batch x T x 50 x 100 x 3`)         |
| Label Encoding     | Sentence string                       | Encoded int sequence                        |
| Conv3D Blocks      | `(Batch, T, 50, 100, 3)`              | Feature maps `(Batch, T, H', W', Channels)` |
| Bi-GRU             | Sequence features                     | `(Batch, T, 2*GRU_units)`                   |
| Dense + Softmax    | GRU outputs                           | Probabilities `(Batch, T, VocabSize)`       |
| CTC Loss           | Probabilities + labels                | Scalar loss                                 |
| Inference Decoding | Probabilities `(Batch, T, VocabSize)` | Final sentence string                       |

---


