# ALBERT for Vietnamese

## Introduction
ALBERT is "A Lite" version of BERT, a popular unsupervised language
representation learning algorithm. ALBERT uses parameter-reduction techniques
that allow for large-scale configurations, overcome previous memory limitations,
and achieve better behavior with respect to model degradation.

For a technical detail description of the algorithm, see the paper:

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut)
and the official repository [Google ALBERT](https://github.com/google-research/ALBERT)

Google researchers introduced three standout innovations with ALBERT. [[1]](https://medium.com/syncedreview/googles-albert-is-a-leaner-bert-achieves-sota-on-3-nlp-benchmarks-f64466dd583)

- Factorized embedding parameterization: Researchers isolated the size of the hidden layers from the size of vocabulary embeddings by projecting one-hot vectors into a lower dimensional embedding space and then to the hidden space, which made it easier to increase the hidden layer size without significantly increasing the parameter size of the vocabulary embeddings.
 
- Cross-layer parameter sharing: Researchers chose to share all parameters across layers to prevent the parameters from growing along with the depth of the network. As a result, the large ALBERT model has about 18x fewer parameters compared to BERT-large.
    
- Inter-sentence coherence loss: In the BERT paper, Google proposed a next-sentence prediction technique to improve the model’s performance in downstream tasks, but subsequent studies found this to be unreliable. Researchers used a sentence-order prediction (SOP) loss to model inter-sentence coherence in ALBERT, which enabled the new model to perform more robustly in multi-sentence encoding tasks.
  
**We preproduced ALBERT for Vietnamese dataset and provided pre-trained model in below.** 

## Data preparation
Training data is the Vietnamese wikipedia corpus from [Wikipedia](https://dumps.wikimedia.org/)

Data is preprocessed and extracted using [WikiExtractor](https://github.com/attardi/wikiextractor) 

Training SentencePiece model for producing vocab file, we used 30000 words from this model on Vietnamese wikipedia corpus. 

SentencePice model and vocab at folder assets.


## Pretraining
### Creating data for pretraining


We trained ALBERT model on config version 2 of base and large.

### Base Config
```json
{
  "attention_probs_dropout_prob": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0,
  "embedding_size": 128,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_hidden_groups": 1,
  "net_structure_type": 0,
  "gap_size": 0,
  "num_memory_blocks": 0,
  "inner_group_num": 1,
  "down_scale_factor": 1,
  "type_vocab_size": 2,
  "vocab_size": 30000
}
```


### Large Config
```json
{
  "attention_probs_dropout_prob": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0,
  "embedding_size": 128,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "num_hidden_groups": 1,
  "net_structure_type": 0,
  "gap_size": 0,
  "num_memory_blocks": 0,
  "inner_group_num": 1,
  "down_scale_factor": 1,
  "type_vocab_size": 2,
  "vocab_size": 30000
}
```
Create tfrecord for training data:
```python
python create_pretraining_data.py \
  --input_file={path to wiki data} \
  --dupe_factor=10 \
  --output_file={path to save tfrecord} \
  --vocab_file assets/albertvi_30k-clean.vocab \
  --spm_model_file assets/albertvi_30k-clean.model 
```

Pre-training base config 
```bash
python run_pretraining.py \
--albert_config_file=assets/base/albert_config.json \
--input_file={tfrecord path} \
--output_dir={}\
--export_dir={}\
--train_batch_size=4096 \
--do_eval=True \
--use_tpu=True \
```


Pre-training large config
```bash
python run_pretraining.py \
--albert_config_file=assets/large/albert_config.json \
--input_file={tfrecord path} \
--output_dir={}\
--export_dir={}\
--train_batch_size=512 \
--do_eval=True \
--use_tpu=True \
```
## Pretrained model
We run ~1M steps for base config and ~250k for large config.

Loss value and eval results show in below:
<<image>>
<<image>>


You could download the pretrained models at [here]()

## Experimential Results
Coming soon


## Acknowledgement
Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).

Thank so much @lampts, @dal team for suporting me to finish this project.

## How to cite this work        
Please cite this repository https://github.com/ngoanpv/albert_vi

Email me or create an issue for any questions .