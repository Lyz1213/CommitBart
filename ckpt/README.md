### Pre-train model checkpoint
We provide our pretrained checkpoint of CommitBART, that is CommitBART-base, and also the CommitBART pretrained without segment embedding CommitBART-unseg. The checkpoint can be found at https://drive.google.com/file/d/1If7knZ9CG1mK_iiC-zaf5HcC2yNffs3R/view?usp=sharing. You can directly load the model's paramters from 'module.bin'.

## Hugginface/transformers
We also upload our model to hugginface. For directly inference using our pre-trained model for text infilling or PLNL2PL, NL2PL generations, you need to load the lm_head.bin to model.lm_head for conditional generation.
Since our CommitBART contains a embedding matrix for segmengts. If you load the CommitBART-base model from huggingface, you still need to load the token_type_embeddings.bin to model.token_type_embeedings.

