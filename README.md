# CommitBART
CommitBART is a Encoder-Decoder pre-trained model on GitHub commit. The model covers 7 programming languges (C, CSharp, JAVA, JAVAScript, PHP, Python, TypeScript).

# Benchmark
We proposed a benchmark for research on commit-related task. We collect 7M instances of commits from top-ranked GitHub projects for pre-training, whereas the other 500K data is used for fine-tuning. The data can be found [here](https://drive.google.com/file/d/1sXYZeP-hwTrwTwa_RQF4qLOvAPEqjNRI/view?usp=sharing).

### Dependency
- pip install torch
- pip install transformers

### Model initialization
We follow the framework of huggingface/transformers. The base model is PLBART. For loading the model, you can simply:
```python
import torch
from transformers import PLBartTokenizer, PLBartConfig, PLBartModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PLBartTokenizer.from_pretrained("NTU-CSL/CommitBART-unseg")
model = PLBartModel.from_pretrained("NTU-CSL/CommitBART-unseg")
model.to(device)
```
Or you can load the model's checkpoint from the link provided in folder ckpt(recommended).

### Comparison with ChatGPT
We also provide the reuslts of ChatGPT for 100 samples at this [link](https://drive.google.com/drive/folders/1xMaQd2xtr4Rgac7vYLBcTWcIf4udzFwU) where the key "commitbart" is the results for CommitBART, "chatgpt" is the results for ChatGPT and "date" refers to the commit date.