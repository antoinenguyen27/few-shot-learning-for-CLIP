# Research Notes

These notes summarize implementation-relevant facts from the public method and dataset sources. They are not a substitute for reading the papers before implementation.

## Methods

PromptSRC:

- PromptSRC regularizes learned prompts to preserve frozen CLIP generalization.
- Its public implementation follows CoOp-style dataset handling and ViT prompt-learning assumptions.
- In this repo, it should consume common split files and the shared OpenCLIP ViT-B/32 model.
- The official reports use base and novel accuracy plus harmonic mean, so the repo includes base/new metric helpers for the later base-to-new protocol.

LP++:

- LP++ is a strong few-shot linear-probe baseline for CLIP.
- It is likely the easiest method to connect to this repo because it can operate on cached OpenCLIP image/text features.
- It should use `common.features.cache` for feature-cache paths.
- The official repo frames LP++ as black-box and feature-oriented, so the common OpenCLIP feature helpers are the intended integration path.

PromptKD:

- PromptKD performs unsupervised prompt distillation from a stronger teacher to a student CLIP model.
- Its official code uses a teacher model and OpenAI CLIP-style setup.
- In strict few-shot experiments, any unlabeled images used for distillation must be documented and must not silently include test-only information.
- The official repo's cross-dataset setting is transductive when it distills on unlabeled domain data, so transductive PromptKD rows must be logged separately.

DPC:

- DPC is a plug-in for prompt-tuned vision-language models.
- The official repo describes a two-stage flow: first tune the backbone prompt method, then tune DPC on top of it.
- It can require DPC-specific annotation JSON files. Treat those as a method dependency and record them in run metadata.
- The official results target base/new trade-off, so DPC should log base accuracy, new accuracy, and harmonic mean when run under a base-to-new protocol.

## Public Sources

- OpenCLIP: https://github.com/mlfoundations/open_clip
- OpenCLIP ViT-B-32 256px model card: https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K
- CoOp dataset organization: https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md
- LP++ repo: https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline
- PromptSRC repo: https://github.com/muzairkhattak/PromptSRC
- PromptKD repo: https://github.com/zhengli97/PromptKD
- DPC repo: https://github.com/JREion/DPC
- Flowers102 Kaggle: https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset
- EuroSAT Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
- Stanford Cars Kaggle: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset/data
