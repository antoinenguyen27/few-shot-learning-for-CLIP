# Embedding t-SNE Analysis

Generated from OpenCLIP image embeddings using local manifests and split files.

## Settings

- Source: `split-test`
- Protocol: `few_shot_all_classes`
- Shots/seed for split source: `16` / `1`
- Max samples per class: `10`
- Model: `ViT-B-32-256`
- Pretrained: `datacomp_s34b_b86k`

## Charts

![Combined dataset t-SNE](figures/tsne/combined_dataset_tsne.svg)

![EuroSAT class t-SNE](figures/tsne/eurosat_class_tsne.svg)

![Flowers102 class t-SNE](figures/tsne/flowers102_class_tsne.svg)

![Stanford Cars class t-SNE](figures/tsne/stanford_cars_class_tsne.svg)
