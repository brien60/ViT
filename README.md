# ViT
A from-scratch implementation of a Vision Transformer (ViT) in PyTorch, based on the original ViT paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

## Lessons:

Experiment with different lr schedulers, try CosineAnnealingLR before warm restarts

Make sure data augmentations are in the correct order

Did not actually create separate EncoderBlocks, instead just copied a single one `num_encoder` times.

Remember dropout should always be switched off for eval (flash attention dropout)

Have separate LNs for mlp and attn, also layer norm the cls token before classification

Accumulate loss*batch size and then divide by number of samples at the end to avoid overweighing the last batch when `samples % batch_size != 0`


## Summary of Optimizations:
- use detach() instead of item() (moves data to cpu, forces cuda sync)
- non_blocking, persistent_workers. (add more)

- Increasing prefetch_factor did not meaningfully improve throughput.

- Full training throughput pre-optimizations: 2,800 img/s. SM util (GPU util) was jumping between 0-5% and 90-96%

- GPU-only training throughput using dummy data was 17,255 img/s, meanwhile the DataLoader throughput was 3,485 img/s

- 16 workers was found to be the best configuration out of 4, 8, 12, 16, 20, and 24 workers. 20 workers was found out to be slightly better after the below optimizations, but it resulted in an OOM when beginning the evaluation step.

- Using conditional RGB converting improved throughput by ~13%

- Moving uint8 -> float32 conversion and image normalization from the CPU to the GPU end increased DataLoader throughput by ~68.6% (3,653 -> 6,167 img/s).

### Closing Thoughts
- The main bottleneck is CPU-side image preprocessing rather than GPU computation

- GPU-side augmentation was considered but not implemented due to the complexity of handling variable-sized source images (likely needing padding) and the uncertain end-to-end benefit.

- Further optimization put off in favor of starting ImageNet training and experiment tracking.


