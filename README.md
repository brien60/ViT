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