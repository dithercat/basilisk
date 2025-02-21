## Model compatibility

- [x] Support for act-order models ~~(a bit slow for now)~~
- [x] ~~Support for v1 models without groupsize~~ Nah.
- [x] Test more models
- [ ] Consider support for loading GGML models
- [ ] Utility to scan and validate .safetensors files
- [x] Figure out if there are quantized models with irregular groupsize (there are some at least with no groupsize)

## GPU compatibility (etc.)

- [ ] Support for ROCm/AMD GPUs
- [ ] Test that CUDA code works on GTX 10-series and RTX 20-series at some point
- [x] Test performance on P40 (would be a good GPU to support)
- [ ] Improve performance on P40
- [x] Tunable kernel parameters
- [ ] More tunable kernel parameters
- [x] Test on Windows
- [ ] Easier extension loading on Windows
- [ ] Setup instructions for Windows

## Testing

- [ ] Figure out an apples-to-apples way of comparing perplexity with other implementations
- [ ] Compile charts of inference speed vs context length for variety of models, compare to other implementations

## VRAM optimization

- [x] ~~Fix layer streaming so it isn't unusably slow~~ (removed)
- [x] ~~Allow layer streaming to integrate with other features like device splitting~~ Nope
- [x] ~~Provide alternative backend to allow layers on CPU~~ Nah

## Speed optimization

- [x] Support for de-quantizing select matrices at load time
- [x] ~~Better vector-matrix multiplication for de-quantized matrices~~ (dequant was a dead end)
- [ ] Fused QKV projection
- [x] Fused MLP
- [x] Fused RoPE
- [x] ~~Build attention mask in CUDA rather than PyTorch~~
- [x] ~~Disable attention mask when it isn't needed~~ (not possible with SDP)
- [x] Figure out why inference appears to be CPU-bound (kernel launch overhead)
- [ ] Reduce no. kernel launches to minimum (tail launch, fusion etc.)
- [x] Measure PyTorch module overhead (negligible in eval mode)
- [x] Examine if scaled_dot_product_attention is actually the best attention method for single tokens (it's not)
- [ ] Implement attention in CUDA
- [x] Rewrite at least the quantized matmul kernel. Should be a bunch of special cases to consider

## Generation

- [x] Memory-efficient beam search implementation
- [ ] Optimized beam search
- [ ] Multi-token censoring/de-censoring
- [ ] Multi-token repetition penalties
- [ ] (Multi) LoRA support
- [ ] Guided generation (chat with multiple bots at once, etc.)
- [ ] Multiple chat modes with prompt templates (instruct, etc.)

## Interface

- [x] Simple web interface?
- [ ] API server 

## Web UI

- [ ] Controls to enable beam search
- [ ] Rewrite/refactor all the JavaScript and CSS
- [ ] Support for prompt formats/instruct mode
- [ ] Make it a little prettier
- [ ] Test various edge cases
- [ ] Better error handling

## ??

- [ ] Allow for backpropagation
- [ ] LoRA training features
- [ ] Soft prompt training