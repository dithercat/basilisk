# basilisk

tiny LLaMA inference server

## setup

1. follow the original ExLlama `README.md`, other than installing `flask` for
   the http server
2. create a `config.json` (see `config_example.json`)
   - `shared_secret` is optional; if set, the `Authorization` header of all
     requests will be checked against this
3. just run `basilisk.py`

## notes

- this doesnt take advantage of ExLlama's excellent caching stuff because it's
  intended for use in multiple channels at the same time
- `BOS` and `EOS` are encoded as control characters `STX` and `ETX` (`\x02` and
  `\x03` respectively) and converted to their actual representations after
  tokenizing
  - most instruct models behave best when a `BOS` is placed in the simulacrum's
    input line
    - example prompt: `"USER: tell me something interesting\nASSISTANT:\x02"`
  - you can use the `special_convert` parameter to lazily replace literal
    `<s>`/`</s>` with `BOS`/`EOS`
    - example prompt: `"USER: tell me something interesting\nASSISTANT:<s>"`
    - this isnt recommended (it makes it trivial for a user to inject these
      special tokens among other things). instead, write your client app to use
      `STX`/`ETX` if they are needed
  - **IMPORTANT**: keep in mind that tokens generally have spaces (if any) in
    the *front*, and HF Transformers tokenizers (*not* basilisk) usually eat
    whitespace surrounding special tokens. basilisk treats `STX`/`ETX` in-place
    as a literal `BOS`/`EOS` (by directly mapping tokens `5 -> 1` and `6 -> 2`),
    and does no preprocessing to make it friendlier for the model. this seems
    small, but it's crucial to getting good output
    - **GOOD**: `"ASSISTANT:\x02 Hello,"`
      (tokenizes as `[..., ":", "<s>", " Hello", ","]`)
    - **BAD**: `"ASSISTANT: \x02Hello, "`
      (tokenizes as `[..., ":", " ", "<s>", "Hello", ",", " "]`,
      caused incoherence in testing with a Vicuna finetune)

# ExLlama

A rewrite of the HF transformers implementation of Llama with the following goals, among others:

* Designed for use with quantized weights
* Fast and memory-efficient inference (not just attention)
* Mapping across multiple devices
* Built-in (multi) LoRA support
* Companion library of funky sampling functions

Disclaimer: This is currently a preview of a work in progress. Or maybe a proof of concept. Either way any part of it
is subject to change.

## Hardware/software requirements

I am developing on an RTX 4090 and an RTX 3090-Ti. Both cards support the CUDA kernel, but there might be
incompatibilities with older cards. I have no way of testing that right now.

I have no idea if this works on Windows/WSL, but feel free to try and contribute/give feedback.

## Dependencies

This list might be incomplete:

* `torch` tested on 2.1.0 (nightly) with cu118, might work with older CUDA versions also
* `safetensors` 0.3.1
* `sentencepiece`
* `ninja`
* `flask` (only for the web UI)

## How to

There is no installer or package at the moment, but try this:

    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
    
    pip install safetensors sentencepiece ninja

    git clone https://github.com/turboderp/exllama
    cd exllama

    python test_benchmark_inference.py -t <path_to_tokenizer.model> -c <path_to_config.json> \ 
      -m <path_to_model.safetensors> -p -ppl

Alternatively, just specify a directory containing `tokenizer.model`, `config.json` and a single `.safetensors` file: 

    python test_benchmark_inference.py -d <path_to_model_files> -p -ppl

The CUDA extension is loaded at runtime so there's no need to install it separately. It will be compiled on the first
run and cached to `~/.cache/torch_extensions/` which could take a little while. If nothing happens at first, give it
a minute to compile.

Chatbot examples:

    python test_chatbot.py -d <path_to_model_files> -un "Jeff" -p prompt_chatbort.txt

    python test_chatbot.py -d <path_to_model_files> -un "Maxine" -p prompt_assistant.txt -nnl \
      -temp 1.00 -topp 0.95 -beams 5 -beamlen 20

## Web UI

I made a simple web UI for it. Like the rest of the project, it's a work in progress. Don't look at the JavaScript,
it was mostly written by ChatGPT and it will haunt your dreams. But it sort of works, and it's kinda fun, especially
multibot mode:

![_screenshot.jpg](_screenshot.jpg)

To run it:

    pip install flask

    python webui/app.py -d <path_to_model_files>

Note that sessions are stored in `~/exllama_sessions/`. 

## Results so far

### New implementation
| Model    | Size | groupsize | act             | Seq. len.            | VRAM      | Prompt    | Best    | Worst   | Ppl  |
|----------|------|-----------|-----------------|----------------------|-----------|-----------|---------|---------|------|
| Llama    | 7B   | 128       | no              | 2,048 t              | 5,063 MB  | 9,382 t/s | 151 t/s | 129 t/s | 6.45 |
| Llama    | 13B  | 128       | no              | 2,048 t              | 8,937 MB  | 5,427 t/s | 94 t/s  | 80 t/s  | 5.62 |
| Llama    | 30B  | 128       | no              | 2,048 t              | 20,496 MB | 2,291 t/s | 44 t/s  | 37 t/s  | 4.60 |
| Llama    | 30B  | 128       | yes             | 2,048 t              | 20,509 MB | 2,166 t/s | 41 t/s  | 36 t/s  | 4.55 |
| Llama    | 30B  | 32        | yes             | 1,550 t <sup>1</sup> | 21,218 MB | 2,152 t/s | 38 t/s  | 34 t/s  | 4.52 |
| Koala    | 13B  | 128       | yes             | 2,048 t              | 8,944 MB  | 5,127 t/s | 86 t/s  | 75 t/s  | 6.73 |
| WizardLM | 30B  | -         | no <sup>2</sup> | 2,048 t              | 19,900 MB | 2,313 t/s | 45 t/s  | 38 t/s  | 5.75 |

<sup>1</sup> Can not achieve full sequence length without OoM (yet)  
<sup>2</sup> Not quite sure if this is act-order or not. Weights have no group index, at least   

All tests done on stock RTX 4090 / 12900K, running with a desktop environment, with a few other apps also using VRAM.

**"Prompt"** speed is inference over the sequence length listed minus 128 tokens. **"Worst"** is the average speed for
the last 128 tokens of the full context (worst case) and **"Best"** lists the speed for the first 128 tokens in an
empty sequence (best case.)

VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured only to verify that the models are working. The dataset used is a particular, small sample from
WikiText, so scores are not necessarily comparable to other Llama benchmarks.

### Dual GPU results

Since many seem to be interested in running 65B models, I can confirm that this works with two 24 GB GPUs. The
following benchmarks are from a 4090 + 3090-Ti with `-gs 17.2,24`:

| Model    | Size | groupsize | act | Seq. len.            | VRAM      | Prompt  | Best   | Worst  | Ppl  |
|----------|------|-----------|-----|----------------------|-----------|---------|--------|--------|------|
| Llama    | 65B  | 128       | yes | 2,048 t              | 39,804 MB | 926 t/s | 19 t/s | 17 t/s | 4.20 |
| Llama    | 65B  | 32        | yes | 2,048 t              | 43,424 MB | 895 t/s | 16 t/s | 15 t/s | 4.11 |


### Testing long sequences

The following tests were all done on **30B/65B, 4bit 128g** with various settings, just to test the max sequence length
and get a sense of what can be achieved with different or multiple GPUs right now. Llama goes incoherent generating 
past 2048 tokens anyway, but with some fine-tuning, who knows? Note that these tests were run a while ago and the
speeds are no longer current.

|                        | Size | Seq. len. | VRAM                 | Long seq. | Ind.   | 
|------------------------|------|-----------|----------------------|-----------|--------|
| 4090/24GB              | 30B  | 2,516 t   | 22,145 MB            | 1140 t/s  | 28 t/s |
| 4090/24GB + 3070Ti/8GB | 30B  | 3,932 t   | 22,055 MB + 7,377 MB | 840 t/s   | 22 t/s |
| A6000/48GB (headless)  | 30B  | 9,032 t   | 46,863 MB            | 645 t/s   | 12 t/s |
| A100/80GB (headless)   | 65B  | 9,520 t   | 79,009 MB            | 650 t/s   | 9 t/s  |

## Todo

Moved the todo list [here](TODO.md).  

## Compatibility

I downloaded a whole bunch of GPTQ models to test compatibility. [Here](model_compatibility.md) is the list of models
confirmed to be working right now.

## Recent updates

**2023-05-19**: Wrote a CUDA implementation of the layer norm. Turns out it was a bit of a bottleneck for the smaller
models. Noticeably faster now.

**2023-05-21**: Added beam search implementation. It doesn't process beams in parallel which saves a lot of VRAM but
does slow it down a bit. There should be ways to mitigate the slowdown. It's not clear how much better beam search
performs in practice, but it's at least theoretically superior and there are other features coming which will build
on it, like multi-token repetition penalties and (de-)censoring.

**2023-05-22**: Added option to auto-split layers across multiple GPUs based on VRAM allocation. 

**2023-05-22**: Added option to dequantize layers at load-time which _should_ speed up inference, but it turns out
Torch's fp16 matmul is actually slower than the quantized matmul. Maybe bandwidth is the only bottleneck right now?
Need to experiment some more.

**2023-05-24**: Downloaded a bunch of models from HF and set up a test script. Should be a good sampling of the most
popular finetunes right now. I'll add more to the list as I come across them. They all seem to be working.

**2023-05-24**: Added fused rotary embeddings and some minor optimizations. 13% faster on 7B, 9% on 13B. Small
improvement on larger models. Added best-case scores to benchmark results and some clarification. For easier
comparisons to other implementations, or whatever.

**2023-05-27**: Better memory management in CUDA. Introduced auto switch between Torch's SDP backend and regular 
matmul attention with some tweaks. Finished CUDA MLP. All in all about 10% faster with these updates.

**2023-05-29**: Web UI is _almost_ up and running. Having to learn JavaScript, and it turns out I hate JavaScript. But
ChatGPT is an incredible resource for learning new languages, I gotta say, so it's not as painful as it could have
been. Anyway, in the process of working with the UI I discovered I've been measuring prompt speed incorrectly. Either
Torch or CUDA or the GPU driver does some sort of caching or self-calibration or lazy initialization during the first
pass through the model, so subsequent passes are actually _way_ faster than what I've been recording. Doesn't do much
for individual tokens, but benchmarks updated anyway. Closing in on 10k tokens/second for 7B. (!)

**2023-06-02**: Web UI is now in a fairly working state. Expect it to be a little scuffed in places. There will be a
rewrite at some point to make the client-side code less seizure-inducing. It has multibot mode, chat rewind and editing
features, sessions, and more. I'm going to build it out with support for instruct prompting and such, in time.