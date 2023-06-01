from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os
import glob
import json
from flask import Flask, request

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "a very simple API server based on ExLlama")

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.SWITCHED)
parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)
parser.add_argument("-mlp", "--mlp", type = ExLlamaConfig.MLPMethod.argparse, choices = list(ExLlamaConfig.MLPMethod), help="Matmul method", default = ExLlamaConfig.MLPMethod.SWITCHED)
parser.add_argument("-s", "--stream", type = int, help = "Stream layer interval", default = 0)
parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
parser.add_argument("-dq", "--dequant", type = str, help = "Number of layers (per GPU) to de-quantize at load time")

parser.add_argument("-temp", "--temperature", type = float, help = "Temperature", default = 0.5)
parser.add_argument("-topk", "--top_k", type = int, help = "Top-K", default = 32)
parser.add_argument("-topp", "--top_p", type = float, help = "Top-P", default = 0.2)
parser.add_argument("-minp", "--min_p", type = float, help = "Min-P", default = 0.00)
parser.add_argument("-repp",  "--repetition_penalty", type = float, help = "Repetition penalty", default = 1.1)
parser.add_argument("-repps", "--repetition_penalty_sustain", type = int, help = "Past length for repetition penalty", default = 32)
parser.add_argument("-repdc", "--repetition_penalty_decay", type = int, help = "Decay length for repetition penalty", default = 64)
parser.add_argument("-beams", "--beams", type = int, help = "Number of beams for beam search", default = 1)
parser.add_argument("-beamlen", "--beam_length", type = int, help = "Number of future tokens to consider", default = 1)

parser.add_argument("-gpfix", "--gpu_peer_fix", action = "store_true", help = "Prevent direct copies of data between GPUs")

cargs = parser.parse_args()

config = {}

# apply arguments

config.update({
    "tokenizer": cargs.tokenizer,
    "config": cargs.config,
    "model": cargs.model,
    "directory": cargs.directory,
    "attention": cargs.attention,
    "matmul": cargs.matmul,
    "mlp": cargs.mlp,
    "stream": cargs.stream,
    "gpu_split": cargs.gpu_split,
    "dequant": cargs.dequant,
    "temperature": cargs.temperature,
    "top_k": cargs.top_k,
    "top_p": cargs.top_p,
    "min_p": cargs.min_p,
    "repetition_penalty": cargs.repetition_penalty,
    "repetition_penalty_sustain": cargs.repetition_penalty_sustain,
    "repetition_penalty_decay": cargs.repetition_penalty_decay,
    "beams": cargs.beams,
    "beam_length": cargs.beam_length,
    "gpu_peer_fix": cargs.gpu_peer_fix
})

# load config

if os.path.exists("config.json"):
    with open("config.json", "rb") as f:
        config.update(json.load(f))

if config.get("directory") is not None:
    config["tokenizer"] = os.path.join(config.get("directory"), "tokenizer.model")
    config["config"] = os.path.join(config.get("directory"), "config.json")
    st_pattern = os.path.join(config.get("directory"), "*.safetensors")
    st = glob.glob(st_pattern)
    if len(st) == 0:
        print(f" !! No files matching {st_pattern}")
        sys.exit()
    if len(st) > 1:
        print(f" !! Multiple files matching {st_pattern}")
        sys.exit()
    config["model"] = st[0]
else:
    if config.get("tokenizer") is None or config.get("config") is None or config.get("model") is None:
        print(" !! Please specify either -d or all of -t, -c and -m")
        sys.exit()

# Some feedback

print(f" -- Loading model")
print(f" -- Tokenizer: {config.get('tokenizer')}")
print(f" -- Model config: {config.get('config')}")
print(f" -- Model: {config.get('model')}")

# Instantiate model and generator

lconfig = ExLlamaConfig(config["config"])
lconfig.model_path = config["model"]
lconfig.attention_method = config["attention"]
lconfig.matmul_method = config["matmul"]
lconfig.mlp_method = config["mlp"]
lconfig.stream_layer_interval = config["stream"]
lconfig.gpu_peer_fix = config["gpu_peer_fix"]
lconfig.set_auto_map(config["gpu_split"])
lconfig.set_dequant(config["dequant"])

model = ExLlama(lconfig)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(config["tokenizer"])

print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")

generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = config["temperature"]
generator.settings.top_k = config["top_k"]
generator.settings.top_p = config["top_p"]
generator.settings.min_p = config["min_p"]
generator.settings.token_repetition_penalty_max = config["repetition_penalty"]
generator.settings.token_repetition_penalty_sustain = config["repetition_penalty_sustain"]
generator.settings.token_repetition_penalty_decay = config["repetition_penalty_decay"]
generator.settings.beams = config["beams"]
generator.settings.beam_length = config["beam_length"]

def searcharr(haystack, needle, start=0):
    for i in range(start, len(haystack) - len(needle) + 1):
        found = True
        for j in range(0, len(needle)):
            if haystack[i + j] != needle[j]:
                found = False
                break
        if found:
            return i
    return None

bos_tok = tokenizer.tokenizer.Encode("\x02")[1]
eos_tok = tokenizer.tokenizer.Encode("\x03")[1]
newline_tok = tokenizer.tokenizer.Encode("\n")[1]
def tokenize_evil(str):
    ids = tokenizer.tokenizer.Encode(str.replace("<s>", "\x02").replace("</s>", "\x03"))
    for i in range(0, len(ids)):
        if ids[i] == bos_tok:
            ids[i] = tokenizer.bos_token_id
        if ids[i] == eos_tok:
            ids[i] = tokenizer.eos_token_id
    return ids

app = Flask(__name__)

@app.route("/basilisk/ping")
def get_ping():
    secret = config.get("shared_secret")
    if secret != None and request.headers.get("authorization") != secret:
        return "unauthorized", 403
    
    return "pong"

@app.route("/basilisk/tokenize", methods=["POST"])
def post_tokens():
    secret = config.get("shared_secret")
    if secret != None and request.headers.get("authorization") != secret:
        return "unauthorized", 403
    
    body = request.get_json()
    if body["prompt"] == None:
        return "prompt required", 400
    
    ids = tokenize_evil(body["prompt"])
    return {
        "tokens": ids
    }

@app.route("/basilisk/infer", methods=["POST"])
def post_infer():
    secret = config.get("shared_secret")
    if secret != None and request.headers.get("authorization") != secret:
        return "unauthorized", 403
    
    body = {
        "min_length": 4,
        "max_new_tokens": 256,
        "stopping_strings": ["\n"],

        # providing the last inference here attempts to prevent any token from
        # being generated at the same position as in the previous sequence
        "positional_repeat_penalty": 1.3,
        "positional_repeat_inhibit": []
    }
    body.update(config)
    body.update(request.get_json())

    if body["prompt"] == None:
        return "prompt required", 400

    # update settings
    generator.settings.temperature = body["temperature"]
    generator.settings.top_k = body["top_k"]
    generator.settings.top_p = body["top_p"]
    generator.settings.min_p = body["min_p"]
    generator.settings.token_repetition_penalty_max = body["repetition_penalty"]
    generator.settings.token_repetition_penalty_sustain = body["repetition_penalty_sustain"]
    generator.settings.token_repetition_penalty_decay = body["repetition_penalty_decay"]
    generator.settings.token_penalized_penalty = body["positional_repeat_penalty"]
    
    # tokenize stopping strings
    stopping_strings_tok = []
    for string in body["stopping_strings"]:
        toked = tokenizer.tokenizer.Encode(string)[1:]
        stopping_strings_tok.append(toked)
    
    # tokenize positional inhibit
    pr_toks = body["positional_repeat_inhibit"]
    
    ids = torch.tensor([tokenize_evil(body["prompt"])])

    # begin inference
    generator.gen_begin(ids)
    initial_len = generator.sequence[0].shape[0]

    generator.begin_beam_search()

    for i in range(body["max_new_tokens"]):
        # penalize inference repeats from last turn
        if i < len(pr_toks):
            generator.penalize_tokens([pr_toks[i]])
        else:
            generator.penalize_tokens(None)

        # Disallowing the end condition tokens seems like a clean way to force longer replies.
        if i < body["min_length"]:
            generator.disallow_tokens([
                newline_tok,
                tokenizer.eos_token_id
            ])
        else:
            generator.disallow_tokens(None)
        
        # Get a token
        token = generator.beam_search()

        # stop on stopping strings
        stophit = False
        for string in stopping_strings_tok:
            haystack = generator.sequence[0][initial_len:][-len(string):].tolist()
            if searcharr(haystack, string) != None:
                stophit = True
        if stophit:
            break

        # If token is EOS, replace it with newline before continuing
        if token.item() == tokenizer.eos_token_id:
            generator.replace_last_token(newline_tok)
            break
    
    tokens = generator.sequence[0][initial_len:]
    text = tokenizer.decode(tokens)

    # clean up stopping strings
    # this sucks
    for string in body["stopping_strings"]:
        for j in range(len(string) - 1, 0, -1):
            if text[-j:] == string[:j]:
                text = text[:-j]
                break
        else:
            continue
        break

    return {
        "text": text,
        "tokens": tokens.tolist()
    }

app.run(threaded=False)