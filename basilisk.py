from model import ExLlama, ExLlamaCache
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import model_init
import argparse
import torch
import os
import json
from flask import Flask, request

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "a very simple API server based on ExLlama")

model_init.add_args(parser)

parser.add_argument("-temp", "--temperature", type = float, help = "temperature", default = 0.5)
parser.add_argument("-topk", "--top_k", type = int, help = "top-k", default = 32)
parser.add_argument("-topp", "--top_p", type = float, help = "top-p", default = 0.2)
parser.add_argument("-minp", "--min_p", type = float, help = "min-p", default = 0.00)
parser.add_argument("-repp",  "--repetition_penalty", type = float, help = "repetition penalty", default = 1.1)
parser.add_argument("-repps", "--repetition_penalty_sustain", type = int, help = "past length for repetition penalty", default = 32)
parser.add_argument("-repdc", "--repetition_penalty_decay", type = int, help = "decay length for repetition penalty", default = 64)
parser.add_argument("-repat", "--positional_repetition_penalty", type = int, help = "positional repetition penalty", default = 1.2)
parser.add_argument("-beams", "--beams", type = int, help = "number of beams for beam search", default = 1)
parser.add_argument("-beamlen", "--beam_length", type = int, help = "number of future tokens to consider", default = 1)

args = parser.parse_args()

# load config

if os.path.exists("config.json"):
    with open("config.json", "rb") as f:
        args.__dict__.update(json.load(f))

model_init.get_model_files(args)

# Some feedback

print(f" -- the system is coming up. please wait. ^^")
print(f" -- tokenizer: {args.tokenizer}")
print(f" -- model config: {args.config}")
print(f" -- model: {args.model}")

# Instantiate model and generator

lconfig = model_init.make_config(args)

model = ExLlama(lconfig)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(args.tokenizer)

model_init.print_stats(model)

generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = args.temperature
generator.settings.top_k = args.top_k
generator.settings.top_p = args.top_p
generator.settings.min_p = args.min_p
generator.settings.token_repetition_penalty_max = args.repetition_penalty
generator.settings.token_repetition_penalty_sustain = args.repetition_penalty_sustain
generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
generator.settings.beams = args.beams
generator.settings.beam_length = args.beam_length

# HACK: optionally replace literal <s> and </s> with STX and ETX
#       and then replace STX and ETX with BOS and EOS
bos_tok = tokenizer.tokenizer.Encode("\x02")[1]
eos_tok = tokenizer.tokenizer.Encode("\x03")[1]
def tokenize_evil(str, special_convert=False):
    if special_convert:
        str = str.replace("<s>", "\x02").replace("</s>", "\x03")
    ids = tokenizer.tokenizer.Encode(str)
    for i in range(0, len(ids)):
        if ids[i] == bos_tok:
            ids[i] = tokenizer.bos_token_id
        if ids[i] == eos_tok:
            ids[i] = tokenizer.eos_token_id
    return ids

# actual app init
app = Flask(__name__)

# validate shared secret
def auth_check():
    secret = args.__dict__.get("shared_secret")
    return secret != None and request.headers.get("authorization") != secret

# dedicated "see if the config is working" endpoint
# maps to kasumiLLM's ping()
# (which used to be a hack using textgen-webui token count endpoint)
@app.route("/basilisk/ping")
def get_ping():
    if auth_check():
        return "unauthorized", 403
    return "pong"

# allow clients to retrieve most of the base config
@app.route("/basilisk/config")
def get_config():
    if auth_check():
        return "unauthorized", 403
    
    cfg = {}
    cfg.update(args.__dict__)

    # dont disclose these
    del cfg["shared_secret"]
    del cfg["directory"]
    del cfg["config"]
    del cfg["model"]
    del cfg["tokenizer"]

    return cfg

# tokenize a string
# primary use is to get token count, but also for positional_repeat_inhibit
@app.route("/basilisk/tokenize", methods=["POST"])
def post_tokens():
    if auth_check():
        return "unauthorized", 403
    
    body = request.get_json()
    prompt = body.get("prompt")
    if prompt == None:
        return "prompt required", 400
    
    ids = tokenize_evil(prompt, body.get("special_convert"))
    return {
        "tokens": ids
    }

# actual inference endpoint
# param names are kind of sort of compatible with text-generation-webui
@app.route("/basilisk/infer", methods=["POST"])
def post_infer():
    if auth_check():
        return "unauthorized", 403
    
    # get parameters+overrides
    body = {
        "min_length": 4,
        "max_new_tokens": 256,
        "stopping_strings": ["\n"],

        # providing the last inference here attempts to prevent any token from
        # being generated at the same position as in the previous sequence
        "positional_repetition_penalty": 1.2,
        "positional_repeat_inhibit": [],

        # convert "<s>" and "</s>"?
        # you should avoid this if possible by using STX and ETX in your app
        "special_convert": False
    }
    body.update(args.__dict__)
    body.update(request.get_json())

    prompt = body.get("prompt")
    if prompt == None:
        return "prompt required", 400
    prompt = prompt.strip()

    # update settings
    generator.settings.temperature = body["temperature"]
    generator.settings.top_k = body["top_k"]
    generator.settings.top_p = body["top_p"]
    generator.settings.min_p = body["min_p"]
    generator.settings.token_repetition_penalty_max = body["repetition_penalty"]
    generator.settings.token_repetition_penalty_sustain = body["repetition_penalty_sustain"]
    generator.settings.token_repetition_penalty_decay = body["repetition_penalty_decay"]
    generator.settings.token_penalized_penalty = body["positional_repetition_penalty"]
    
    # tokenize stopping strings
    stopping_strings_tok = []
    for string in body["stopping_strings"]:
        toked = tokenizer.tokenizer.Encode(string)[1:]
        stopping_strings_tok.append(toked)
    
    # build positional inhibit lists
    pr_toks = []
    for seq in body["positional_repeat_inhibit"]:
        for i in range(len(seq)):
            if i == len(pr_toks):
                pr_toks.append([seq[i]])
            else:
                pr_toks[i].append(seq[i])
    
    ids = torch.tensor([tokenize_evil(prompt, body["special_convert"])])

    # begin inference
    generator.gen_begin(ids)
    initial_len = generator.sequence[0].shape[0]

    generator.begin_beam_search()

    for i in range(body["max_new_tokens"]):
        # penalize inference repeats from last turn
        if i < len(pr_toks):
            generator.penalize_tokens(pr_toks[i])
        else:
            generator.penalize_tokens(None)

        # Disallowing the end condition tokens seems like a clean way to force longer replies.
        if i < body["min_length"]:
            generator.disallow_tokens([
                tokenizer.newline_token_id,
                tokenizer.eos_token_id
            ])
        else:
            generator.disallow_tokens(None)
        
        # Get a token
        token = generator.beam_search()

        # stop on stopping strings
        stophit = False
        for needle in stopping_strings_tok:
            haystack = generator.sequence[0][initial_len:][-len(needle):].tolist()
            if len(needle) > len(haystack):
                continue
            found = True
            for j in range(0, len(needle)):
                if haystack[j] != needle[j]:
                    found = False
                    break
            if found:
                stophit = True
        if stophit:
            break

        # If token is EOS, replace it with newline before continuing
        if token.item() == tokenizer.eos_token_id:
            generator.replace_last_token(tokenizer.newline_token_id)
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