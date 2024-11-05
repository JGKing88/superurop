import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch as torch
import pickle

from superurop.utilities.utils import *

dataset_manager = DatasetManager()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exp_with_gaussian_rise(x, a, b, d, sigma):
    return a * np.exp(-b * x) + d * np.exp(-((x - 1)**2) / (2 * sigma**2))

def exponential(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e

def consruct_fourgram_model_wrapper(next_token_dist, vocab_distribution, num_base_threegrams, tree_depth, buckets, vocab_size):
    fourgram_model = {}
    entropy = {}

    def construct_fourgram_model(threegram, threegram_prob, depth=0):
        tokens_dist = []
        while sum(tokens_dist) < 1:
            next_token_prob = np.random.choice(buckets, 1, p=next_token_dist)[0]
            if sum(tokens_dist) + next_token_prob > 1:
                next_token_prob = 1 - sum(tokens_dist)
            tokens_dist.append(next_token_prob)

        num_tokens = len(tokens_dist)
        tokens = np.random.choice(vocab_size, num_tokens, p=vocab_distribution)

        #if we are at the leaf node, only pick continuations that are in the model
        if depth == tree_depth:
            cycle_tokens = []
            cycle_tokens_dist = []
            for i in range(num_tokens):
                new_threegram = threegram[1:] + (tokens[i],)
                if new_threegram in fourgram_model:
                    cycle_tokens.append(tokens[i])
                    cycle_tokens_dist.append(tokens_dist[i])
            if not len(cycle_tokens) == 0:
                tokens = cycle_tokens
                tokens_dist = cycle_tokens_dist / np.sum(cycle_tokens_dist)
            #if there is no possible continuation, just use previous tokens

        fourgram_model[threegram] = (tokens, tokens_dist)
        entropy[threegram] = -np.sum(tokens_dist * np.log(tokens_dist)) * threegram_prob

        if depth < tree_depth:
            for i in range(num_tokens):
                new_threegram = threegram[1:] + (tokens[i],)
                new_threegram_prob = threegram_prob * tokens_dist[i]
                construct_fourgram_model(new_threegram, new_threegram_prob, depth + 1)
    
    base_threegrams = np.random.choice(len(vocab_distribution), num_base_threegrams*3, p=vocab_distribution)
    base_threegrams = base_threegrams.reshape(-1, 3)
    for base_index in range(num_base_threegrams):
        base_threegram = base_threegrams[base_index]
        base_threegram = tuple(base_threegram)
        #probability of the base threegram
        threegram_prob = 1/num_base_threegrams
        construct_fourgram_model(base_threegram, threegram_prob)
    
    base_threegrams = [tuple(base_threegram) for base_threegram in base_threegrams]
    
    return fourgram_model, entropy, base_threegrams

def get_distributions(pv_dist_multiplier, pnt_dist_multiplier, vocab_size, buckets, base_vocab_dist_params, base_next_token_dist_params):
    vocab_dist_params = base_vocab_dist_params.copy()
    vocab_dist_params[1] = base_vocab_dist_params[1] * pv_dist_multiplier
    vocab_dist = exponential(np.arange(vocab_size), *vocab_dist_params)
    vocab_dist = vocab_dist / np.sum(vocab_dist)

    next_token_dist_params = base_next_token_dist_params.copy()
    next_token_dist_params[1] = base_next_token_dist_params[1] * pnt_dist_multiplier
    next_token_dist = exp_with_gaussian_rise(buckets, *next_token_dist_params)
    next_token_dist = next_token_dist / np.sum(next_token_dist)

    return vocab_dist, next_token_dist

def generate_dataset(generator, params, base_threegrams):
    num_samples = params["num_samples"]
    context_length = params["context_length"]

    dataset = []
    entropy = 0
    count = 0

    for sample_idx in range(num_samples):
        first_threegram_idx = np.random.randint(0, len(base_threegrams))
        context = base_threegrams[first_threegram_idx]
        sample = list(context)
        for token_idx in range(context_length - 3):
            tokens, token_probs = generator[context]
            token = np.random.choice(tokens, 1, p=token_probs)[0]
            token_prob = token_probs[np.where(tokens == token)[0][0]]
            entropy += -np.log(token_prob) * token_prob
            context = context[1:] + (token,)
            if context not in generator:
                count += 1
                context = base_threegrams[np.random.randint(0, len(base_threegrams))]
                sample.extend(list(context))
            else:
                sample.append(token)
        dataset.append(sample)
    print("Number of times we had to reset context: " + str(count))
    entropy = entropy / num_samples
    return dataset, entropy

if __name__ == "__main__":
    vocab_multipliers = {"rh": 0.005, "h": 0.02, "m": 1, "l": 10, "rl": 50}
    next_token_multipliers = {"rh": 1, "h": 0.75, "m": 0.6, "l": 0.5, "rl": 0.25}
    base_next_token_dist_params = pickle.load(open("/om2/user/jackking/superurop/scripts/base_next_token_dist_params.pkl", 'rb'))
    base_vocab_dist_params = pickle.load(open("/om2/user/jackking/superurop/scripts/base_vocab_dist_params.pkl", 'rb'))

    vocab_size = 2000
    num_base_threegrams = 50
    tree_depth = 4
    buckets = np.linspace(0, 1, 1001)[1:]
    generator_type = "toy"

    for nt_dist_level in next_token_multipliers.keys():
        for v_dist_level in vocab_multipliers.keys():

            print(f"Next Token Multiplier: {next_token_multipliers[nt_dist_level]} ({nt_dist_level}), Vocab Multiplier: {vocab_multipliers[v_dist_level]} ({v_dist_level})")

            generator_name = f"v{v_dist_level}_nt{nt_dist_level}"
            v_multiplier = vocab_multipliers[v_dist_level]
            nt_multiplier = next_token_multipliers[nt_dist_level]
            vocab_dist, next_token_dist = get_distributions(v_multiplier, nt_multiplier, vocab_size, buckets, base_vocab_dist_params, base_next_token_dist_params)
            fourgram_model, entropy, base_threegrams = consruct_fourgram_model_wrapper(next_token_dist, vocab_dist, num_base_threegrams, tree_depth, buckets, vocab_size)
            params = {"vocab_size": vocab_size, "num_base_threegrams": num_base_threegrams, "tree_depth": tree_depth, "generator_type": generator_type, "generator_name": generator_name, "v_dist_level": v_dist_level, "nt_dist_level": nt_dist_level}
            dataset_manager.save_dataset_generator(generator_type, generator_name, fourgram_model, params)

            print("Generator Entropy: " + str(np.sum(list(entropy.values()))))
            print("Total Number of Threegrams: " + str(len(fourgram_model.keys())))

            num_samples = 300
            context_length = 16

            params["num_samples"] = num_samples
            params["context_length"] = context_length
            dataset, entropy = generate_dataset(fourgram_model, params, base_threegrams)
            print(f"Dataset Entropy: {entropy}")

