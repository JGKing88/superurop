import os
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import Dataset
import transformers
import pickle

import os
import json
import pickle
import torch

class DatasetManager:
    def __init__(self, base_dir="/om2/user/jackking/superurop/data/datasets"):
        self.base_dir = base_dir

    def _get_path(self, *args):
        """Helper to construct paths based on the base directory and other arguments."""
        return os.path.join(self.base_dir, *filter(None, args))

    def _save_json(self, path, data):
        """Helper to save data as JSON."""
        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_json(self, path):
        """Helper to load data from JSON."""
        with open(path, 'r') as f:
            return json.load(f)

    def _save_pickle(self, path, data):
        """Helper to save data as a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, path):
        """Helper to load data from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_tensor(self, path, data):
        """Helper to save data as a PyTorch tensor file."""
        torch.save(data, path)

    def _load_tensor(self, path):
        """Helper to load data as a PyTorch tensor file."""
        return torch.load(path)

    def load_dataset_generator(self, dataset_type, dataset_name):
        dataset_path = self._get_path(dataset_type, dataset_name)
        return (
            self._load_pickle(self._get_path(dataset_path, "generator.pkl")),
            self._load_json(self._get_path(dataset_path, "generator_params.json"))
        )

    def save_dataset_generator(self, dataset_type, dataset_name, generator, params):
        dataset_path = self._get_path(dataset_type, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        self._save_json(self._get_path(dataset_path, "generator_params.json"), params)
        self._save_pickle(self._get_path(dataset_path, "generator.pkl"), generator)

    def load_dataset(self, dataset_type, dataset_name, model_name=None, context_type=None, prefix = ""):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name, context_type)
        return (
            self._load_tensor(self._get_path(dataset_path, f"{prefix}data.pt")),
            self._load_json(self._get_path(dataset_path, "params.json"))
        )

    def save_dataset(self, dataset_type, dataset_name, tokenized_data, model_name=None, context_type=None, prefix = "", params={}):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name, context_type)
        os.makedirs(dataset_path, exist_ok=True)
        self._save_json(self._get_path(dataset_path, "params.json"), params)
        self._save_tensor(self._get_path(dataset_path, f"{prefix}data.pt"), tokenized_data)

    def save_surprisals(self, dataset_type, dataset_name, surprisals, model_name, context_length, params={}):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name)
        os.makedirs(dataset_path, exist_ok=True)
        self._save_json(self._get_path(dataset_path, f"{context_length}_surprisals_params.json"), params)
        self._save_tensor(self._get_path(dataset_path, f"{context_length}_surprisals.pt"), surprisals)

    def load_surprisals(self, dataset_type, dataset_name, model_name, context_length):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name)
        return (
            self._load_tensor(self._get_path(dataset_path, f"{context_length}_surprisals.pt")),
            self._load_json(self._get_path(dataset_path, f"{context_length}_surprisals_params.json"))
        )

    def save_curvatures(self, dataset_type, dataset_name, curvatures, model_name):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name)
        os.makedirs(dataset_path, exist_ok=True)
        self._save_pickle(self._get_path(dataset_path, f"curvatures.pt"), curvatures)
    
    def load_curvatures(self, dataset_type, dataset_name, model_name):
        dataset_path = self._get_path(dataset_type, dataset_name, model_name)
        return self._load_pickle(self._get_path(dataset_path, f"curvatures.pt"))

    def list_datasets(self):
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(self._get_path(d))]

    def load_all_datasets(self):
        return {
            dataset_name: {
                'data': self.load_dataset(dataset_name)[0],
                'params': self.load_dataset(dataset_name)[1]
            }
            for dataset_name in self.list_datasets()
        }


# class DatasetManager:
#     def __init__(self, base_dir="/om2/user/jackking/superurop/data/datasets"):
#         self.base_dir = base_dir

#     def load_dataset_generator(self, dataset_type, dataset_name):
#         dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name)
#         params_path = os.path.join(dataset_path, "generator_params.json")
#         data_path = os.path.join(dataset_path, "generator.pkl")
        
#         # Load dataset parameters
#         with open(params_path, 'r') as f:
#             params = json.load(f)
        
#         # Load generator dictionary
#         with open(data_path, 'rb') as f:
#             generator = pickle.load(f)
        
#         return generator, params

    
#     def save_dataset_generator(self, dataset_type, dataset_name, generator, params):
#         dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name)
        
#         # Create folder for the dataset if it doesn't exist
#         if not os.path.exists(dataset_path):
#             os.makedirs(dataset_path)

#         # Save parameters as a JSON file
#         params_path = os.path.join(dataset_path, "generator_params.json")
#         with open(params_path, 'w') as f:
#             json.dump(params, f)

#         # Save generator dictionary as a pickle file
#         data_path = os.path.join(dataset_path, "generator.pkl")
#         with open(data_path, 'wb') as f:
#             pickle.dump(generator, f)


#     def load_dataset(self, dataset_type, dataset_name, model_name=None, context_type=None):
#         if model_name is not None:
#             dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name, model_name)
#             if context_type is not None:
#                 dataset_path = os.path.join(dataset_path, context_type)
#         else:
#             dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name)

#         params_path = os.path.join(dataset_path, "params.json")
#         data_path = os.path.join(dataset_path, "data.pt")
        
#         # Load dataset parameters
#         with open(params_path, 'r') as f:
#             params = json.load(f)
        
#         # Load tokenized text data as tensors
#         data = torch.load(data_path)
        
#         return data, params


#     def save_dataset(self, dataset_type, dataset_name, tokenized_data, model_name=None, context_type=None, params={}):
#         if model_name is not None:
#             dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name, model_name)
#             if context_type is not None:
#                 dataset_path = os.path.join(dataset_path, context_type)
#         else:
#             dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name)
        
#         # Create folder for the dataset if it doesn't exist
#         if not os.path.exists(dataset_path):
#             os.makedirs(dataset_path)

#         # Save parameters as a JSON file
#         params_path = os.path.join(dataset_path, "params.json")
#         with open(params_path, 'w') as f:
#             json.dump(params, f)

#         # Save tokenized data as a tensor file using PyTorch
#         data_path = os.path.join(dataset_path, "data.pt")
#         torch.save(tokenized_data, data_path)
    
#     def save_surprisals(self, dataset_type, dataset_name, surprisals, model_name, context_length, params={}):
#         dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name, model_name)
        
#         # Create folder for the dataset if it doesn't exist
#         if not os.path.exists(dataset_path):
#             os.makedirs(dataset_path)

#         # Save parameters as a JSON file
#         params_path = os.path.join(dataset_path, f"{context_length}_surprisals_params.json")
#         with open(params_path, 'w') as f:
#             json.dump(params, f)

#         # Save tokenized data as a tensor file using PyTorch
#         data_path = os.path.join(dataset_path, f"{context_length}_surprisals.pt")
#         torch.save(surprisals, data_path)
    
#     def load_surprisals(self, dataset_type, dataset_name, model_name, context_length):
#         dataset_path = os.path.join(self.base_dir, dataset_type, dataset_name, model_name)
#         params_path = os.path.join(dataset_path, f"{context_length}_surprisals_params.json")
#         data_path = os.path.join(dataset_path, f"{context_length}_surprisals.pt")
        
#         # Load dataset parameters
#         with open(params_path, 'r') as f:
#             params = json.load(f)
        
#         # Load tokenized text data as tensors
#         data = torch.load(data_path)
        
#         return data, params

#     def list_datasets(self):
#         return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

#     def load_all_datasets(self):
#         datasets = {}
#         for dataset_name in self.list_datasets():
#             data, params = self.load_dataset(dataset_name)
#             datasets[dataset_name] = {'data': data, 'params': params}
#         return datasets
    
# class ModelManager:
#     def __init__(self, base_dir="/om2/user/jackking/superurop/data/models"):
#         self.base_dir = base_dir

#     def load_model(self, model_type, model_name):
#         model_path = os.path.join(self.base_dir, model_type, model_name)
        
#         # Load model with Transformers
#         model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        
#         return model

#     def save_model(self, model_type, model_name, model):
#         model_path = os.path.join(self.base_dir, model_type)
        
#         # Create folder for the model if it doesn't exist
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)

#         # Save model state dict using Transformers
#         model_path = os.path.join(model_path, model_name)
#         model.save_pretrained(model_path)

#     def list_models(self):
#         return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]


class AutoregressiveDataset(Dataset):
    tokenized_data: List[torch.Tensor]

    def __init__(self, tokenized_data: List[torch.Tensor]):
        self.tokenized_data = tokenized_data

    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.tokenized_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input sequence and the target sequence (shifted by 1 position).
        """
        sequence = self.tokenized_data[idx]
        return sequence
        # input_seq = sequence[:-1]  # Input is the sequence without the last token
        # target_seq = sequence[1:]  # Target is the sequence shifted by one token
        
        # return input_seq, target_seq