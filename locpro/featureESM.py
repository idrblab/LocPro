import torch
import esm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import os
from collections import defaultdict

# Determine device: use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Custom dataset for handling input data
class ProteinDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the dataset with a list of protein IDs and sequences.
        """
        self.data = data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample at the given index.
        """
        return self.data[idx]


def compute_sequence_representations(model, batch, batch_converter, alphabet):
    """
    Computes sequence representations for a batch of protein sequences using the ESM2 model.

    Args:
        model: Pre-trained ESM2 model.
        batch: Batch of protein sequences.
        batch_converter: Converts raw sequences into tensor format.
        alphabet: Alphabet object containing model tokenization details.

    Returns:
        batch_labels: List of protein IDs in the batch.
        sequence_representations: List of sequence representations (averaged embeddings).
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

    sequence_representations = [
        token_representations[i, 1 : tokens_len - 1].mean(0).cpu().numpy()
        for i, tokens_len in enumerate(batch_lens)
    ]
    return batch_labels, sequence_representations


def weighted_average(representations):
    """
    Computes the weighted average of sequence embeddings based on sequence lengths.

    Args:
        representations: List of tuples (embedding, length).

    Returns:
        Weighted average embedding.
    """
    total_weight = sum(length for _, length in representations)
    weighted_sum = sum(embedding * length for embedding, length in representations)
    return weighted_sum / total_weight



def process_esm2(input_fasta, feature_path, model, alphabet, batch_converter, batch_size, splitLength=2000):
    """
    Main function to process protein sequences and save their embeddings.

    Args:
        input_fasta: FASTA file containing protein sequences.
        feature_path: Path to save computed embeddings.
        model: Pre-trained ESM2 model.
        alphabet: Alphabet object for model tokenization.
        batch_converter: Function to convert batches into tensor format.
        batch_size: Batch size for DataLoader.

    Returns:
        None
    """
    # Prepare data for processing
    data = [(protein.id, str(protein.seq)) for protein in input_fasta]
    filtered_data = [
        (protein_id, sequence)
        for protein_id, sequence in data
        if not os.path.exists(os.path.join(feature_path, f"{protein_id}.npy"))
    ]

    dataset = ProteinDataset(filtered_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()  # Ensure the model is in evaluation mode

    for batch in tqdm(dataloader, desc="Processing batches"):
        # Convert batch into a format compatible with the batch converter
        batch = [i for i in zip(*batch)]

        max_length = max(len(item[1]) for item in batch)
        results_dict = defaultdict(list)
        # Process sequences in chunks of 2000 tokens
        for start in range(0, max_length, splitLength):
            sub_batch = [(protein_id, seq[start : start + splitLength]) for protein_id, seq in batch]
            batch_labels, sequence_representations = compute_sequence_representations(
                model, sub_batch, batch_converter, alphabet
            )
            
            # Store embeddings and sequence lengths
            for label, embedding, length in zip(
                batch_labels, sequence_representations, [len(seq[1]) for seq in sub_batch]
            ):
                if length > 0:
                    results_dict[label].append((embedding, length))

        # Save weighted average embeddings for each protein in the batch
        for label, reps in results_dict.items():
            np.save(os.path.join(feature_path, f"{label}.npy"), weighted_average(reps))

        # Clear GPU cache after each batch to free memory
        torch.cuda.empty_cache()


def generate_esm2_dataframe(input_fasta, feature_path, batch_size=1):
    """
    Generates a Pandas DataFrame containing protein IDs and their ESM2 embeddings.

    Args:
        input_fasta: FASTA file containing protein sequences.
        feature_path: Path to save computed embeddings.
        batch_size: Batch size for DataLoader.

    Returns:
        DataFrame with protein IDs and embeddings.
    """
    # Ensure the feature path exists
    os.makedirs(feature_path, exist_ok=True)
    
    # Load the pre-trained ESM2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    
    # Process the sequences and generate embeddings
    process_esm2(input_fasta, feature_path, model, alphabet, batch_converter, batch_size)

    # Load saved embeddings into a DataFrame
    esm_data = [
        {
            "Proteins": os.path.splitext(file)[0],
            "ESM2_feature": np.load(os.path.join(feature_path, file)),
        }
        for file in os.listdir(feature_path)
        if file.endswith(".npy")
    ]
    return pd.DataFrame(esm_data)


