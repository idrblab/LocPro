import pandas as pd
import pickle
from os.path import join, exists
from shutil import rmtree
import numpy as np
from profeat import profeat_to_df
from fasta import FASTA
import profeat
import os
import locpro.resources as resources



def load_data(file, num):
    """
    Load a CSV file, drop rows with missing values, and return specific columns.
    
    Args:
        file (str): Path to the CSV file.
        num (int): Number of columns to extract.
    
    Returns:
        pd.DataFrame: Processed DataFrame with selected columns.
    """
    data = pd.read_csv(file, header=None)
    data.dropna(axis=0, inplace=True)
    return data.iloc[:, 1:num]

class DataProcessor:
    """
    Class for processing protein data and generating features using Profeat.
    """
    def __init__(self, protein_file, proteins_fasta_file, save_file, num,
                 grid_file="data_grid.pkl", assess_file="row_asses.pkl",
                 f_min="cafa4_min.pkl", f_max="cafa4_max.pkl"):
        """
        Initialize the DataProcessor with necessary file paths and parameters.

        Args:
            protein_file (str): Path to the protein feature file.
            proteins_fasta_file (str): Path to the protein sequence FASTA file.
            save_file (str): Path to save the output.
            num (int): Number of features to use.
            grid_file (str): Path to the grid mapping file.
            assess_file (str): Path to the assignment mapping file.
            f_min (str): Path to the min values file for normalization.
            f_max (str): Path to the max values file for normalization.
        """
        self.protein_file = protein_file
        self.split_file = proteins_fasta_file
        self.save_file = save_file
        self.grid_file = grid_file
        self.assess_file = assess_file
        self.f_min = f_min
        self.f_max = f_max
        self.num = num
        self._load_data()

    def _load_data(self):
        """
        Load data, normalize features, and load grid/assignment mappings.
        """
        # Load and preprocess features
        proteins_f = profeat_to_df(self.protein_file)
        proteins_f.dropna(axis=0, inplace=True)
        self.feature_data = proteins_f.iloc[:, :self.num]

        # Load normalization parameters
        with resources.open_binary(self.f_min) as fmin, resources.open_binary(self.f_max) as fmax:
            f_min = pickle.load(fmin)
            f_max = pickle.load(fmax)
            
        # Normalize feature data
        self.feature_data = (self.feature_data - f_min) / ((f_max - f_min) + 1e-8)
        self.proteins = list(proteins_f.index)
        
        # Load grid and assignment mappings
        with resources.open_binary(self.grid_file) as gf, resources.open_binary(self.assess_file) as af:
            self.data_grid = pickle.load(gf)
            self.row_asses = pickle.load(af)



    def calculate_feature(self, row_num, size):
        """
        Calculate protein features using preloaded data and mappings.

        Args:
            row_num (int): Number of rows in the grid.
            size (tuple): Size of the feature matrix.
        """
        
        protein_seqs = FASTA(self.split_file).sequences
        class_labels = ['Composition', 'Autocorrelation', 'Physiochemical', 'Interaction',
                        'Quasi-sequence-order descriptors', 'PAAC for amino acid index set', 'Amphiphilic Pseudo amino acid composition']


        data_grid = self.data_grid
        row_asses = self.row_asses
        proteins = self.proteins

        processed_data = []
        
        
        # protein_all = []
        # sequences_all = []
        # feature_all = []
        for i, protein in enumerate(proteins):
            if protein in protein_seqs:
                col_list = np.zeros(size)
                for j in range(len(data_grid['x'])):
                    channel = data_grid['subtype'][j]
                    index = class_labels.index(channel)
                    feature_index = row_asses[j]
                    row = j % row_num
                    col = j//row_num
                    col_list[col][row][index] = self.feature_data.iloc[i, feature_index]
                processed_data.append({
                    "Proteins": protein,
                    "Sequence": str(protein_seqs[protein].seq),
                    "Profeat_feature": col_list
                })
        # Save processed data as a DataFrame
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_pickle(self.save_file)


def process(proteins_fasta_file, profeat_file, save_file):
    """
    Main function to process the protein FASTA and feature files.
    
    Args:
        proteins_fasta_file (str): Path to the input FASTA file.
        profeat_file (str): Path to the Profeat output file.
        save_file (str): Path to save processed data.
    """
    if proteins_fasta_file == None:
        raise ValueError("Must provide the input fasta sequences.")

    data_processor = DataProcessor(protein_file=profeat_file,
                                   proteins_fasta_file=proteins_fasta_file,
                                   save_file=save_file, num=1484)
    data_processor.calculate_feature(row_num=39, size=(39, 39, 7))
    
def ProfeatMain(proteins_fasta_file, output_dir = None, overwrite = False):
    """
    Run Profeat on a given FASTA file and process the output.

    Args:
        proteins_fasta_file (str): Path to the input FASTA file.
        output_dir (str): Directory to save Profeat outputs.
        overwrite (bool): Whether to overwrite existing output directory.
    """
    if not output_dir:
        output_dir = f"{proteins_fasta_file}.output"

    if exists(output_dir):
        if overwrite:
            rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists!")
            exit(1)

    profeat.run(proteins_fasta_file, output_dir)
    process(
        proteins_fasta_file=proteins_fasta_file,
        profeat_file=join(output_dir, "output-protein.dat"),
        save_file=join(output_dir, "profeat_features.pkl"),
    )


def generate_profeat_dataframe(inputFASTA, outputFolder):
    """
    Process multiple protein sequences using Profeat and aggregate results.

    Args:
        inputFASTA: Iterable of protein sequences.
        outputFolder (str): Directory to save processed data.

    Returns:
        pd.DataFrame: Aggregated DataFrame of Profeat features.
    """
    os.makedirs(outputFolder, exist_ok=True)
    allSeqs = [seq for seq in inputFASTA]
    step = 5000
    promap_df = pd.DataFrame()
    for i in range(0, len(allSeqs), step):
        proteins = allSeqs[i : i + step]
        folderStep = f"./{outputFolder}/proteins{i}-{i+step}"
        
        with open(f"{folderStep}.fasta", "w") as f:
            for protein in proteins:
                sequence = str(protein.seq)
                f.write(f">{protein.id}\n{sequence}\n")
                print("", file=f)
        if exists(folderStep):
            rmtree(folderStep)
        ProfeatMain(f"{folderStep}.fasta", output_dir=folderStep)
        with open(f"{folderStep}/profeat_features.pkl", "rb") as t:
            tmp_df = pickle.load(t)[["Proteins", "Sequence", "Profeat_feature"]]
            promap_df = pd.concat([promap_df, tmp_df], ignore_index=True)
    return promap_df
