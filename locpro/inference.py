import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from locpro.constants import Categories
from collections import defaultdict
from locpro.featurePROFEAT import generate_profeat_dataframe
from locpro.featureESM import generate_esm2_dataframe

# Setting device for computation (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    """
    Custom Dataset class for inference.
    Loads the protein features and returns them in a usable format.
    """
    def __init__(self, data_df):
        self.data = data_df
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract data for each index
        protein_id = self.data.iloc[idx]['Proteins']
        inp_map = self.data.iloc[idx]['Profeat_feature']
        inp_esm = self.data.iloc[idx]['ESM2_feature']

        # Convert to tensors and adjust dimensions
        inp_map = torch.tensor(inp_map, dtype=torch.float32).permute(2, 0, 1)  # Reorder dimensions
        inp_esm = torch.tensor(inp_esm, dtype=torch.float32)
        return inp_map, inp_esm, protein_id
    
class LocPro(nn.Module):
    def __init__(self, nb_classes, lstm_layers=2):
        super(LocPro, self).__init__()
        
        # CNN layers
        self.cnn1 = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.dense1 = nn.Linear(128 * 9 * 9, 2048)  # 128 * 9 * 9 from CNN output shape
        self.dense2 = nn.Linear(2048, 1024)
        
        # ESM feature processing
        self.dense_esm = nn.Linear(1280, 1024)  # Adjust ESM input dimension
        
        # Concatenation of CNN and ESM features
        self.dense3 = nn.Linear(1024 + 1024, 512)
        self.dropout = nn.Dropout(0.5)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(2048, 256, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.output = nn.Linear(512, nb_classes)


    def forward(self, inp_map, inp_esm,apply_sigmoid=False):
        # CNN processing
        x = self.pool1(torch.relu(self.cnn1(inp_map)))  # (batch_size, 64, 19, 19)
        x = self.pool2(torch.relu(self.cnn2(x)))  # (batch_size, 128, 9, 9)
        x = self.flatten(x)  # (batch_size, 128 * 9 * 9)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dropout(x)
        
        # ESM processing
        y = torch.relu(self.dense_esm(inp_esm))  # (batch_size, 1280)

        # Concatenate CNN and ESM features
        concat = torch.cat((x, y), dim=1)  # (batch_size, 2048 + 512)
        net = self.dropout(concat)

        # LSTM processing
        net = net.unsqueeze(1).repeat(1, 11, 1)  # Repeat features across sequence dimension
        lstm_out, _ = self.lstm(net)  # (batch_size, 11, 512)
        lstm_out = lstm_out[:, -1, :]  # Get output from last time step

        # Final classification
        classify = self.output(lstm_out)  # (batch_size, nb_classes)

        if apply_sigmoid:
            classify = torch.sigmoid(classify)

        return classify





def modelInference(model_path = './', data_file='./input.pkl', label_num=1):
    """
    Perform model inference on the input dataset.
    Loads the pre-trained models and thresholds, and applies them to the test dataset.
    """
    # Load the test dataset
    test_df = pd.read_pickle(data_file)
    test_dataset = InferenceDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load optimal thresholds for classification
    with open(f'{model_path}/optimal_thresholds_{label_num}.pkl', 'rb') as file:
        optimal_thresholds = pickle.load(file)

    # Define labels for the classification
    labels = Categories[f"label-{label_num}"]
    final_result = defaultdict(lambda: defaultdict(list))
    for fold in range(5):  # Iterate through each fold
        # Load the model for the current fold
        model = LocPro(nb_classes=label_num).to(device)
        model_file = f'{model_path}/model_v{label_num}_fold_{fold}.pkl'
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        all_predictions = []
        all_protein_ids = []
        
        with torch.no_grad():
            for batch_idx, (inp_map, inp_esm, protein_id) in enumerate(test_loader):
                inp_map, inp_esm = inp_map.to(device), inp_esm.to(device)
                outputs = model(inp_map, inp_esm, apply_sigmoid=True)

                all_predictions.append(outputs.cpu().numpy())
                all_protein_ids.extend(protein_id)
        all_predictions = np.concatenate(all_predictions, axis=0)
        optimal_threshold = optimal_thresholds[f'Fold{fold}']
        
        
        for protein_id, probits in zip(all_protein_ids, all_predictions):
            probit_threshold = probits / optimal_threshold
            probit_threshold = (probit_threshold == np.max(probit_threshold)).astype(int)

            for label, probit, threshold, probit_threshold in zip(labels, probits, optimal_threshold, probit_threshold):
                final_result[protein_id][label].append({
                    'Fold': f'Fold{fold}',
                    'Probit': round(float(probit), 4),
                    'Threshold': round(float(threshold), 4),
                    'Result_exceeds_threshold': int(probit > threshold),
                    'Result_max_probability': round(float(probit_threshold), 4),
                    'Result_final': int(probit > threshold or probit_threshold),
                })
    # Aggregate results
    result = []
    for protein_id, locations in final_result.items():
        for location, folds in locations.items():
            result_sum = np.mean([i['Result_final'] for i in folds])
            result.append({
                'ProteinID': protein_id,
                'Label': location,
                'folds': folds,
                'ResultSum': result_sum,
                'ResultFinal': int(result_sum > 0.5),
            })

    return result



def getInference(input_fasta, output_folder, plk_name='features.pkl'):
    """
    Prepare the input FASTA file, generate features, and perform inference.
    """
    # Generate ESM2 and PROF features for the input FASTA
    esm2_df = generate_esm2_dataframe(input_fasta, f"{output_folder}/esm2")
    profeat_df = generate_profeat_dataframe(input_fasta, f"{output_folder}/profeat")
    # Merge features and save to a pickle file
    final_df = pd.merge(profeat_df, esm2_df, on="Proteins", how="inner")
    final_df.to_pickle(f"{output_folder}/{plk_name}")
    
    # Choose the appropriate model based on the model type
    

    result_main = modelInference('./models/location-main', f"{output_folder}/{plk_name}", label_num=10)
    result_all = modelInference('./models/location-all', f"{output_folder}/{plk_name}", label_num=91)

    return result_main, result_all