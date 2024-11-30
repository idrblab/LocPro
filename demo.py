from locpro.inference import getInference
import pandas as pd
from fasta import FASTA

# Define input and output paths
output_folder = 'result_demo'
input_fasta_path = "./demo.fasta"

# Load the FASTA file using the FASTA utility
input_fasta = FASTA(input_fasta_path)

# Perform inference using the "Main" model type
result_main,result_all = getInference(input_fasta=input_fasta, output_folder=output_folder)

# Save the "Main" inference results to a CSV file
pd.DataFrame(result_main).to_csv(f'./{output_folder}/resultMain.csv', index=False)

# # Save the "All" inference results to a CSV file
# pd.DataFrame(result_all).to_csv(f'./{output_folder}/resultAll.csv', index=False)








