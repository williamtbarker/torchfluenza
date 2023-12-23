import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

# --- Constants ---
NUM_AMINO_ACIDS = 20  # Number of amino acids
MAX_SEQ_LENGTH = 800  # Maximum sequence length (adjust as needed)

# --- Utility Functions ---

def encode_protein_sequence(sequence):
    """Encode a protein sequence into a numerical format."""
    # Simple encoding: each amino acid mapped to an integer
    amino_acid_mapping = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    encoded = [amino_acid_mapping.get(aa, 0) for aa in sequence]
    # Pad sequences to have the same length
    padded = encoded + [0] * (MAX_SEQ_LENGTH - len(encoded))
    return padded[:MAX_SEQ_LENGTH]

def mutate_sequence(sequence):
    """Introduce a random mutation in the protein sequence."""
    # Choose a random position for mutation
    pos = random.randint(0, len(sequence) - 1)
    mutation = random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
    return sequence[:pos] + mutation + sequence[pos+1:]

# --- Model Definition ---

class InfluenzaVirusModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InfluenzaVirusModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Taking the last output of sequence
        x = self.fc(x)
        return x

# --- Main Execution ---

def main():
    # Load data
    df = pd.read_csv('protein_sequences.csv')  # Adjust filename as needed
    sequences = df['sequence'].apply(encode_protein_sequence)
    labels = df['label']

    # Convert to PyTorch tensors
    sequences_tensor = torch.tensor(sequences.tolist(), dtype=torch.float32)
    labels_tensor = torch.tensor(labels.tolist(), dtype=torch.long)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sequences_tensor, labels_tensor, test_size=0.2)

    # Model initialization
    model = InfluenzaVirusModel(input_size=NUM_AMINO_ACIDS, hidden_size=128, output_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10  # Adjust as needed
    for epoch in range(epochs):
        model.train()
        for i, (sequence, label) in enumerate(zip(X_train, y_train)):
            optimizer.zero_grad()
            sequence = mutate_sequence(sequence)  # Introduce mutation
            sequence_tensor = torch.tensor([encode_protein_sequence(sequence)], dtype=torch.float32)
            output = model(sequence_tensor)
            loss = criterion(output, label.unsqueeze(0))
            loss.backward()
            optimizer.step()

        # Validation step (optional)

    # Testing loop
    model.eval()
    # Add testing logic here

if __name__ == "__main__":
    main()


