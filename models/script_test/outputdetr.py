import torch
from torch.utils.data import DataLoader

# import your dataset + sequence encoder
from models.scripts.SequencePointCloudDataset import SequencePointCloudDataset
from models.scripts.SequenceEncoder import SequenceEncoder
from models.scripts.TemporalEncoder import TemporalCLS
from models.scripts.DETRDecoder import JointSetDecoder, JointSlotHeads

def main():
    root_dir = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"  
    dataset = SequencePointCloudDataset(root_dir, num_points=10000, sequence_length=12, need_resampling=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_encoder = SequenceEncoder(use_backbone=True).to(device)
    temporal_model = TemporalCLS(input_dim=512, model_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_len=12).to(device)
    joint_decoder = JointSetDecoder().to(device)
    joint_heads = JointSlotHeads().to(device)

    model = torch.nn.Sequential(seq_encoder, temporal_model, joint_decoder, joint_heads)
    model.eval()

    # get one batch
    for batch in dataloader:
        sequences, labels = batch   # sequences: (B,T,P,3)
        print("Input sequences shape:", sequences.shape)

        with torch.no_grad():
            heads = model(sequences.to(device))  # (B,T,512)

        for head in heads:
            print(f"Shape of {head}: {heads[head].shape}")

        # print("Heads shape:", heads)  # expect (2, 6, 512)
        print("joint_count shape:", labels["joint_count"].shape)
        print("joint_params shape:", labels["joint_params"].shape)
        print("joint_deltas shape:", labels["joint_deltas"].shape)
        break  # just test first batch

if __name__ == "__main__":
    main()
