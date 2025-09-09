import torch
from torch.utils.data import DataLoader

# import your dataset + sequence encoder
from models.scripts.SequencePointCloudDataset import SequencePointCloudDataset
from models.scripts.SequenceEncoder import SequenceEncoder
from models.scripts.TemporalEncoder import TemporalCLS
from models.scripts.DETRDecoder import JointSetDecoder, JointSlotHeads
from models.scripts.HungarianMatcher import HungarianJointMatcher
from models.scripts.GTAdapter import parse_gt_batch
from models.scripts.loss import JointSetLoss

def main():
    root_dir = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"  
    dataset = SequencePointCloudDataset(root_dir, num_points=10000, sequence_length=12, need_resampling=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_encoder = SequenceEncoder(use_backbone=True).to(device)
    temporal_model = TemporalCLS(input_dim=512, model_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_len=12).to(device)
    joint_decoder = JointSetDecoder().to(device)
    joint_heads = JointSlotHeads().to(device)
    matcher = HungarianJointMatcher()
    loss_fn = JointSetLoss(w_exist=1.0, w_type=1.0, w_axis=1.0, w_point=1.0,
                       w_rank_l1=1.0, w_rank_pair=0.5, pair_margin=0.1)

    model = torch.nn.Sequential(seq_encoder, temporal_model, joint_decoder, joint_heads)
    model.eval()

    # get one batch
    for batch in dataloader:
        sequences, labels = batch   # sequences: (B,T,P,3)
        print("Input sequences shape:", sequences.shape)
        sequences = sequences.to(device)

        with torch.no_grad():
            heads = model(sequences)  # (B,T,512)
            # apply GT parsing
            gt_batch = parse_gt_batch(labels, device=device)

            assignments = matcher(heads, gt_batch)
            loss, stats = loss_fn(heads, gt_batch, assignments)
            print("Loss:", stats)

        break  # just test first batch

if __name__ == "__main__":
    main()
