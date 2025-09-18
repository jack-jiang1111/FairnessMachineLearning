import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='Disables CUDA training.')
	parser.add_argument('--cuda_device', type=int, default=0,
	                    help='cuda device running on.')
	parser.add_argument('--dataset', type=str, default='bail',
	                    help='a dataset from credit, german and bail.')
	parser.add_argument('--epochs', type=int, default=1000,
	                    help='Number of epochs to train.')
	parser.add_argument('--lr', type=float, default=1e-3,
	                    help='Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=1e-5,
	                    help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=8,
	                    help='Number of hidden units.')
	parser.add_argument('--dropout', type=float, default=0.3,
	                    help='Dropout rate (1 - keep probability).')
	parser.add_argument('--seed', type=int, default=0,
	                    help='seed.')
	parser.add_argument('--model', type=str, default='mlp',
	                    help='mlp model.')
	parser.add_argument('--topK', type=int, default=1,
	                    help='features to be masked when computing fidelity.')
	parser.add_argument('--lambda_', type=float, default=1.0,
	                    help='lambda_: coefficient for fairness loss')
	parser.add_argument('--top_ratio', type=float, default=0.2,
	                    help='top_ratio.')
	parser.add_argument('--opt_start_epoch', type=int, default=400,
	                    help='the epoch we start optimization')
	# Adversarial training specific options
	parser.add_argument('--adv_hidden', type=int, default=16,
	                    help='Number of hidden units in adversary.')
	parser.add_argument('--adv_lr', type=float, default=1e-3,
	                    help='Learning rate for adversary.')
	parser.add_argument('--adv_weight_decay', type=float, default=1e-5,
	                    help='Weight decay for adversary.')
	parser.add_argument('--adv_lambda', type=float, default=2.0,
	                    help='Lambda for gradient reversal.')
	parser.add_argument('--adv_steps', type=int, default=1,
	                    help='Number of adversary steps per epoch.')
	parser.add_argument('--adv_weight', type=float, default=2.0,
	                    help='Weight for adversarial loss in total loss (total_loss = y_loss + adv_weight * adv_loss).')
	# Similarity-based split options
	parser.add_argument('--use_similarity_split', action='store_true', default=False,
	                    help='Use the new similarity-based split with fair noise.')
	parser.add_argument('--t1', type=float, default=0.6,
	                    help='Training ratio for similarity-based split.')
	parser.add_argument('--t2', type=float, default=0.2,
	                    help='Validation ratio for similarity-based split.')
	parser.add_argument('--t3', type=float, default=0.2,
	                    help='Testing ratio for similarity-based split.')
	parser.add_argument('--fair_noise', type=float, default=0.15,
	                    help='Fair noise probability (0-1).')
	parser.add_argument('--max_split_seed_tries', type=int, default=20,
	                    help='Max number of seed retries to ensure train fairness > val/test fairness.')
	parser.add_argument('--split_seed_base', type=int, default=42,
	                    help='Base seed value for split seeding retries.')
	return parser.parse_args()