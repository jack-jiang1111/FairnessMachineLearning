import argparse
from .trainer import MoETrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='bail')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda_device', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--gate_lr', type=float, default=1e-3)
    p.add_argument('--entropy_coeff', type=float, default=1e-3)
    p.add_argument('--lb_coeff', type=float, default=1e-3)
    p.add_argument('--lambda_rep', type=float, default=1.0)
    p.add_argument('--lambda_fair', type=float, default=1.0) #model 2 focus on more fairness
    p.add_argument('--lambda_attention', type=float, default=1.0)
    p.add_argument('--lambda_adv', type=float, default=1.0)
    p.add_argument('--skip_gate', action='store_true', 
                   help='Skip gate training - experts only mode for hyperparameter tuning')
    # Caching / skip-pretrain options
    p.add_argument('--use_cached_experts', action='store_true', help='Load cached pretrained experts and skip pretraining')
    p.add_argument('--cache_dir', type=str, default='weights/moe_experts', help='Directory to cache expert weights')
    p.add_argument('--use_cached_gate', action='store_true', help='Load cached gate and skip gate training')
    return p.parse_args()


def main():
    args = parse_args()
    trainer = MoETrainer(dataset=args.dataset, seed=args.seed, cuda_device=args.cuda_device,
                         epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                         lambda_rep=args.lambda_rep, lambda_fair=args.lambda_fair,
                         lambda_attention=args.lambda_attention, lambda_adv=args.lambda_adv,
                         gate_lr=args.gate_lr, entropy_coeff=args.entropy_coeff, lb_coeff=args.lb_coeff,
                         use_cached_experts=args.use_cached_experts, cache_dir=args.cache_dir,
                         use_cached_gate=args.use_cached_gate, skip_gate=args.skip_gate)
    stats = trainer.run()
    print("Final MoE Test Stats:", stats)


if __name__ == '__main__':
    main()


