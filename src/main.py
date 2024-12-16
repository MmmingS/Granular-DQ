import torch
import data
import utility
from model.srresnet import SRResNet
from model.srresnet_gdq import SRResNet_GDQ
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')
def main():
    if checkpoint.ok:
        loader = data.Data(args)
        if args.model == 'SRResNet':
            model = SRResNet_GDQ(args, bias=True, k_bits=args.k_bits).to(device)
        elif args.model == 'IDN':
            pass
        elif args.model == 'EDSR':
            pass
        elif args.model == 'SwinIR':
            pass
        elif args.model == 'HAT':
            pass
        else:
            raise ValueError('not expected model = {}'.format(args.model))
        if args.test_only:
            ckpt = torch.load(args.pre_train)
            checkpoints = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            for key in list(checkpoints.keys()):
                if 'fine_grain_ratito1' in key or 'fine_grain_ratito2' in key:
                    checkpoints[key] = checkpoints[key].squeeze(-1).squeeze(-1)
            model.load_state_dict(checkpoints)
        t = Trainer(args, loader, model, checkpoint)
        print(f'{args.save} start!')
        while not t.terminate():
            t.test()
        checkpoint.done()
        print(f'{args.save} done!')
if __name__ == '__main__':
    main()
