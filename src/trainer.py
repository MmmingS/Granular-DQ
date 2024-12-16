import numpy as np
import torch
from tqdm import tqdm
import utility
from utils import common as util
import numpy as np

class Trainer():
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.scale = args.scale
        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.entropy_thred1 = 0.50
        self.entropy_thred2 = 0.80
        self.resume_epoch = 0
        self.device =  torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')
        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.model.load_state_dict(ckpt['state_dict'])
            self.resume_epoch = ckpt['epoch']
            # self.epoch -= self.resume_epoch
        # --------------- Print # Params ---------------------
        n_params = 0
        for p in list(model.parameters()):
            n_p=1
            for s in list(p.size()):
                n_p = n_p*s
            n_params += n_p
        self.ckp.write_log('Parameters: {:.1f}K'.format(n_params/(1e+3)))
        self.test_patch_size = args.patch_size 
        self.step_size = args.step_size
        
    def test(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        model = self.model
        model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                if self.args.test_patch:
                    d.dataset.set_scale(idx_scale)
                    i = 0
                    tot_bits = 0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        i += 1        
                        lr, hr = self.prepare(lr, hr)
                        lr_list, num_h, num_w, h, w = self.crop(lr[0], self.test_patch_size, self.step_size)
                        hr_list=self.crop(hr[0], self.test_patch_size*self.args.scale[0], self.step_size*self.args.scale[0])[0]
                        sr_list = []
                        p = 0
                        tot_bits_image = 0
                        avg_bit = 0
                        for lr_sub_img, hr_sub_img in zip (lr_list, hr_list):
                            sr_sub, _, _, _,s_bits = model(lr_sub_img.unsqueeze(0),[self.entropy_thred1,self.entropy_thred2])
                            if self.args.model == 'IDN':
                                avg_bit = s_bits.item() / 6. / self.args.n_resblocks
                            else:
                                avg_bit = s_bits.item()/self.args.n_resblocks / 2.
                            tot_bits_image += avg_bit
                            patch_psnr = utility.calc_psnr(sr_sub, hr_sub_img, scale, self.args.rgb_range, dataset=d)
                            self.ckp.write_log(
                                '{}-{:3d}: {:.2f} dB, {:.2f} avg bits'.format(filename[0], p, patch_psnr, avg_bit))
                            sr_sub = utility.quantize(sr_sub, self.args.rgb_range)
                            sr_list.append(sr_sub)
                            p+=1
                        sr = self.combine(sr_list, num_h, num_w, h, w, self.test_patch_size, self.step_size)
                        sr = sr.unsqueeze(0)
                        save_list = [sr]
                        cur_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                        cur_ssim = utility.calc_ssim(sr, hr, scale, benchmark=d.dataset.benchmark)
                        self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                        self.ckp.bit_log[-1, idx_data, idx_scale] += tot_bits_image / p
                        self.ckp.ssim_log[-1, idx_data, idx_scale] += cur_ssim
                        tot_bits += tot_bits_image/p 
                        self.ckp.write_log(
                            '\n[{}] PSNR: {:.3f} dB; SSIM: {:.3f}; Avg_bit: {:.2f}; Num_patch: {}'.format(
                                filename[0],
                                cur_psnr,
                                cur_ssim,
                                tot_bits_image/p,
                                p
                            )
                        )
                        if self.args.save_gt:
                            save_list.extend([lr, hr])
                        if self.args.save_results:
                            save_name = '{}_{:.2f}'.format(filename[0], cur_psnr)
                            self.ckp.save_results(d, save_name, save_list, scale)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                    best_psnr = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}] PSNR: {:.3f} SSIM:{:.3f} (Best PSNR: {:.3f} @epoch {}) {:.2f} bits t1:{:.2f}t2:{:.2f}'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            self.ckp.ssim_log[-1, idx_data, idx_scale],
                            best_psnr[0][idx_data, idx_scale],
                            best_psnr[1][idx_data, idx_scale] + 1 + self.resume_epoch,
                            tot_bits / len(d),
                            self.entropy_thred1,
                            self.entropy_thred2
                        )
                    )
        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            is_best_psnr = (best_psnr[1][0, 0] + 1 == self.epoch)
            state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            }
            util.save_checkpoint(state, is_best_psnr, checkpoint =self.ckp.dir + '/model')
            util.plot_psnr(self.args, self.ckp.dir, self.epoch - self.resume_epoch, self.ckp.log)
            util.plot_bit(self.args, self.ckp.dir, self.epoch - self.resume_epoch, self.ckp.bit_log) 
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
    
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

    def crop(self, img, crop_sz, step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            c, h, w = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, max(h - crop_sz,0) + 1, step)
        w_space = np.arange(0, max(w - crop_sz,0) + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    if x == h_space[-1]:
                        if y == w_space[-1]:
                            crop_img = img[:, x:h, y:w]
                        else:
                            crop_img = img[:, x:h, y:y + crop_sz]
                    elif y == w_space[-1]:
                        crop_img = img[:, x:x + crop_sz, y:w]
                    else:    
                        crop_img = img[:, x:x + crop_sz, y:y + crop_sz]
                lr_list.append(crop_img)
        return lr_list, num_h, num_w, h, w
    


    def combine(self,sr_list,num_h, num_w,h,w,patch_size,step):
        index=0

        sr_img = torch.zeros((3, h*self.scale[0], w*self.scale[0])).to(self.device)
        s = int(((patch_size - step) / 2)*self.scale[0])
        index1=0
        index2=0
        if num_h == 1:
            if num_w ==1:
                sr_img[:,:h*self.scale[0],:w*self.scale[0]]+=sr_list[index][0]
            else:
                for j in range(num_w):
                    y0 = j*step*self.scale[0]
                    if j==0:
                        sr_img[:,:,y0:y0+s+step*self.scale[0]]+=sr_list[index1][0][:,:,:s+step*self.scale[0]]
                    elif j==num_w-1:
                        sr_img[:,:,y0+s:w*self.scale[0]]+=sr_list[index1][0][:,:,s:]
                    else:
                        sr_img[:,:,y0+s:y0+s+step*self.scale[0]]+=sr_list[index1][0][:,:,s:s+step*self.scale[0]]
                    index1+=1

        elif num_w ==1:
            for i in range(num_h):
                x0 = i*step*self.scale[0]
                if i==0:
                    sr_img[:,x0:x0+s+step*self.scale[0],:]+=sr_list[index2][0][:,:s+step*self.scale[0],:]
                elif i==num_h-1:
                    sr_img[:,x0+s:h*self.scale[0],:]+=sr_list[index2][0][:,s:,:]
                else:
                    sr_img[:,x0+s:x0+s+step*self.scale[0],:]+=sr_list[index2][0][:,s:s+step*self.scale[0],:]
                index2+=1

        else:
            for i in range(num_h):
                for j in range(num_w):
                    x0 = i*step*self.scale[0]
                    y0 = j*step*self.scale[0]

                    if i==0:
                        if j==0:
                            sr_img[:,x0:x0+s+step*self.scale[0],y0:y0+s+step*self.scale[0]]+=sr_list[index][0][:,:s+step*self.scale[0], :s+step*self.scale[0]]
                        elif j==num_w-1:
                            sr_img[:,x0:x0+s+step*self.scale[0],y0+s:w*self.scale[0]]+=sr_list[index][0][:,:s+step*self.scale[0],s:]
                        else:
                            sr_img[:,x0:x0+s+step*self.scale[0],y0+s:y0+s+step*self.scale[0]]+=sr_list[index][0][:,:s+step*self.scale[0], s:s+step*self.scale[0]]
                    elif j==0:
                        if i==num_h-1:
                            sr_img[:,x0+s:h*self.scale[0],y0:y0+s+step*self.scale[0]]+=sr_list[index][0][:,s:,:s+step*self.scale[0]]
                        else:
                            sr_img[:,x0+s:x0+s+step*self.scale[0],y0:y0+s+step*self.scale[0]]+=sr_list[index][0][:,s:s+step*self.scale[0], :s+step*self.scale[0]]
                    elif i==num_h-1:
                        if j==num_w-1:
                            sr_img[:,x0+s:h*self.scale[0],y0+s:w*self.scale[0]]+=sr_list[index][0][:,s:,s:]
                        else:
                            sr_img[:,x0+s:h*self.scale[0],y0+s:y0+s+step*self.scale[0]]+=sr_list[index][0][:,s:,s:s+step*self.scale[0]]
                    elif j==num_w-1:
                        sr_img[:,x0+s:x0+s+step*self.scale[0],y0+s:w*self.scale[0]]+=sr_list[index][0][:,s:s+step*self.scale[0],s:]
                    else:
                        sr_img[:,x0+s:x0+s+step*self.scale[0],y0+s:y0+s+step*self.scale[0]]+=sr_list[index][0][:,s:s+step*self.scale[0], s:s+step*self.scale[0]]
                    
                    index+=1

        return sr_img
