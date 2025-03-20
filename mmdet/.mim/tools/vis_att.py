import argparse
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.core import tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--show" or "'
         '--show-dir"')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    model = MMDataParallel(model, device_ids=[0])
    single_gpu_vis(model, data_loader, args.show, args.show_dir)


hidden_outputs = {}


def context_mask_hook(name):

    def hook(module, input, output):
        x = input[0]
        batch, channel, height, width = x.size()
        # [N, 1, H, W]
        context_mask = module.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, H, W]
        context_mask = module.softmax(context_mask)

        hidden_outputs[name] = context_mask.view(batch, height, width)

    return hook

def dkp_hook(name):

    def hook(module, input, output):
        x = input[0]
        # batch, channel, height, width = x.size()
        # # [N, 1, H, W]
        kq = module.dkpconv(x)
        kq=kq.permute(0,2,3,1).contiguous()
        b,c,h,w=x.shape
        kq_shape=kq.shape
        kq=kq.view(-1,module.scale)
        select_mask=torch.mm(kq,module.ko)
        select_mask=select_mask.view(kq_shape)
        select_mask=select_mask.permute(0,3,1,2).contiguous() #b,8,h,w
        
        feature=[ torch.ones([1,1,h,w])*(idx+1)  for idx in range(module.scale)] 
        feature=torch.cat(feature,dim=1) #1,8,h,w
        feature=feature.to('cuda')

        feature_mask = torch.zeros_like(select_mask).scatter_(1, select_mask.argmax(dim=1, keepdim=True), 1) #b,8,h,w
        select_mask=torch.sum(feature_mask*feature , dim=1)#B,4,C,W,H -> B,C,W,H

        # select_mask=select_mask.unsqueeze(1)
        # # [N, 1, H * W]
        # context_mask = context_mask.view(batch, 1, height * width)
        # # [N, H, W]
        # context_mask = module.softmax(context_mask)
        # mask=output/x
        # hidden_outputs[name] =torch.sum(mask,dim=1)
        # for idx in range(module.scale):
        hidden_outputs[name+"c"] = select_mask

    return hook

import torch.nn.functional as F
def relu1(input,inplace=False):
    return F.hardtanh(input, 0., 1., inplace)

def cbam_s_hook(name):

    def hook(module, input, output):
        x = input[0]
        batch, channel, height, width = x.size()
        # # [N, 1, H, W]
        # context_mask = module.conv_mask(x)

        # # [N, 1, H * W]
        # context_mask = context_mask.view(batch, 1, height * width)
        # # [N, H, W]
        # context_mask = module.softmax(context_mask)
        # identity=x
        # poolh = nn.AdaptiveAvgPool2d((None, 1))
        # poolw = nn.AdaptiveAvgPool2d((1, None))
        # x=torch.mean(x,dim=1,keepdim=True) #0.005 68.7:1epo
        # x_h=poolh(x).permute(0,1,3,2) #B,1,1,H
        # x_w=poolw(x)#B,1,1,
        # y_w=torch.matmul(x_h,identity) #B,C,1,W
        # y_h=torch.matmul(x_w,identity.permute(0,1,3,2)).permute(0,1,3,2)#B,C,H,1
        # y_w=relu1(y_w)
        # y_h=relu1(y_h)

        # mask=output-x
        # out=identity + y_w + y_h
        # mask=out-y_w-identity
        # mask=output-x
        # print(x.shape)
        # print(output.shape)
        hidden_outputs[name] = torch.sum(output,dim=1)/channel
        # torch.sum(mask,dim=1)/channel
        # for idx in range(16):
        #     hidden_outputs[name+"c"+str(idx)] = mask[:,idx]

    return hook




def register_cbam_s_hook(model):
    for module_name, module in model.module.named_modules():
        if 'CBAM_S' in str(
                module.__class__) and 'layer4.1' in module_name:
            module.register_forward_hook(cbam_s_hook(module_name))
            print(f'{module_name} is registered')



def single_gpu_vis(model, data_loader, show=False, out_dir=None):
    model.eval()
    register_cbam_s_hook(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                img_show = mmcv.bgr2rgb(img_show)

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                for hidden_name, hidden_output in hidden_outputs.items():
                    plt.imshow(img_show)

                    hidden_output_np = hidden_output.detach().cpu().numpy()[0]
                    att_map = hidden_output_np.copy()
                    att_map[hidden_output_np < (np.percentile(
                        hidden_output_np, 80))] = hidden_output_np.min()
                    att_map[hidden_output_np > (
                        np.percentile(hidden_output_np, 95))] = np.percentile(
                            hidden_output_np, 95)
                    hidden_output_show=hidden_output_np
                    hidden_output_show = mmcv.imresize_like(
                        hidden_output_np, img_show)
                    # plt.imshow(hidden_output_show / hidden_output_show.max(),
                    #            cmap='viridis',
                    #            interpolation='bilinear', vmin=0., vmax=1.,
                    #            alpha=0.5)
                    plt.imshow(
                        hidden_output_show,
                        cmap='jet',
                        # cmap='gnuplot',
                        # interpolation='none',
                        interpolation='bilinear',
                        alpha=0.3)
                    if out_dir is not None:
                        dst_dir = osp.join(out_dir, hidden_name)
                        mmcv.mkdir_or_exist(dst_dir)
                        filename = img_meta['ori_filename']
                        print(f'saving {osp.join(dst_dir, filename)}')
                        plt.savefig(
                            osp.join(dst_dir, img_meta['ori_filename']))
                    else:
                        plt.title(img_meta['ori_filename'] + hidden_name)
                        plt.show()
                    plt.clf()

        hidden_outputs.clear()

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()


if __name__ == '__main__':
    main()
