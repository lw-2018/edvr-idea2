import importlib
import mmcv
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import numpy as np
from basicsr.models import networks as networks
from basicsr.models.base_model import BaseModel
from basicsr.utils import ProgressBar, get_root_logger, tensor2img
import torch.nn.functional as F
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.offset_frame= []
        self.offset_mask = []
        self.avg_feat_gt= []
        self.avg_feat_out = []
        

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
            
        ##load arcface 
        load_pth_name= '/home/wei/exp/EDVR/arcfacemodel/model_ir_se50.pth'
        pretrain_dict = torch.load( load_pth_name,map_location=lambda storage, loc: storage)
        # conv1_weight_new=np.zeros( (64,5,7,7) )
        # conv1_weight_new[:,:3,:,:]=pretrain_dict['conv1.weight'].cpu().data
        # pretrain_dict['conv1.weight']=torch.from_numpy(conv1_weight_new  )

        state_dict = self.net_g.state_dict()
        model_dict = state_dict
        keys = state_dict.keys()
    #    print('state_dict',state_dict)
        for k, v in pretrain_dict.items():
            # kk='backbone.'+k
            kk = k
            for key in keys:    
                if kk in key:
                    model_dict[key] = v
                    break
        state_dict.update(model_dict)
        self.net_g.load_state_dict(state_dict)
        
               # param.requires_grad = True

        
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_g.arcface.eval()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('offset_opt'):
            offset_type = train_opt['offset_opt'].pop('type')
            cri_offset_cls = getattr(loss_module, offset_type)
            self.cri_offset = cri_offset_cls(
                **train_opt['offset_opt']).to(self.device)
        else:
            self.cri_offset = None
            
        if train_opt.get('id_opt'):
            id_type = train_opt['id_opt'].pop('type')
            cri_id_cls = getattr(loss_module, id_type)
            self.cri_id = cri_id_cls(
                **train_opt['id_opt']).to(self.device)
        else:
            self.cri_id = None
            
       # print(self.cri_offset)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'flow' in data:
            self.flow = data['flow'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        #self.output, self.offset_frames,self.mask_frames = self.net_g(self.lq)
        #print('max:offset :', self.offset_frames.max())
        self.output, self.avg_feat_gt, self.avg_feat_out,_= self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
#        if self.cri_pix:
        if None:
            l_pix = self.cri_pix(self.output, self.gt)
#             if(int(current_iter/10000)%2==0):
#                 l_total += l_pix
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_id:
            l_id = self.cri_id(self.avg_feat_gt, self.avg_feat_out)
#             if(int(current_iter/10000)%2==0):
#                 l_total += l_pix
            l_total += l_id
            loss_dict['l_id'] = l_id
        # offset loss
        ######
        
       # weight  = (55000-20000)/50000.0
        #print(torch.mean(self.flow_7),torch.mean(self.flow))
        if False:
            l_offset = 0
            b,t,p,c,h,w = self.offset_frames.size()

            self.offset_frames = self.offset_frames.view(b,t,p*8,3,3,2,h,w).permute(2,3,4, 0,1,5,6,7)  ##16, 3(batchsize), 3, 3, 7, 2, 112, 112
            self.flow = F.pad(self.flow,(1,1,1,1),'constant',0)   #3(b) 7 2 450 450

            for group in self.offset_frames:
                for i in range(0,3):
                    for j in range(0,3):
                        #print(group[i][j].shape ,self.flow[:,:,:,i:i+h,j:j+h].shape)
                        l_offset += self.cri_offset(group[i][j],self.flow[:,:,:,i:i+h,j:j+h])
                       #print(abs(torch.sub(self.flow[:,:,:,1:1+h,1:1+h],group[1][1])).max())
                        
            l_total += l_offset
            #print('hhhh',l_offset)
            loss_dict['l_offset'] = l_offset
        
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        if True:
            gt = self.avg_feat_gt/torch.norm(self.avg_feat_gt,2,1).reshape(-1,1).repeat(1,self.avg_feat_gt.shape[1])
            out = self.avg_feat_out/torch.norm(self.avg_feat_out,2,1).reshape(-1,1).repeat(1,self.avg_feat_out.shape[1])
            cos_similarity = torch.mean(torch.sum(gt*out,1))
            loss_dict['cos_similarity'] = cos_similarity
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = ProgressBar(len(dataloader))

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
#             del self.lq
#             del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')

                mmcv.imwrite(sr_img, save_img_path)
#                 np.save('/home/wei/exp/EDVR/flow_save_160/offset.npy', visual['flow'])
#                 np.save('/home/wei/exp/EDVR/flow_save_160/mask.npy', visual['mask'])
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)


        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        out_dict['embedding_gt'] = self.output[1].detach().cpu().numpy()
        out_dict['embedding_out'] = self.output[2].detach().cpu().numpy()
        out_dict['embedding_center'] = self.output[3].detach().cpu().numpy()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
