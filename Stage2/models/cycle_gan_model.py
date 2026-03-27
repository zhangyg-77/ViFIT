import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import util.util as util
import torch.autograd
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import os
def get_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def content_loss(self, real_A, fake_A, real_B, fake_B):
    # print('self.rec_A -----> ',self.rec_A.shape)
    # print('self.real_A -----> ',self.real_A.shape)
    # print('self.rec_B -----> ',self.rec_B.shape)
    # print('self.real_B -----> ',self.real_B.shape)
    # print('self.fake_A -----> ',self.fake_A.shape)
    # print('self.fake_B -----> ',self.fake_B.shape)

    L1_function = torch.nn.L1Loss()
    real_A_mean = torch.mean(real_A,dim=1,keepdim=True)
    real_B_mean = torch.mean(real_B,dim=1,keepdim=True)
    fake_A_mean = torch.mean(fake_A,dim=1,keepdim=True)
    fake_B_mean = torch.mean(fake_B,dim=1,keepdim=True)

    real_A_normal = (real_A_mean - (self.opt.threshold_A/127.5-1))*10
    real_B_normal = (real_B_mean - (self.opt.threshold_B/127.5-1))*10

    fake_A_normal = (fake_A_mean - (self.opt.threshold_B/127.5-1))*10
    fake_B_normal = (fake_B_mean - (self.opt.threshold_A/127.5-1))*10

    real_A_sigmoid = torch.sigmoid(real_A_normal)
    real_B_sigmoid = 1 - torch.sigmoid(real_B_normal)

    fake_A_sigmoid = torch.sigmoid(fake_A_normal)
    fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)

    content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )
    content_loss_B = L1_function( fake_A_sigmoid , real_B_sigmoid )

    content_loss_rate = 50*np.exp(-(self.opt.counter/self.opt.batch_size))
    content_loss = (content_loss_A + content_loss_B)#*content_loss_rate
    return content_loss

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.8, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--threshold_A', type=float, default=255, help='')
            parser.add_argument('--threshold_B', type=float, default=220, help='')
            parser.add_argument('--counter', type=int, default=0, help='')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','cls_AB','cls_BA' ,'lab_A' , 'lab_B','ap_AC']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        


        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','G_C', 'G_D']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        self.opt = opt
        self.cls = opt.cls
        self.lab = opt.lab
        self.netG = opt.netG
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_C = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_D = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # dist.init_process_group(backend='nccl', init_method='env://')
        # self.netG_A = nn.parallel.DistributedDataParallel(self.netG_A,device_ids=[opt.gpu_ids], broadcast_buffers=False,find_unused_parameters=True)
        # self.netG_B = nn.parallel.DistributedDataParallel(self.netG_B,device_ids=[opt.gpu_ids], broadcast_buffers=False,find_unused_parameters=True)
        # self.netG_C = nn.parallel.DistributedDataParallel(self.netG_C,device_ids=[opt.gpu_ids], broadcast_buffers=False,find_unused_parameters=True)
        # self.netG_B2 = nn.parallel.DistributedDataParallel(self.netG_B2,device_ids=[opt.gpu_ids], broadcast_buffers=False,find_unused_parameters=True)
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_C = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_D = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_D_pool = ImagePool(opt.pool_size)
            
            self.real_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_C_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_D_pool = ImagePool(opt.pool_size)
                
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.labloss = torch.nn.L1Loss()
            self.cls_loss = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_G2 = torch.optim.Adam(itertools.chain(self.netG_C.parameters(), self.netG_D.parameters()), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D2 = torch.optim.Adam(itertools.chain(self.netD_C.parameters(), self.netD_D.parameters()), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D2)
            self.trans = get_transform()
    def set_input(self, input,total_iters=30001,dataset_size = 10):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.total_iters = total_iters
        self.dataset_size = dataset_size
        if self.total_iters<=self.dataset_size:
            self.real_D = input['D'].to(self.device)
            self.real_C = input['C'].to(self.device)
        self.x = os.getcwd()
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.lab:
            self.B_label = torch.tensor(self.B_label, dtype=torch.float32).view(-1, 1, 1, 1)
            self.A_label = float(self.A_label[0])
            self.A_label = torch.tensor(self.A_label, dtype=torch.float32).view(-1, 1, 1, 1)
            self.B_label = F.interpolate(self.B_label, size=(self.real_A.shape[2], self.real_A.shape[3]), mode='nearest')
            self.A_label = F.interpolate(self.A_label, size=(self.real_A.shape[2], self.real_A.shape[3]), mode='nearest')
        # if self.cls:
        #     if self.lab:
        #         self.fake_B , self.cls_fake_B , self.fake_B_label= self.netG_A(self.real_A , self.B_label)  # G_A(A)
        #         self.rec_A , self.cls_rec_A , self.rec_fake_A_label= self.netG_B(self.fake_B , self.A_label)   # G_B(G_A(A))
        #         self.fake_A , self.cls_fake_A , self.fake_A_label = self.netG_B(self.real_B , self.A_label )  # G_B(B)
        #         self.rec_B , self.cls_rec_B  , self.rec_fake_B_label= self.netG_A(self.fake_A , self.B_label)   # G_A(G_B(B))
        #     else:
        #         lab =  torch.tensor([0])
        #         self.fake_B , self.cls_fake_B = self.netG_A(self.real_A ,lab )  # G_A(A)
        #         self.rec_A , self.cls_rec_A = self.netG_B(self.fake_B ,lab)   # G_B(G_A(A))
        #         self.fake_A , self.cls_fake_A  = self.netG_B(self.real_B ,lab)  # G_B(B)
        #         self.rec_B , self.cls_rec_B  = self.netG_A(self.fake_A ,lab )   # G_A(G_B(B))
            
        #         if self.total_iters<=self.dataset_size:
        #             self.fake_B2 , self.cls_fake_B2 = self.netG_C(self.real_C ,lab )  # G_A(A)
        #             self.rec_C , self.cls_rec_C = self.netG_B2(self.fake_B2 ,lab)   # G_B(G_A(A))
        #             self.fake_C , self.cls_fake_C  = self.netG_B2(self.real_B ,lab)  # G_B(B)
        #             self.rec_B2 , self.cls_rec_B2  = self.netG_C(self.fake_C ,lab )   # G_A(G_B(B))

        else:
            if self.netG == 'resnet_9blocks':
                lab =  torch.tensor([0])
                self.fake_B  = self.netG_A(self.real_A )  # G_A(A)
                self.rec_A   = self.netG_B(self.fake_B  )   # G_B(G_A(A))
                self.fake_A  = self.netG_B(self.real_B  )  # G_B(B)
                self.rec_B   = self.netG_A(self.fake_A  )   # G_A(G_B(B))
                if self.total_iters<=self.dataset_size:
                    self.fake_D  = self.netG_C(self.real_C  )  # G_A(A)
                    self.rec_C  = self.netG_D(self.fake_D )   # G_B(G_A(A))
                    self.fake_C   = self.netG_D(self.real_D )  # G_B(B)
                    self.rec_D   = self.netG_C(self.fake_C  )   # G_A(G_B(B))
            else:
                self.fake_B  = self.netG_A(self.real_A  )  # G_A(A)
                self.rec_A   = self.netG_B(self.fake_B  )   # G_B(G_A(A))
                self.fake_A  = self.netG_B(self.real_B  )  # G_B(B)
                self.rec_B   = self.netG_A(self.fake_A  )   # G_A(G_B(B))
                
        if self.total_iters%100 == 0:
            save_path = self.x + '/checkpoints/' + self.opt.name +'/web/img/'
            # total_iters  = str(self.total_iters )
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            real_A = util.tensor2im(self.real_A)
            fake_A = util.tensor2im(self.fake_A)
            rec_A = util.tensor2im(self.rec_A)
            util.save_image(real_A, save_path + '/' + str(self.total_iters) +'real_A.png', aspect_ratio=1.0)
            util.save_image(fake_A, save_path + '/' + str(self.total_iters) + 'fake_A.png', aspect_ratio=1.0)
            util.save_image(rec_A, save_path + '/' + str(self.total_iters) + 'rec_A.png', aspect_ratio=1.0)
            
            real_B = util.tensor2im(self.real_B)
            fake_B = util.tensor2im(self.fake_B)
            rec_B = util.tensor2im(self.rec_B)
            util.save_image(real_B, save_path + '/' + str(self.total_iters) +'real_B.png', aspect_ratio=1.0)
            util.save_image(fake_B, save_path + '/' + str(self.total_iters) + 'fake_B.png', aspect_ratio=1.0)
            util.save_image(rec_B, save_path + '/' + str(self.total_iters) + 'rec_B.png', aspect_ratio=1.0)
            if self.total_iters<=self.dataset_size:
                fake_D = self.fake_D[0]
                fake_D = transforms.ToPILImage()(fake_D)
                fake_D = np.array(fake_D)
                fake_D = 255 - fake_D
                fake_D = Image.fromarray(fake_D)
                
                fake_D_F = self.trans(fake_D).to('cpu')
                fake_D_F = fake_D_F.cpu().float().numpy()
                fake_D_F = (np.transpose(fake_D_F, (1, 2, 0)) + 1) / 2.0 * 255.0
                fake_D_F = fake_D_F.astype(np.uint8)  # 将数组转换为 uint8 数据类型
                util.save_image(fake_D_F, save_path + '/' + str(self.total_iters) +'fake_D_2.png', aspect_ratio=1.0)
                real_C = util.tensor2im(self.real_C)
                util.save_image(real_C, save_path + '/' + str(self.total_iters) +'real_C.png', aspect_ratio=1.0)
                
                fake_D = util.tensor2im(self.fake_D)
                util.save_image(fake_D, save_path + '/' + str(self.total_iters) + 'fake_D.png', aspect_ratio=1.0)
            
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 
        if loss_D>0.1:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""

        if self.total_iters<=self.dataset_size:
            fake_D = self.fake_D_pool.query(self.fake_D)
            real_D = self.real_D_pool.query(self.real_D)
            self.loss_D_D = self.backward_D_basic(self.netD_D, real_D, fake_D)
            fake_B = self.fake_B_pool.query(self.fake_B)
            real_B = self.real_B_pool.query(self.real_B)
            self.loss_D_B = self.backward_D_basic(self.netD_B, real_B, fake_B)
        else:
            
            fake_B = self.fake_B_pool.query(self.fake_B)
            real_B = self.real_B_pool.query(self.real_B)
            if self.loss_G_B <1.5:
                self.loss_D_B = self.backward_D_basic(self.netD_B, real_B, fake_B) 
            else:
                self.loss_D_B = 0
            
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""

        if self.total_iters<=self.dataset_size:
            fake_C = self.fake_C_pool.query(self.fake_C)
            real_C = self.real_C_pool.query(self.real_C)
            self.loss_D_C = self.backward_D_basic(self.netD_C, real_C, fake_C)
            fake_A = self.fake_A_pool.query(self.fake_A)
            real_A = self.real_A_pool.query(self.real_A)
            self.loss_D_A = self.backward_D_basic(self.netD_A, real_A, fake_A)
        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
            real_A = self.real_A_pool.query(self.real_A)
            self.loss_D_A = self.backward_D_basic(self.netD_A, real_A, fake_A)
            
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            if self.cls:
                lab =  torch.tensor([0])
                # self.idt_C , C  = self.netG_C(self.real_B ,lab )
                # self.loss_idt_C = self.criterionIdt(self.idt_C, self.real_B) * lambda_B * lambda_idt
                # self.idt_B2 , B2 = self.netG_B2(self.real_C ,lab )
                # self.loss_idt_B2 = self.criterionIdt(self.idt_B2, self.real_C) * lambda_A * lambda_idt
            else:
                self.idt_C   = self.netG_C(self.real_D )
                self.loss_idt_C = self.criterionIdt(self.idt_C, self.real_D) * lambda_B * lambda_idt
                self.idt_D  = self.netG_D(self.real_C  )
                self.loss_idt_D = self.criterionIdt(self.idt_D, self.real_C) * lambda_A * lambda_idt
        else:
            self.loss_idt_C = 0
            self.loss_idt_D = 0
        # GAN loss D_A(G_A(A))
        self.loss_G_D = self.criterionGAN(self.netD_D(self.fake_D), True) 
        # GAN loss D_B(G_B(B))
        self.loss_G_C = self.criterionGAN(self.netD_C(self.fake_C), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_C = self.criterionCycle(self.rec_C, self.real_C) * lambda_A 
        self.loss_cycle_D = self.criterionCycle(self.rec_D, self.real_D) * lambda_B
        
        
        # middle cls colss  
        if self.cls:
            self.loss_cls_CB =  self.cls_loss (self.cls_fake_B2, self.cls_rec_C)* lambda_B * 12
            self.loss_cls_BC =  self.cls_loss (self.cls_fake_C, self.cls_rec_B2)* lambda_B * 12
        else:
            self.loss_cls_CB = 0
            self.loss_cls_BC = 0
        

        self.loss_G2 = self.loss_G_C + self.loss_G_D + self.loss_cycle_C + self.loss_cycle_D + self.loss_idt_C + self.loss_idt_D + self.loss_cls_BC + self.loss_cls_CB  
        self.loss_G2.backward()
        
    def backward_G2(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            if self.cls:
                    lab =  torch.tensor([0])
                    self.idt_A , A  = self.netG_A(self.real_B ,lab )
                    self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                    self.idt_B , B = self.netG_B(self.real_A ,lab )
                    self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A = self.netG_A(self.real_B )
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A ) 
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)  
        if self.loss_G_B >1.5:
            self.loss_G_B = self.loss_G_B*2
        # GAN loss D_B(G_B(B))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True) 
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        self.loss_cont_AABB = 0 #content_loss(self , real_A = self.real_A, fake_A = self.fake_A,real_B=self.real_B,fake_B=self.fake_B)

        # middle cls colss  
        if self.cls:
            self.loss_cls_AB =  self.cls_loss (self.cls_fake_B, self.cls_rec_A)* lambda_B * 12
            self.loss_cls_BA =  self.cls_loss (self.cls_fake_A, self.cls_rec_B)* lambda_B * 12
        else:
            self.loss_cls_AB = 0
            self.loss_cls_BA = 0
        if self.total_iters<=self.dataset_size:
            fake_D = self.fake_D[0]
            fake_B = self.fake_B[0]
            fake_D = transforms.ToPILImage()(fake_D)
            fake_D = np.array(fake_D)
            fake_D = 255 - fake_D
            fake_D = Image.fromarray(fake_D)
            fake_D = self.trans(fake_D).to(self.device)
            self.loss_ap_AC = self.criterionCycle(fake_B ,fake_D ) * lambda_B  * 2
            self.loss_ap_BD = 0 #self.criterionCycle(self.fake_A ,self.fake_C ) * lambda_B  * 2
            # self.loss_ap_CA = self.criterionCycle(self.fake_A , self.fake_C_F)
        else:
            self.loss_ap_AC = 0
            self.loss_ap_BD = 0
        print(self.loss_ap_AC,self.loss_ap_BD)
        self.loss_G = self.loss_G_A + self.loss_G_B  + self.loss_cycle_A + self.loss_cycle_B  + self.loss_idt_A + self.loss_idt_B  + self.loss_cls_BA + self.loss_cls_AB  +  self.loss_ap_AC  + self.loss_cont_AABB + self.loss_ap_BD
        self.loss_G.backward()
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        i = 1
        # self.forward()    
        # self.set_requires_grad([self.netD_A, self.netD_B,self.netD_C, self.netD_B2], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netD_C, self.netD_D], False)
        
        if self.total_iters<=self.dataset_size:
            self.loss_names =['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','cls_AB','cls_BA' ,'ap_AC','cont_AABB','ap_BD']# ['D_C', 'G_C', 'cycle_C', 'idt_C', 'D_D', 'G_D', 'cycle_D', 'idt_D','cls_CB','cls_BC' ]
            self.forward()
            self.optimizer_G2.zero_grad()
            self.backward_G()
            self.optimizer_G2.step()
            
            self.forward()
            self.optimizer_G.zero_grad() 
            self.backward_G2()
            self.optimizer_G.step()
            
            self.set_requires_grad([self.netD_C, self.netD_D], True)
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  
            self.optimizer_D2.zero_grad() 
            self.backward_D_A() 
            self.backward_D_B()
            self.optimizer_D2.step()
            self.optimizer_D.step()  # update D_A and D_B's weights
            
        else:
            
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','cls_AB','cls_BA' ,'ap_AC','cont_AABB','ap_BD']
            self.forward()
            self.optimizer_G.zero_grad() 
            self.backward_G2()
            self.optimizer_G.step()
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights
            
