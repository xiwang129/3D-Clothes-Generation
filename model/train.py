import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
import random 
import os
from dataloader import plyDataloader
import time
from tqdm import tqdm
from utilities import AverageValueMeter, dis_loss, gen_loss, pc_normalize,requires_grad,normalize_point_cloud
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


seed =123
random.seed(seed)
torch.cuda.manual_seed(seed)
num_classes = 2

# this file is modified from original author https://github.com/liruihui/SP-GAN
# major contribution includes finetuning, add conditional labels, and feature extractions

class Model(object):
    def __init__(self,opts):
        self.opts = opts
          
    def build_model(self):
        self.G = Generator(self.opts)
        self.D = Discriminator(self.opts)

        self.multi_gpu = False 

        if torch.cuda.device_count() > 1:
            print("let's use".torch.cuda.device_count(),"GPUs!")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.multi_gpu = True

        print('# generator parameters:', sum(param.numel() for param in self.G.parameters()))
        print('# discriminator parameters:', sum(param.numel() for param in self.D.parameters()))

        self.G.cuda()
        self.D.cuda()

    
        beta1 = 0.5
        beta2 = 0.99
        self.optimizerG = optim.Adam(filter(lambda p:p.requires_grad,self.G.parameters()),lr = self.opts.lr_g,betas = (beta1,beta2))
        self.optimizerD = optim.Adam(filter(lambda p:p.requires_grad,self.D.parameters()),lr = self.opts.lr_d,betas = (beta1,beta2))
        
        if self.opts.lr_decay:
            if self.opts.use_sgd:
                self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerG,self.opts.max_epoch, eta_min = self.opts.lr_g)
            else:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size = self.opts.lr_decay_feq, gamma = self.opts.lr_decay_rate)
        else:
            self.scheduler_G = None
        
        if self.opts.lr_decay:
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size = self.opts.lr_decay_feq, gamma = self.opts.lr_decay_rate)

        else: 
            self.scheduler_D = None
        
        self.z = torch.FloatTensor(self.opts.bs, self.opts.nz).cuda()
        self.z = Variable(self.z)

        self.ball =None
        self.fix_z = None

    def set_parameter_requires_grad(self,model,feature_extration):
        if feature_extration:
            for param in model.parameters():
                param.requires_grad = False
    
    def noise_generator(self,bs=1, masks=None):
        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0,self.opts.nv,(bs,self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0,self.opts.nv, (bs,1,self.opts.nz))
                noise = np.tile(noise,(1,self.opts.np,1))

        sim_noise = Variable(torch.tensor(noise)).cuda()
        return sim_noise
    
    def sphere_generator(self,bs=2, static=True):
        if self.ball is None:
            self.ball = np.loadtxt('2048.xyz')[:,:3]
            self.ball = pc_normalize(self.ball)

            N = self.ball.shape[0]
            xx = np.sum(self.ball **2, axis=(1)).reshape(N,1)
            yy = xx.T
            xy = -1 * xx @ yy 
            self.ball_dist = xy+xx+yy
        
        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball,(bs,1,1))
        
        else:
            ball = np.zeros((bs,self.opts.np,3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0],self.opts.np)
                ball[i] = self.ball[idx]
        
        ball = Variable(torch.tensor(ball)).cuda()
        return ball 
    

    def train(self):
        global epoch
        self.build_model()
        start_epoch = 1
 
        # load pretrained generator model
        if self.opts.pretrain_model_G:
            checkpoint = torch.load('human_G.pth',map_location=torch.device('cuda'))
            print("[*] LOAD SUCCESS")
            self.G.load_state_dict(checkpoint['G_model'] )
            weight_head = self.G.head[0].weight.clone()
            dim=128
            self.G.head[0] = nn.Conv1d(3+self.opts.nz+num_classes*4,dim,1).to(torch.double).cuda()

            with torch.no_grad():
                self.G.head[0].weight[:,:3+self.opts.nz,:] = weight_head
                self.G.head[0].weight[:,3+self.opts.nz:,:] = torch.full((dim,num_classes*4,1),0,dtype = torch.float32).to(torch.double)
                  
            self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
            print("[*] LOAD SUCCESS")
 
           
        self.log_string('PARAMETER ...')
        with open(os.path.join(self.opts.log_dir,'args.txt'),'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg+ ':'+str(getattr(self.opts,arg)) +'\n')
        # pprint(self.opts)
        self.writer = SummaryWriter()
        # self.writer = None

        '''DATA LOADING'''
        self.log_string('Load dataset ...')
        self.train_dataset = plyDataloader(self.opts)
        self.dataloader = DataLoader(self.train_dataset, batch_size= self.opts.bs, shuffle=True, num_workers=int(self.opts.workers),drop_last=True,pin_memory=True)
        self.num_batches = len(self.train_dataset) // self.opts.bs
  
        self.z_test = torch.FloatTensor(self.opts.bs,self.opts.nz).cuda()
        self.z_test.data.resize(self.opts.bs,self.opts.nz).normal_(0.0,1.0)

        start_time = time.time()
        d_avg_meter = AverageValueMeter()
        g_avg_meter = AverageValueMeter()
        real_acc_avg_meter = AverageValueMeter()
        fake_acc_avg_meter = AverageValueMeter()

        global_step = 0

        x = self.sphere_generator(bs=self.opts.bs)
        self.fix_z = self.noise_generator(bs=64)

        for epoch in range(start_epoch,self.opts.max_epoch+1):

            self.D.train()
            self.G.train()

            step_d = 0
            step_g = 0
          
            for idx, data in tqdm(enumerate(self.dataloader, 0),total=len(self.dataloader)):
                requires_grad(self.G, False)
                requires_grad(self.D, True)
           
                self.optimizerD.zero_grad()

                z = self.noise_generator(bs=self.opts.bs)
                label_embedding = nn.Embedding(num_classes,num_classes).cuda()
              
                # real labels, real pc
                dat,train_label = data
                real_points = Variable(dat,requires_grad=True)
                real_labels = Variable(torch.tensor(train_label),requires_grad=True).to(torch.long).cuda()
                real_labels = torch.reshape(real_labels,(self.opts.bs,self.opts.np,))

                # one hot encoding real labels 
                real_labels = F.one_hot(real_labels,num_classes = num_classes).float()

                
                # fake label, fake generated pc
                all_fake_labels = []
                for i in range(train_label.shape[0]):
                    fake_l = np.random.randint(0,num_classes,1)
                    fake_l = np.repeat(fake_l,self.opts.np)
                    all_fake_labels.append(fake_l)
                
                all_fake_labels = np.asarray(all_fake_labels)
                fake_labels = torch.from_numpy(all_fake_labels).cuda()

                # one hot encoding fake labels 
                fake_labels = F.one_hot(fake_labels,num_classes = num_classes).float()
                l = fake_labels.repeat(1,1,4)
          

                # l = label_embedding(fake_labels).cuda()

                z_l = torch.cat([z,l],dim=2)     
                d_fake_preds = self.G(x,z_l)
             

                real_points = real_points.transpose(2,1).cuda()
                d_fake_preds = d_fake_preds.detach()


                # real_l = label_embedding(real_labels).cuda()
                real_l = real_labels.cuda()
                real_l = real_l.transpose(2,1).contiguous()
           
                real_points_l = torch.cat([real_points,real_l],1)


                # fake_labels_d = label_embedding(fake_labels).cuda() 
                fake_labels_d = fake_labels.cuda()
                l_d = fake_labels_d.transpose(2,1).contiguous()
             
                d_fake_preds_l = torch.cat([d_fake_preds,l_d],1)
 
                
                d_real_logit = self.D(real_points_l.float())
                d_fake_logit = self.D(d_fake_preds_l.float())
               
                lossD,info = dis_loss(d_real_logit,d_fake_logit,gan=self.opts.gan, noise_label = self.opts.flip_d)

                lossD.backward()
                self.optimizerD.step()

                #train G
                requires_grad(self.G,True)
                requires_grad(self.D,False)

                feature_extration = False
                self.set_parameter_requires_grad(self.G,feature_extration)
            
                # freeze different layers in the generator 
                params_to_update = self.G.parameters()
                if feature_extration:
                    params_to_update = []
                    for name, param in self.G.named_parameters():
                        if 'tail' in name:
                        # if 'tail.4.weight' in name or 'tail.4.bias' in name:
                        # if 'tail' in name or 'global_conv' in name:
                            param.requires_grad = True
                            params_to_update.append(param)

                beta1 = 0.5 
                beta2 = 0.99
                if feature_extration:
                    self.optimizerG = optim.Adam(params_to_update,lr = self.opts.lr_g,betas = (beta1,beta2))

                self.optimizerG.zero_grad()

                z_g = self.noise_generator(bs =self.opts.bs)

                # create fake labels 
                all_fake_labels_g = []
                for i in range(train_label.shape[0]):
                    fake_lg = np.random.randint(0,num_classes,1)
                    fake_lg = np.repeat(fake_lg,self.opts.np)
                    all_fake_labels_g.append(fake_l)
                
                all_fake_labels_g = np.asarray(all_fake_labels_g)
                fake_labels_g = torch.from_numpy(all_fake_labels_g).cuda()

                fake_labels_g = F.one_hot(fake_labels_g,num_classes = num_classes).float()
             
               
                l_g = fake_labels_g.repeat(1,1,4)
                # l_g = label_embedding(fake_labels_g).cuda()
                z_l = torch.cat([z_g,l_g],dim=2)   
                g_fake_preds = self.G(x,z_l)

                l_g_fake = fake_labels_g
                l_g_fake = l_g_fake.transpose(2,1).contiguous() 
                g_fake_preds_l = torch.cat([g_fake_preds,l_g_fake],1)


                g_real_logit = self.D(real_points_l.float())
                g_fake_logit = self.D(g_fake_preds_l.float())
               
                lossG,_ = gen_loss(g_real_logit,g_fake_logit,gan = self.opts.gan, noise_label = self.opts.flip_g)

                lossG.backward()
                self.optimizerG.step()

                d_avg_meter.update(lossD.item())
                g_avg_meter.update(lossG.item())

                # real_acc_avg_meter.update(info['real_acc'])
                # fake_acc_avg_meter.update(info['fake_acc'])

                if self.writer is not None:
                    self.writer.add_scalar("loss/d_loss",lossD.data, global_step)
                    self.writer.add_scalar("loss/g_loss",lossG.data,global_step)
                    # self.writer.add_scalar("acc/real_acc",info['real_acc'],global_step)
                    # self.writer.add_scalar("acc/fake_acc",info['fake_acc'],global_step)
                    
                    self.writer.add_histogram("d_real_logit", d_real_logit,global_step)
                    self.writer.add_histogram("d_fake_logit",d_fake_logit, global_step)
                    self.writer.add_histogram("g_fake_logit",g_fake_logit,global_step)

                    self.writer.add_scalar("lr/lr_g",self.optimizerG.param_groups[0]['lr'],global_step)
                    self.writer.add_scalar("lr/lr_d",self.optimizerD.param_groups[0]['lr'],global_step)


                global_step +=1
                if self.opts.save and global_step%20 ==0:
                    requires_grad(self.G,False)
                    self.draw_sample_save(epoch,step = global_step)
                    requires_grad(self.G,True)
                
        
            if self.scheduler_G is not None:
                self.scheduler_G.step(epoch)
            if self.scheduler_D is not None:
                self.scheduler_D.step(epoch)
            
            time_tick = time.time() - start_time
            self.log_string("Epoch: [%2d] time: %2dm %2ds d_loss4: %.8f, g_loss: %.8f" \
                % (epoch, time_tick /60, time_tick %60, d_avg_meter.avg, g_avg_meter.avg))
            self.log_string("real_acc: %f fake_acc: %f" % (real_acc_avg_meter.avg, fake_acc_avg_meter.avg))
            self.log_string("lr_g: %f lr_d: %f" % (self.optimizerG.param_groups[0]['lr'], self.optimizerD.param_groups[0]['lr']))
            print("step_d: %d step_g:%d"%(step_d,step_g))
            if self.scheduler_G is not None and self.scheduler_D is not None:
                print("lr_g: %f  lr_d: %f"%(self.scheduler_G.get_lr()[0],self.scheduler_D.get_lr()[0]))

            requires_grad(self.G,False)
            requires_grad(self.D,True)

            if epoch % self.opts.snapshot == 0:
                self.save(self.opts.log_dir,epoch)
            
            if False and not self.opts.save:
                self.draw_sample(epoch)

        self.save(self.opts.log_dir,epoch)
        self.LOG_FOUT.close()

 
    def log_string(self,out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)
    
    def save(self,checkpoint_dir,index_epoch):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        save_name =  str(index_epoch) + '_' + self.opts.choice
        path_save_G = os.path.join(checkpoint_dir,save_name+'_G.pth')
        path_save_D = os.path.join(checkpoint_dir,save_name+'_D.pth')
        print('Save Path for G: {}'.format(path_save_G))
        print('Save Path for D: {}'.format(path_save_D))

        torch.save({'G_model': self.G.state_dict(),'G_optimizer': self.optimizerG.state_dict(),
        'G_epoch': index_epoch,},path_save_G)
        torch.save({'D_model': self.D.state_dict(),'D_optimizer': self.optimizerD.state_dict(),
        'D_epoch': index_epoch,},path_save_D)
    
  















                










