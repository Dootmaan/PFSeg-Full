import torch as pt
import numpy as np
from model.GuidedDSRL3D import GuidedPFSegFull
from medpy.metric.binary import jc,dc,hd95
from dataset.GuidedBraTSDataset3D import GuidedBraTSDataset3D
# from loss.FALoss3D import FALoss3D
import cv2
from loss.PathFusionLoss import PathFusionLossPlus
from loss.DiceLoss import BinaryDiceLoss
from config import config
from tensorboardX import SummaryWriter
# from sklearn.model_selection import KFold

lr=0.0001
epoch=80
batch_size=1
model_path='/newdata/why/Saved_models'
w_sr=0.5
w_pf=0.5
img_size=[64,96,96]
crop_size=[64,96,96]
# crop_size=config.crop_size
size=crop_size[2] #用于最后cv2显示

trainset=GuidedBraTSDataset3D('/newdata/why/BraTS20',mode='train',downsample_times=6)
valset=GuidedBraTSDataset3D('/newdata/why/BraTS20',mode='val')
testset=GuidedBraTSDataset3D('/newdata/why/BraTS20',mode='test',downsample_times=6)

train_dataset=pt.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True)
val_dataset=pt.utils.data.DataLoader(valset,batch_size=1,shuffle=True,drop_last=True)
test_dataset=pt.utils.data.DataLoader(testset,batch_size=1,shuffle=False,drop_last=True)

# allset=GuidedBraTSDataset3D('/newdata/why/BraTS20',mode='all')
# all_dataset=pt.utils.data.DataLoader(allset,batch_size=1,shuffle=False,drop_last=True)
# train_dataset=[]
# val_dataset=[]

model=GuidedPFSegFull(in_channels=1,out_channels=1).cuda()
# model.load_state_dict(pt.load(model_path+'/DSRL/6xinput_DSRL_3D_BraTS_withFullMS_dynamiccrop_val_guided_finetune_patch-free_bs1121208_best.pt',map_location = 'cpu'))  #!!!!

lossfunc_sr=pt.nn.MSELoss()
lossfunc_seg=pt.nn.BCELoss()
lossfunc_dice=BinaryDiceLoss()
lossfunc_pf=PathFusionLossPlus()
optimizer = pt.optim.Adam(model.parameters(), lr=lr)
scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=7)

writer=SummaryWriter(log_dir='./tensorboard_log',comment='train_loss')
# def TestPreprocess():
#     model.train()
#     print("Test phase finetune")
#     for i,data in enumerate(test_dataset):
#         (inputs,_,labels_sr,guidance,mask)=data
#         optimizer.zero_grad()

#         inputs = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
#         guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
#         mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
#         # labels_seg = pt.autograd.Variable(labels_seg).type(pt.FloatTensor).cuda().unsqueeze(1)
#         labels_sr = pt.autograd.Variable(labels_sr).type(pt.FloatTensor).cuda().unsqueeze(1)
#         outputs_seg,outputs_sr = model(inputs,guidance)
#         # loss_seg = lossfunc_seg(outputs_seg.detach(), outputs_seg.detach()) # unused
#         loss_sr = lossfunc_sr(outputs_sr, labels_sr)
#         # loss_pf = lossfunc_pf(outputs_seg,outputs_sr,labels_seg*labels_sr)
#         loss_guide=lossfunc_sr(mask*outputs_sr,mask*labels_sr)
#         # loss=loss_seg+w_sr*loss_sr

#         loss=w_sr*(loss_sr+loss_guide)

#         loss.backward()
#         optimizer.step()


def ValModel(val_dataset):
    model.eval()
    dice_sum=0
    hd_sum=0
    jc_sum=0
    weight_map=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
    for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
        for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
            for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                weight_map[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=1

    weight_map=1./weight_map
    for i,data in enumerate():
        output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
        label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

        (inputs,labels,_,guidance,mask)=data
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
            for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
                for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                    inputs3D = pt.autograd.Variable(inputs[:,a:(a+crop_size[0]),b:(b+crop_size[1]),c:(c+crop_size[2])]).type(pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D,_ = model(inputs3D,guidance)
                    outputs3D=np.array(outputs3D.cpu().data.numpy())
                    # outputs3D[outputs3D<0.5]=0
                    # outputs3D[outputs3D>=0.5]=1
                    output_list[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=outputs3D

        label_list=np.array(labels3D.cpu().data.numpy())

        output_list=np.array(output_list)*weight_map
        # label_list=np.array(label_list)

        output_list[output_list<0.5]=0
        output_list[output_list>=0.5]=1

        final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
        final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
        final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
        cv2.imwrite('TestPhase_BraTS.png',final_img)
        
        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        print("dice:",dice)

        # hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        # jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        # hd_sum+=hausdorff
        # jc_sum+=jaccard

    print("Finished. Total dice: ",dice_sum/len(val_dataset),'\n')
    print("Finished. Avg Jaccard: ",jc_sum/len(val_dataset))
    print("Finished. Avg hausdorff: ",hd_sum/len(val_dataset))
    return dice_sum/len(val_dataset)


def TestModel():
    model.eval()
    dice_sum=0
    hd_sum=0
    jc_sum=0
    dice_list=[]
    hd_list=[]
    jc_list=[]
    weight_map=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
    for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
        for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
            for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                weight_map[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=1

    weight_map=1./weight_map
    for i,data in enumerate(test_dataset):
        output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
        label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

        (inputs,labels,_,guidance,mask)=data
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
            for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
                for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                    inputs3D = pt.autograd.Variable(inputs[:,a:(a+crop_size[0]),b:(b+crop_size[1]),c:(c+crop_size[2])]).type(pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D,_ = model(inputs3D,guidance)
                    outputs3D=np.array(outputs3D.cpu().data.numpy())
                    # outputs3D[outputs3D<0.5]=0
                    # outputs3D[outputs3D>=0.5]=1
                    output_list[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=outputs3D

        label_list=np.array(labels3D.cpu().data.numpy())

        output_list=np.array(output_list)*weight_map
        # label_list=np.array(label_list)

        output_list[output_list<0.5]=0
        output_list[output_list>=0.5]=1

        final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
        final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
        final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
        cv2.imwrite('TestPhase_BraTS.png',final_img)
        
        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        dice_list.append(dice)

        try:
            hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        except:
            hausdorff=0

        jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        # print("dice:",dice,";hd95:",hausdorff,";jaccard:",jaccard)

        hd_sum+=hausdorff
        jc_sum+=jaccard
        hd_list.append(hausdorff)
        jc_list.append(jaccard)

    print("Finished. Test Total dice: ",dice_sum/len(test_dataset),'(',np.std(dice_list),')','\n')
    print("Finished. Test Avg Jaccard: ",jc_sum/len(test_dataset),'(',np.std(jc_list),')')
    print("Finished. Test Avg hausdorff: ",hd_sum/len(test_dataset),'(',np.std(hd_list),')')
    return dice_sum/len(test_dataset)

# TestPreprocess()
TestModel()
# raise Exception("end of test")

# TestModel()
# best_dice_sum=0
# data_induce = np.arange(0, allset.__len__())
# kf = KFold(n_splits=5)
# fold=1
# for train_index, val_index in kf.split(data_induce):
#     model=GuidedDSRL3D(in_channels=1,out_channels=1).cuda()
#     print('Fold',fold,'start')
#     train_subset = pt.utils.data.dataset.Subset(allset, train_index)
#     val_subset = pt.utils.data.dataset.Subset(allset, val_index)
#     train_dataset = pt.utils.data.DataLoader(train_subset,batch_size=1,shuffle=False,drop_last=True)
#     val_dataset = pt.utils.data.DataLoader(val_subset,batch_size=1,shuffle=False,drop_last=True)

#     optimizer = pt.optim.Adam(model.parameters(), lr=lr)
#     scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=20)
    # scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# TestPreprocess()
# TestModel()
# raise Exception("end of test")

best_dice=0
for x in range(epoch):
    model.train()
    loss_sum=0
    print('==>Epoch',x,': lr=',optimizer.param_groups[0]['lr'],'==>\n')

    for i,data in enumerate(train_dataset):
        (inputs,labels_seg,labels_sr,guidance,mask)=data
        optimizer.zero_grad()

        inputs = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        labels_seg = pt.autograd.Variable(labels_seg).type(pt.FloatTensor).cuda().unsqueeze(1)
        labels_sr = pt.autograd.Variable(labels_sr).type(pt.FloatTensor).cuda().unsqueeze(1)
        outputs_seg,outputs_sr = model(inputs,guidance)
        loss_seg = lossfunc_seg(outputs_seg, labels_seg)
        loss_sr = lossfunc_sr(outputs_sr, labels_sr)
        loss_pf = lossfunc_pf(outputs_seg,outputs_sr,labels_seg*labels_sr)
        loss_guide=lossfunc_sr(mask*outputs_sr,mask*labels_sr)
        # loss=loss_seg+w_sr*loss_sr

        loss=lossfunc_dice(outputs_seg,labels_seg)+loss_seg+w_sr*(loss_sr+loss_guide)+w_pf*loss_pf

        loss.backward()
        optimizer.step()

        # loss_sum+=loss.item()
        writer.add_scalar('training loss',loss.item(),i+x*len(train_dataset))

        if i%10==0:
            final_img=np.zeros(shape=(2*size,2*size*5))
            print('[epoch {:3d},iter {:5d}]'.format(x,i),'loss:',loss.item(),'; loss_seg:',loss_seg.item(),'; loss_sr:',loss_sr.item())
            final_img[:,0:(2*size)]=outputs_seg.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(2*size):(4*size)]=outputs_sr.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(4*size):(6*size)]=labels_seg.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(6*size):(8*size)]=labels_sr.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(8*size):]=cv2.resize(inputs.cpu().data.numpy()[0,0,size//8,:,:],((2*size),(2*size)))*255
            cv2.imwrite('dsrl_3d_combine.png',final_img)

    
    scheduler.step()

    print('==>End of epoch',x,'==>\n')

    print('===VAL===>')
    dice=ValModel()
    writer.add_scalar('testing DSC',dice,x)
    # scheduler.step(dice)
    if dice>best_dice:
        best_dice=dice
        print('New best dice! Model saved to',model_path+'/DSRL/6xinput_DSRL_3D_BraTS_withFullMS_dynamiccrop_val_guided_finetune_patch-free_bs'+str(batch_size)+'121208_best.pt')
        pt.save(model.state_dict(), model_path+'/DSRL/6xinput_DSRL_3D_BraTS_withFullMS_dynamiccrop_val_guided_finetune_patch-free_bs'+str(batch_size)+'121208_best.pt')
        print('===TEST===>')
        TestModel()
        # print('Fold',fold,'best', best_dice)
        # best_dice_sum+=best_dice
        # fold+=1

print('\nBest Dice:',best_dice)
writer.close()