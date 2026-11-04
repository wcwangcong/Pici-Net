import torch
import h5py
import numpy as np
from PIL import Image
from model.generator import Test_Set, denormalize_speed
from model import network
from torch.utils.data import DataLoader
from util.util_tool import getInput
from util import cal_metric
from util.cal_metric import compute_flow
from scipy import io
import os
from model.network import Prediction_Net
from util.util_tool import getInput, get_GLCM
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from postprocess import smooth
import pandas as pd
def test(opt, test_data, speed_list, min_speed, max_speed, translation_list, min_translational_speed, max_translational_speed, train_val_test_lon_list, min_lon, max_lon, index, lead_time):
    f = open('./log/test.txt', 'a')
    f.write('Epoch' + '\t' +
            'rmse' + '\t' +
            'after_smooth' + '\t' +
            'mae' + '\t' +
            'after_smooth' + '\t' +
            'psnr' + '\t' +
            'sam' + '\t' +
            'ssim' + '\t' +
            'rmse_fuse' + '\t' +
            'WCPrNet_lead' + str(lead_time) + '_' + str(index) + '\n')
    # weight_path = "./weights/WCNet_" + str(index) + '/WCNET.pth'
    # weight_path = "./weights/WCNet_" + str(index) + '/epoch_42_weight/WCNET.pth'
    WCNET1_weight = './weights/predict/WCPr_lead' + str(lead_time)+'_'+str(index)+'/WCNET1.pth'
    WCNET2_weight = './weights/predict/WCPr_lead' + str(lead_time) + '_' + str(index) + '/WCNET2.pth'
    Prediction_weight = './weights/predict/WCPr_lead' + str(lead_time) + '_' + str(index) + '/Prediction_Net.pth'

    test_dataset = Test_Set(test_data[0], test_data[1], speed_list[2], min_speed, max_speed,
                                 translation_list[2], min_translational_speed, max_translational_speed,
                                 train_val_test_lon_list[2], min_lon, max_lon)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)
    WCNET1 = network.define_WCNET(input_ch=opt.lr_channels)
    WCNET2 = network.define_WCNET(input_ch=opt.lr_channels)
    prediction_net = Prediction_Net(opt.lr_channels).cuda()
    WCNET1.load_state_dict(torch.load(WCNET1_weight))
    WCNET2.load_state_dict(torch.load(WCNET2_weight))
    prediction_net.load_state_dict(torch.load(Prediction_weight))
    predict_sp = []
    label_sp = []
    path_list = []
    translational_speed_list = []
    lon_list = []
    structure_list = []
    speed1_list = []
    psnr2 = 0.0
    sam2 = 0.0
    ssim2 = 0.0
    rmse2 = 0.0

    psnr4 = 0.0
    sam4 = 0.0
    ssim4 = 0.0
    rmse4 = 0.0

    path_predict_dic = {}
    with torch.no_grad():
        prediction_net.eval()
        for i, data in enumerate(test_loader):
            data1_1, data1_2, data2_1, data2_2, path_name, label = data

            data_2km_1 = data1_1['data_2km_1']
            data_2km_2 = data1_2['data_2km_2']
            data_4km_1 = data2_1['data_4km_1']
            data_4km_2 = data2_2['data_4km_2']
            data_name = path_name['path_name']

            speed_label = label['speed'].float().cuda()
            translational_speed_label = label['translational_speed'].float().cuda()
            lon_label = label['lon'].float().cuda()
            translational_speed_label = translational_speed_label.unsqueeze(1)
            lon_label = lon_label.unsqueeze(1)

            input_2km_1, input_4km_1 = getInput(data_2km_1, data_4km_1)
            input_2km_2, input_4km_2 = getInput(data_2km_2, data_4km_2)
            input_2km_1 = input_2km_1.cuda()
            input_2km_2 = input_2km_2.cuda()
            input_4km_1 = input_4km_1.cuda()
            input_4km_2 = input_4km_2.cuda()
            data_2km_1 = data_2km_1.cuda()
            data_2km_2 = data_2km_2.cuda()
            data_4km_1 = data_4km_1.cuda()
            data_4km_2 = data_4km_2.cuda()

            rec_data, rec_eye, spat_edge1_12, spat_edge1_22, spat_edge2_12, spat_edge2_22, spec_edge12, spec_edge22, x_cloud_band = WCNET1(
                input_2km_2, input_4km_2)
            rec_data2, rec_eye2, spat_edge1_1, spat_edge1_2, spat_edge2_1, spat_edge2_2, spec_edge1, spec_edge2, x_cloud_band2 = WCNET2(input_2km_1, input_4km_1)#当前时刻的融合
            time_series = torch.sub(rec_data2, rec_data)
            # time_series =torch.abs(time_series)
            # psnr2 += cal_metric.compute_psnr(data_2km_2[0, :7, :, :], rec_data[0, :7, :, :])
            # sam2 += cal_metric.compute_SAM(data_2km_2[0, :7, :, :], rec_data[0, :7, :, :])
            # ssim2 += cal_metric.compute_SSIM(data_2km_2[0, :7, :, :], rec_data[0, :7, :, :])
            # rmse2 += cal_metric.compute_RMSE(data_2km_2[0, :7, :, :], rec_data[0, :7, :, :])

            psnr4 += cal_metric.compute_psnr(input_4km_2[0, 7:, :, :], rec_data[0, 7:, :, :])
            sam4 += cal_metric.compute_SAM(input_4km_2[0, 7:, :, :], rec_data[0, 7:, :, :])
            ssim4 += cal_metric.compute_SSIM(input_4km_2[0, 7:, :, :], rec_data[0, 7:, :, :])
            rmse4 += cal_metric.compute_RMSE(input_4km_2[0, 7:, :, :], rec_data[0, 7:, :, :])
            # predict_speed = prediction_net(rec_data, rec_data2, x_cloud_band2, translational_speed_label, lon_label, eye_GLCM)
            predict_speed, structure, speed1 = prediction_net(rec_data, rec_data2, x_cloud_band2, translational_speed_label, lon_label, spat_edge1_2, time_series)

            predict_speed = torch.mean(predict_speed, dim=1)
            structure = torch.mean(structure, dim=1)
            speed1 = torch.mean(speed1, dim=1)
            speed_label = denormalize_speed(speed_label, min_speed, max_speed)  # compute rmse
            translational_speed_label = denormalize_speed(translational_speed_label, min_translational_speed, max_translational_speed)
            lon_label = denormalize_speed(lon_label, min_lon, max_lon)
            predict_speed = denormalize_speed(predict_speed, min_speed, max_speed)
            for j in range(len(data_name)):   #tensor.item会取出数值类型
                predict_sp.append(predict_speed[j].item())
                label_sp.append(speed_label[j].item())
                translational_speed_list.append(translational_speed_label[j].item())
                lon_list.append(lon_label[j].item())
                structure_list.append(structure[j].item())
                speed1_list.append(speed1[j].item())
                path_list.append(data_name[j])
                path_predict_dic[data_name[j]] = [predict_speed[j].item(), speed_label[j].item()]

        rmse = mse(label_sp, predict_sp, squared=False)
        lmae = mae(label_sp, predict_sp)
        smooth_rmse, smooth_mae, tc_num_smooth, tc_num_label = smooth(path_predict_dic)

        # data = pd.DataFrame({'path': path_list, 'speed': label_sp, 'predict_speed': predict_sp})
        # data.to_csv('./result/track_curve/WCPr_lead' + str(lead_time) + '_' + str(index) + '.csv', index=False)
        data = pd.DataFrame({'path': path_list, 'speed': label_sp, 'predict_speed': predict_sp, 'translation speed':
            translational_speed_list, 'pressure': lon_list, 'structure': structure_list, 'speed1': speed1_list})
        data.to_csv('./result/re/WCPr_lead' + str(lead_time) + '_' + str(index) + '.csv', index=False)
        # data1 = pd.DataFrame({'path': path_list, 'speed': tc_num_label, 'predict_speed': tc_num_smooth})
        # data1.to_csv('./result/track_curve/WCPr_lead' + str(lead_time) + '_' + str(index) + '.csv', index=False)

        psnr2 /=len(test_loader)
        sam2 /= len(test_loader)
        ssim2 /= len(test_loader)
        rmse2 /= len(test_loader)

        psnr4 /= len(test_loader)
        sam4 /= len(test_loader)
        ssim4 /= len(test_loader)
        rmse4 /= len(test_loader)

        # psnr = (psnr2 + psnr4)/2
        # sam = (sam2 + sam4)/2
        # ssim = (ssim2 + ssim4)/2
        # rmse_fuse = (rmse2 + rmse4)/2

        f.write(str(index) + '\t' +
                str(rmse) + '\t' +
                str(smooth_rmse) + '\t' +
                str(lmae) + '\t' +
                str(smooth_mae) +'\t' +
                str(psnr4) + '\t' +
                str(sam4) + '\t' +
                str(ssim4) + '\t' +
                str(rmse4) + '\n')
        f.close()


if __name__ == '__main__':
    img = Image.open('./vis-ir/RoadScene/vis/2.jpg')
    h5 = h5py.File('./Training_dataset/vis_ir_dataset64.h5', 'r')
    im = h5['data'][:]
    img = np.array(img)
    # print(img)
    print(img.shape)
