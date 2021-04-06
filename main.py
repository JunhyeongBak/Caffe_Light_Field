import argparse
from model import *
from function import *
import math
import skimage

caffe.set_mode_gpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Light field')
    parser.add_argument('--trainset_path', required=False, default='./datas/face_dataset/face_train_5x5_f50_b2.5', help='trainset path')
    parser.add_argument('--testset_path', required=False, default='./datas/face_dataset/face_test_5x5_f50_b2.5', help='testset path')
    parser.add_argument('--train_path', required=False, default='./scripts/denseUNet_train.prototxt', help='train path')
    parser.add_argument('--test_path', required=False, default='./scripts/denseUNet_test.prototxt', help='test path')
    parser.add_argument('--solver_path', required=False, default='./scripts/denseUNet_solver.prototxt', help='solver path')
    parser.add_argument('--model_path', required=False, default= './models/denseUNet', help='model path')
    parser.add_argument('--model_name', required=False, default= './models/denseUNet_iter_130000.caffemodel', help='model name')
    parser.add_argument('--train_size', required=False, default=600, help='train size')
    parser.add_argument('--test_size', required=False, default=98, help='test size')
    parser.add_argument('--n_sai', required=False, default=25, help='num of sai')
    parser.add_argument('--shift_val', required=False, default=0, help='shift value')
    parser.add_argument('--batch_size', required=False, default=4, help='batch size')
    parser.add_argument('--pick_mode', required=False, default='9x9', help='pick mode')
    parser.add_argument('--center_id', required=False, default=12, help='center id')
    parser.add_argument('--epoch', required=False, default=100000, help='epoch')
    parser.add_argument('--lr', required=False, default=0.0005, help='learning rate')
    parser.add_argument('--mode', required=False, default='test', help='mode')
    args = parser.parse_args()

    if args.mode == 'step':
        # Generate prototxt
        train_proto_gen(args)
        #test_proto_gen(args)
        solver_proto_gen(args)
    
        # Get solver model
        solver = caffe.get_solver(args.solver_path)
        solver.net.copy_from(args.model_name)

        # Run solver model
        solver.step(1)

    elif args.mode == 'train': 
        # Generate prototxt
        train_proto_gen(args)
        test_proto_gen(args)
        solver_proto_gen(args)

        # Get solver model
        solver = caffe.get_solver(args.solver_path)
        solver.net.copy_from(args.model_name)

        # Run solver model
        solver.solve()
    elif args.mode == 'test':
        # Generate prototxt
        test_proto_gen(args)

        # Get model
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)

        time_mean = 0
        val_psnr_mean = 0
        val_ssim_mean = 0
        for i_tot in range(args.test_size):
            # Run model
            start = time.time()
            input_color = cv2.imread(args.testset_path+'/sai{}_40.png'.format(i_tot), cv2.IMREAD_COLOR)
            input_color = cv2.resize(input_color, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            n.blobs['input_color'].data[...] = image_to_blob(input_color)
            n.blobs['input'].data[...] = image_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.
            n.forward()

            # Print predicted LF
            predict_blob = predict_to_blob(n.blobs['predict'].data[...])
            predict_blob = view_center_change_5x5(predict_blob, 2.5)
            predict_lf = blob_to_lf(predict_blob)
            cv2.imwrite('./results/pr{}_lf.png'.format(i_tot), predict_lf)
            finish = time.time()

            # Validation
            gt_blob = lf_load_5x5(args.testset_path+'/sai'+str(i_tot)+'_{}.png', is_color=True)
            gt_blob = view_center_change_5x5(gt_blob, 2.5)

            predict_tensor = np.uint8(blob_to_tensor(predict_blob))
            gt_tensor = np.uint8(blob_to_tensor(gt_blob))

            val_psnr = 0
            for i in range(25):
                predict_tensor_slice = cv2.cvtColor(predict_tensor[i, :, :, :], cv2.COLOR_BGR2GRAY)
                gt_tensor_slice = cv2.cvtColor(gt_tensor[i, :, :, :], cv2.COLOR_BGR2GRAY)
                val_psnr = val_psnr + cv2.PSNR(predict_tensor_slice, gt_tensor_slice)
            val_psnr = val_psnr / 25.

            val_ssim = 0
            for i in range(25):
                predict_tensor_slice = cv2.cvtColor(predict_tensor[i, :, :, :], cv2.COLOR_BGR2GRAY)
                gt_tensor_slice = cv2.cvtColor(gt_tensor[i, :, :, :], cv2.COLOR_BGR2GRAY)
                val_ssim = val_ssim + skimage.metrics.structural_similarity(predict_tensor_slice, gt_tensor_slice, data_range=255)
            val_ssim = val_ssim / 25.

            time_mean = time_mean + (finish-start)
            val_psnr_mean = val_psnr_mean + val_psnr
            val_ssim_mean = val_ssim_mean + val_ssim
            print('Image: {}, Time: {}, PSNR: {}, SSIM: {}'.format(i_tot, finish-start, val_psnr, val_ssim))
        
        time_mean = time_mean / args.test_size
        val_psnr_mean = val_psnr_mean / args.test_size
        val_ssim_mean = val_ssim_mean / args.test_size
        print('=== Total mean === Time: {}, PSNR: {}, SSIM: {}'.format(time_mean, val_psnr_mean, val_ssim_mean))

    elif args.mode == 'run':
        # Generate prototxt
        test_proto_gen(args)

        # Get model
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)

        # Run model
        start = time.time()
        input_color = cv2.imread('./input.png', cv2.IMREAD_COLOR)
        input_color = cv2.resize(input_color, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        n.blobs['input_color'].data[...] = image_to_blob(input_color)
        n.blobs['input'].data[...] = image_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.
        n.forward()

        # Print predicted LF
        predict_blob = predict_to_blob(n.blobs['predict'].data[...])
        predict_blob = view_center_change_5x5(predict_blob, 2.5)
        predict_lf = blob_to_lf(predict_blob)
        cv2.imwrite('./results/pr_lf.png', predict_lf)
        print('LF predict time: {}'.format(time.time()-start))

        print_net_parameters(args.test_path)
    elif args.mode == 'anal':
        only_pr = False
        i_tot = 2

        ### Anal predict ###
        # Generate prototxt
        test_proto_gen(args)

        # Get model
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)

        # Run model
        input_color = cv2.imread(args.testset_path+'/sai{}_40.png'.format(i_tot), cv2.IMREAD_COLOR)
        input_color = cv2.resize(input_color, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        n.blobs['input_color'].data[...] = image_to_blob(input_color)
        n.blobs['input'].data[...] = image_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.
        n.forward()

        # Print predicted LF
        predict_blob = predict_to_blob(n.blobs['predict'].data[...])
        predict_blob = view_center_change_5x5(predict_blob, 2.5)
        predict_lf = blob_to_lf(predict_blob)
        cv2.imwrite('./results/pr_lf.png', predict_lf)

        # Print predicted EPI
        epi_hor, epi_ver = epi_slicing(predict_blob, 2, 2, 256//2, 256//2)
        epi_hor = cv2.resize(blob_to_image(epi_hor), dsize=(256, 50), interpolation=cv2.INTER_AREA)
        epi_ver = cv2.resize(blob_to_image(epi_ver), dsize=(50, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite('./results/pr_epi_hor.png', epi_hor)
        cv2.imwrite('./results/pr_epi_ver.png', epi_ver)

        # Print predicted grid
        predict_grid = blob_to_grid(predict_blob)
        cv2.imwrite('./results/pr_grid.png', predict_grid)

        # Print predicted flow
        flow_v_blob = n.blobs['flow_v'].data[...]
        flow_h_blob = n.blobs['flow_h'].data[...]
        flow = np.zeros((256, 256, 2))
        for i in range(25):
            flow_v_blob_slice = flow_v_blob[0, i, :, :]
            flow_h_blob_slice = flow_h_blob[0, i, :, :]
            flow[:, :, 0] = (flow_v_blob_slice-(np.mean(flow_v_blob_slice)/2))*2
            flow[:, :, 1] = (flow_h_blob_slice-(np.mean(flow_h_blob_slice)/2))*2

            flow_color = flow_to_color(flow, convert_to_bgr=False)
            cv2.imwrite('./results/pr_flow{}.png'.format(index_picker_5x5(i)), flow_color)

        if only_pr == True:
            exit()

        ### Anal GT ###
        # Print GT LF
        gt_blob = lf_load_5x5(args.testset_path+'/sai'+str(i_tot)+'_{}.png', is_color=True)
        gt_blob = view_center_change_5x5(gt_blob, 2.5)
        gt_lf = blob_to_lf(gt_blob)
        cv2.imwrite('./results/gt_lf.png', gt_lf)

        # Print GT EPI
        epi_hor, epi_ver = epi_slicing(gt_blob, 2, 2, 256//2, 256//2)
        epi_hor = cv2.resize(blob_to_image(epi_hor), dsize=(256, 50), interpolation=cv2.INTER_AREA)
        epi_ver = cv2.resize(blob_to_image(epi_ver), dsize=(50, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite('./results/gt_epi_hor.png', epi_hor)
        cv2.imwrite('./results/gt_epi_ver.png', epi_ver)

        # Print GT grid
        gt_grid = blob_to_grid(gt_blob)
        cv2.imwrite('./results/gt_grid.png', gt_grid)

    else:
        pass