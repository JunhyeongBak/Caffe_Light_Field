import argparse
from model import *
from function import *

caffe.set_mode_gpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Light field')
    parser.add_argument('--trainset_path', required=False, default='./datas/face_dataset/face_train_9x9', help='trainset path')
    parser.add_argument('--testset_path', required=False, default='./datas/face_dataset/face_train_9x9', help='testset path')
    parser.add_argument('--train_path', required=False, default='./scripts/denseUNet_train.prototxt', help='train path')
    parser.add_argument('--test_path', required=False, default='./scripts/denseUNet_test.prototxt', help='test path')
    parser.add_argument('--solver_path', required=False, default='./scripts/denseUNet_solver.prototxt', help='solver path')
    parser.add_argument('--model_path', required=False, default= './models/denseUNet', help='model path')
    parser.add_argument('--model_name', required=False, default= './models/denseUNet_iter_130000.caffemodel', help='model name')
    parser.add_argument('--train_size', required=False, default=460, help='train size')
    parser.add_argument('--test_size', required=False, default=460, help='test size')
    parser.add_argument('--n_sai', required=False, default=25, help='num of sai')
    parser.add_argument('--shift_val', required=False, default=0, help='shift value')
    parser.add_argument('--batch_size', required=False, default=4, help='batch size')
    parser.add_argument('--pick_mode', required=False, default='9x9', help='pick mode')
    parser.add_argument('--center_id', required=False, default=12, help='center id')
    parser.add_argument('--epoch', required=False, default=100000, help='epoch')
    parser.add_argument('--lr', required=False, default=0.0005, help='learning rate')
    parser.add_argument('--mode', required=False, default='run', help='mode')
    args = parser.parse_args()

    if args.mode == 'train': 
        # Generate prototxt
        train_proto_gen(args)
        test_proto_gen(args)
        solver_proto_gen(args)

        # Get solver model
        solver = caffe.get_solver(args.solver_path)
        #solver.net.copy_from(args.model_name)

        # Run solver model
        solver.solve()
    elif args.mode == 'test':
        # Generate prototxt
        test_proto_gen(args)

        # Get model
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)

        # Run model
        for i_tot in range(args.test_size):
            input_color = cv2.imread(args.testset_path+'/sai{}_40.png'.format(i_tot), cv2.IMREAD_COLOR)

            n.blobs['input_color'].data[...] = image_to_blob(input_color)
            n.blobs['input'].data[...] = image_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.

            n.forward()          
    elif args.mode == 'run':
        # Generate prototxt
        time_stamp = time.time()
        test_proto_gen(args)
        print('Generate prototxt time: {}'.format(time.time()-time_stamp))

        # Get model
        start = time.time()
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)
        print('Get model time: {}'.format(time.time()-start))

        # Run model
        start = time.time()
        input_color = cv2.imread('./infer_input.png', cv2.IMREAD_COLOR)
        input_color = cv2.resize(input_color, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        n.blobs['input_color'].data[...] = image_to_blob(input_color)
        n.blobs['input'].data[...] = image_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.

        n.forward()
        print('Run model time: {}'.format(time.time()-start))

        # Print LF result
        start = time.time()
        predict_blob = predict_to_blob(n.blobs['predict'].data[...])

        predict_blob = view_center_change_5x5(predict_blob, 2.5)
        predict_lf = blob_to_lf(predict_blob)
        cv2.imwrite('./infer_output.png', predict_lf)
        print('Print LF result time: {}'.format(time.time()-start))

        # Print EPI result
        start = time.time()
        epi_hor, epi_ver = epi_slicing(predict_blob, 2, 2, 256//2, 256//2)
        epi_hor = cv2.resize(blob_to_image(epi_hor), dsize=(256, 50), interpolation=cv2.INTER_AREA)
        epi_ver = cv2.resize(blob_to_image(epi_ver), dsize=(50, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite('./epi_hor.png', epi_hor)
        cv2.imwrite('./epi_ver.png', epi_ver)
        print('Print EPI result time: {}'.format(time.time()-start))

        # Print EPI con result
        '''
        epi_hor_con = None
        epi_ver_con = None

        for i in range(5):
            epi_hor, _ = epi_slicing(predict_blob, i, 2, 256//2, 256//2)
            _, epi_ver = epi_slicing(predict_blob, 2, i, 256//2, 256//2)

            if i == 0:
                epi_hor_con = epi_hor
                epi_ver_con = epi_ver
            else:
                epi_hor_con = np.concatenate((epi_hor_con, epi_hor), axis=2)
                epi_ver_con = np.concatenate((epi_ver_con, epi_ver), axis=3)

        print('epi_hor_con:', epi_hor_con.shape)
        print('epi_ver_con:', epi_ver_con.shape)
        cv2.imwrite('./epi_hor_con.png', np.uint8(blob_to_image(epi_hor_con)))
        cv2.imwrite('./epi_ver_con.png', np.uint8(blob_to_image(epi_ver_con)))
        '''
    else:
        pass