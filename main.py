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
    parser.add_argument('--model_name', required=False, default= './models/denseUNet_iter_50000.caffemodel', help='model name')
    parser.add_argument('--result_path', required=False, default='./output', help='result path')
    parser.add_argument('--train_size', required=False, default=190, help='train size')
    parser.add_argument('--test_size', required=False, default=190, help='test size')
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
        train_proto_gen(args)
        test_proto_gen(args)
        solver_proto_gen(args)

        solver = caffe.get_solver(args.solver_path)
        solver.net.copy_from(args.model_name)
        solver.solve()
    elif args.mode == 'test':
        test_proto_gen(args)

        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)

        for i_tot in range(args.test_size):
            input_color = cv2.imread(args.testset_path+'/sai{}_40.png'.format(i_tot), cv2.IMREAD_COLOR)

            n.blobs['input_color'].data[...] = img_to_blob(input_color)
            n.blobs['input'].data[...] = img_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.

            n.forward()          
    elif args.mode == 'run':
        time_stamp = time.time()
        test_proto_gen(args)
        print('Model proto gen time: {}', time.time()-time_stamp)

        time_stamp = time.time()
        n = caffe.Net(args.test_path, args.model_name, caffe.TEST)
        print('Model gen time: {}', time.time()-time_stamp)

        time_stamp = time.time()
        input_color = cv2.imread('./infer_input.png', cv2.IMREAD_COLOR)
        input_color = cv2.resize(input_color, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        
        n.blobs['input_color'].data[...] = img_to_blob(input_color)
        n.blobs['input'].data[...] = img_to_blob(cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY))/256.

        n.forward()

        output_spatial = blob_to_spatial(n.blobs['predict'].data[...], args.n_sai)
        cv2.imwrite('./infer_output.png', output_spatial)
        print('Model run time: {}', time.time()-time_stamp)

        output_angular = blob_to_angular(n.blobs['predict'].data[...], args.n_sai)
        cv2.imwrite('./infer_output.png', output_angular)
    else:
        pass