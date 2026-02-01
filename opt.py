import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--window_size',
        type=int,
        default=280)
    parser.add_argument(
        '--kernel_list',
        type=list,
        default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251])
    parser.add_argument(
        '--stride_factor',
        type=int,
        default=50)
    parser.add_argument(
        '--d_model',
        type=int,
        default=512)
    parser.add_argument(
        '--vis_dropout',
        type=float,
        default=0.3)
    parser.add_argument(
        '--train_sample',
        type=int,
        default=20)
    parser.add_argument(
        '--pos_thresh',
        type=float,
        default=0.7)
    parser.add_argument(
        '--neg_thresh',
        type=float,
        default=0.3)
    parser.add_argument(
        '--cls_weight',
        type=float,
        default=1.0)
    parser.add_argument(
        '--reg_weight',
        type=float,
        default=10.0)
    parser.add_argument(
        '--label_weight',
        type=float,
        default=0.25)
    parser.add_argument(
        '--n_classes',
        type=int,
        default=18
    )

    parser.add_argument(
        '--max_prop',
        type=int,
        default=200
    )
    parser.add_argument(
        '--min_prop',
        type=int,
        default=100
    )
    parser.add_argument(
        '--min_prop_before_nms',
        type=int,
        default=200
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=100
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.95)
    parser.add_argument(
        '--beta',
        type=float,
        default=0.999)
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-8)

    parser.add_argument(
        '--patience_epoch',
        type=int,
        default=1)
    parser.add_argument(
        '--reduce_factor',
        type=float,
        default=0.75)
    parser.add_argument(
        '--grad_norm',
        type=int,
        default=1)
    parser.add_argument(
        '--save_every_epoch',
        type=int,
        default=1)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )
    #file_configure
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default='./checkpoint/model_epoch_49.t7')
    parser.add_argument(
        '--frames_path',
        type=str,
        default='/home/dataset/dota/frames')
    parser.add_argument(
        '--meta_data',
        type=str,
        default='./dataset/metadata_all.json')
    parser.add_argument(
        '--train_X',
        type=str,
        default='./dataset/train.txt')
    parser.add_argument(
        '--test_X',
        type=str,
        default='./dataset/test.txt')
    parser.add_argument(
        '--logs_path',
        type=str,
        default='./runs')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--results_path',
        type=str,
        default='./results')

    # ******** Inference **********
    parser.add_argument(
        '-v', '--video', default='trimmed.mp4',
        type=str,
        help='video file name')
    parser.add_argument(
        '-o', '--output', default='./out_frames',
        type=str,
        help='video file name')
    

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
                        
    args = parser.parse_args()

    return args
