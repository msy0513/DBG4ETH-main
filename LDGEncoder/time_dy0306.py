from block0306 import *
from encoders1 import *
from data0306 import *
import torch
from torch.utils.data import DataLoader

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--max-links', dest='max_links', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-node', dest='num_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--method', dest='method',
                        help='Method.')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='bmname',
                        bmname='att_Ethereum_dynamic_1k_64',
                        num_nodes=1000,
                        max_links=2000,
                        feature_type='default',
                        lr=0.01,
                        clip=2.0,
                        batch_size=10,#10
                        num_epochs=30,# 50
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,# 2
                        num_gc_layers=3,
                        dropout=0.0,
                        method='diffpool',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()

args = arg_parse()
# export scalar data to JSON for external processing
path = os.path.join(args.logdir, gen_prefix(args))
writer = SummaryWriter(path)
writer.close()

# Print CUDA information
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# data_path = "/datasets/TEGs_mining(2k).npz"
# data_path = "/datasets/TEGs_fishing(2k).npz"
# data_path = "/TEGs_phishing_new(2k).npz"
# data_path = "/TEGs_exchange_new(2k).npz"
data_path = "/TEGs_mining_new(2k).npz"


result=con_dynamic_Gset(data_path,args.batch_size, args.max_links, max_n=args.max_links)
train_dataset, val_dataset, test_dataset = result['train_Graph'], result['val_Graph'], result['test_Graph']

# args.batch_size

# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=None )
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=None)
# test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=None)

# max_num_nodes = train_dataset['Adj'][0][0].shape[0]
# print(type(max_num_nodes))
# print(f"max_num_nodes:{max_num_nodes}")
max_num_nodes=2000
# print(train_dataloader)
# feature dimension
input_dim = 15# 2
assign_input_dim = 15# 2


train_num=7
val_num=3
test_num=1


model = diffpool(
    max_num_nodes,
    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, pool_method = 'diffpool',
    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
    assign_input_dim=assign_input_dim).cuda()

from torchinfo import summary

summary(model)

_, val_accs = train_phishing_detector_dy(train_dataset, model, train_num, val_num, test_num, args,
                    val_dataset=val_dataset, test_dataset=test_dataset, writer=writer)


#TEGDetect
# _, val_accs = train_phishing_detector_dy(train_dataset, model, train_num, val_num, test_num, args,
#                     val_dataset=val_dataset,test_dataset=test_dataset, writer=writer)
# _, val_accs = train_phishing_detector_dy(train_dataset, model,train_num, val_num, test_G_num, args, 
#                     val_dataset=val_dataset, test_dataset=test_G, writer=writer)



# test performance
# mth = os.path.join('model_param/TEGD.pth')
# checkpoint = torch.load(mth)
# model.load_state_dict(checkpoint['net'])
# evaluate_dynamic(test_dataset, test_num, model, args, name='Test')
