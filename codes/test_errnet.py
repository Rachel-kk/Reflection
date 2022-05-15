from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import utils.util as util


opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

datadir = '/home/ivdai/Annotations/KX/data/Reflection/testsets'
# datadir = '/home/rackel_kk/Annotation/KX/data/Reflection/testsets'
# Define evaluation/test dataset

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_sir2 = datasets.CEILTestDataset(join(datadir, 'SIR2/WildSceneDataset/withgt'))

# eval_dataset_real = datasets.CEILTestDataset(
#     join(datadir, 'real20'),
#     fns=read_fns('../real_test.txt'),
#     size=20)
eval_dataset_real90 = datasets.CEILTestDataset(join(datadir, 'real_train'))

eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'SIR2/PostcardDataset'))
eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'SIR2/SolidObjectDataset'))

# test_dataset_internet = datasets.RealDataset(join(datadir, 'internet'))
# test_dataset_unaligned300 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned300/blended'))
# test_dataset_unaligned150 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned150/blended'))
# test_dataset_unaligned_dynamic = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned_dynamic/blended'))
# test_dataset_sir2 = datasets.RealDataset(join(datadir, 'sir2_wogt/blended'))


eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real90, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_sir2 = datasets.DataLoader(
    eval_dataset_sir2, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_internet = datasets.DataLoader(
#     test_dataset_internet, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_sir2 = datasets.DataLoader(
#     test_dataset_sir2, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned300 = datasets.DataLoader(
#     test_dataset_unaligned300, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned150 = datasets.DataLoader(
#     test_dataset_unaligned150, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned_dynamic = datasets.DataLoader(
#     test_dataset_unaligned_dynamic, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)

"""Main Loop"""
result_dir = './results'

# evaluate on synthetic test data from CEILNet
# res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2', savedir=join(result_dir, 'CEILNet_table2'))

# evaluate on four real-world benchmarks
# res = engine.eval(eval_dataloader_real, dataset_name='testdata_real')

res = engine.eval(eval_dataloader_real, dataset_name='testdata_real', savedir=join(result_dir, 'real90'))
res = engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=join(result_dir, 'solidobject'))
res = engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=join(result_dir, 'postcard'))
res = engine.eval(eval_dataloader_sir2, dataset_name='testdata_sir2', savedir=join(result_dir, 'sir2_withgt'))


# test on our collected unaligned data or internet images
# res = engine.test(test_dataloader_internet, savedir=join(result_dir, 'internet'))
# res = engine.test(test_dataloader_unaligned300, savedir=join(result_dir, 'unaligned300'))
# res = engine.test(test_dataloader_unaligned150, savedir=join(result_dir, 'unaligned150'))
# res = engine.test(test_dataloader_unaligned_dynamic, savedir=join(result_dir, 'unaligned_dynamic'))
# res = engine.test(test_dataloader_sir2, savedir=join(result_dir, 'sir2_wogt'))