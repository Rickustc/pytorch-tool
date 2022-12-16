""


def run_check_dataset(json_path):
     
 
    ##### 划分fold 
	# pdb.set_trace()
	# fold=0
	# train_df, valid_df = make_fold(fold)


    ##### 构造dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    dataset = DatasetCCsagpi(opt)
    tensor_list = ['L', 'H', 'H_path', 'mask']
    for i in range(1,5):
		# pdb.set_trace() 
        r = dataset[i]
        print(r['index'], 'id = ', r['id'],'-----------')
        for k in tensor_list:
            v = r[k]
            print(k)
            print('\t',v.shape, v.is_contiguous(), v.min(), v.max())
            print('\t',v.reshape(-1)[:8], '...')
            print('\t',v.reshape(-1)[-8:])
        print('')
        
    loader = DataLoader(
		dataset,
		sampler = SequentialSampler(dataset),
		batch_size  = 8,
		drop_last   = True, #
		num_workers = 0,   # if debug  num_workers= 0
		pin_memory  = False,
		# worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
		# collate_fn = null_collate,     #    更多collate_fn的使用，参考多分辨率DETR的代码
	)
    print(loader.batch_size,len(loader),len(dataset))
    print('')
    for t, batch in enumerate(loader):
        if t>5: break
        print('batch ', t,'===================')
        print('index', batch['index'])
        for k in tensor_list:
            v = batch[k]
            print(k)
            print('\t',v.shape, v.is_contiguous())
            print('\t',v.reshape(-1)[:8])
        print('')
        
if __name__ == '__main__':
    json = "/home/jupyter/share/SotaTransformerModel/SwinMR/options/SwinMR/example/train_swinmr_CCnpi_G1D30.json"
    run_check_dataset(json)
    
    
    #dasdasdasasfsafd