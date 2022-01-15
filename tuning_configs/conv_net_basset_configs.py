from ray import tune

basset_original = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_dropout_5_weight_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.5}

basset_original_dropout_15_weight_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.01,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.5}

basset_original_dropout_13_weight_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.01,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_dropout_3_weight_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_dropout_1 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.1}

basset_original_dropout_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.2}

basset_original_dropout_4 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.4}

basset_original_dropout_5 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.5}

basset_original_dropout_6 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.6}

basset_original_dropout_7 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.1}

basset_original_dropout_8 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.8}


basset_original_long_large = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 200, 
                   "num_filters_c2" : 300,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_long = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_h1_200 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_h1_100 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 100, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_h1_150 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 150, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_h1_250 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 250, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_h1_300 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 300, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}





basset_original_c2_pool_4 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_pool_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 2,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_pool_1 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 1,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_pool_3 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 3,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_pool_5 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 5,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_8 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 8,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_10 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 10,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_12 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 12,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_c2_15 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 15,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}



basset_original_pool_size_1 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 1,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_pool_size_2 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 2,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_pool_size_5 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 5,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_pool_size_10 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 10,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_GPU = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 300, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_GPU_100 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 100, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_GPU_200 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 200, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_GPU_250 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 250, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_GPU_350 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 350, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_GPU_400 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 400, 
                   "num_filters_c2" : 200,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}


basset_original_12 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 12,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}



basset_original_15 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 15,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_17 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 17,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_19 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 19,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_21 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 21,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}

basset_original_23 = {"batch_size": 1024,
                   "learning_rate": 0.00001,
                   "weight_decay" : 0.0001,
                   "num_filters_c1" : 50, 
                   "num_filters_c2" : 50,
                   "conv_filter_size_c1" : 23,
                   "conv_filter_size_c2" : 11,
                   "conv_filter_stride" : 4,
                   "pooling_kernel_size_c1" : 3,
                   "pooling_kernel_size_c2" : 4,
                   "h1_size" : 200, 
                   "h2_size" : 200,
                   "dropout_p" : 0.3}





basset_Bayesian_TR1 = {"batch_size": 512,
                       "learning_rate": 0.00001,
                       "weight_decay" : tune.loguniform(0.00001, 0.001),
                       "num_filters_c1" : tune.qrandint(100, 300, 50), 
                       "num_filters_c2" : tune.qrandint(100, 300, 50),
                       "conv_filter_size_c1" : tune.randint(7, 19),
                       "conv_filter_size_c2" : tune.randint(7, 11),
                       "conv_filter_stride" : 4,
                       "pooling_kernel_size_c1" : 3,
                       "pooling_kernel_size_c2" : 4,
                       "h1_size" : tune.qrandint(50, 200, 50), 
                       "h2_size" : tune.qrandint(50, 200, 50),
                       "dropout_p" : 0.3}

#Change the number of threds for trails to be hardcoded in the py file, and a number that is multiplies as threads in snakefile
#Remove one hidden layer


    