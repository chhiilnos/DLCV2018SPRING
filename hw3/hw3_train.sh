mkdir weight
wget "https://www.dropbox.com/s/9urkq5gbt1wfd7x/vgg16_weights_tf_dim_ordering_tf_kernels.h5?dl=0" -O weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5
bash train.sh FCN_Vgg16_32s
