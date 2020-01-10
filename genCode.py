import os
import torch.nn as nn
import torch.nn.modules.pooling as pooling
import torch.nn.modules.conv as conv
import torch.nn.modules.linear as linear
import numpy as np

# --------------------------------------------------------------------
# this code should generate a C code programme based on CMSIS-NN library
# the only restriction is to go through the correct quantize program before
# to use this code in your prefered IDE you have to include:
#                   CMSIS-DSP
#                   CMSIS-NN
# prior to build
#
# the supported layer are:
#           Conv2d
#           ReLu
#           Linear
#           AvgPool2d
#           MaxPool2d
#
# the kernell, size or shape can be non-square
# if so, inlcude the local_NN folder containing the non-square pooling functions
#
# the generated code doesn't do any interpretation of the ouput.
# --------------------------------------------------------------------


def __convert_to_x4_weights(weights):
    """This function convert the fully-connected layer weights
       to the format that accepted by X4 implementation"""
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows/4)):
        # we only need to do the re-ordering for every 4 rows
        row_base = 4*i
        for j in range(int(num_of_cols/4)):
            # for each 4 entries
            column_base = 4*j
            new_weights[counter] = weights[row_base][column_base]
            new_weights[counter+1] = weights[row_base+1][column_base]
            new_weights[counter+2] = weights[row_base][column_base+2]
            new_weights[counter+3] = weights[row_base+1][column_base+2]
            new_weights[counter+4] = weights[row_base+2][column_base]
            new_weights[counter+5] = weights[row_base+3][column_base]
            new_weights[counter+6] = weights[row_base+2][column_base+2]
            new_weights[counter+7] = weights[row_base+3][column_base+2]

            new_weights[counter+8] = weights[row_base][column_base+1]
            new_weights[counter+9] = weights[row_base+1][column_base+1]
            new_weights[counter+10] = weights[row_base][column_base+3]
            new_weights[counter+11] = weights[row_base+1][column_base+3]
            new_weights[counter+12] = weights[row_base+2][column_base+1]
            new_weights[counter+13] = weights[row_base+3][column_base+1]
            new_weights[counter+14] = weights[row_base+2][column_base+3]
            new_weights[counter+15] = weights[row_base+3][column_base+3]
            counter = counter + 16
        # the remaining ones are in order
        for j in range(int(num_of_cols-num_of_cols % 4), int(num_of_cols)):
            new_weights[counter] = weights[row_base][j]
            new_weights[counter+1] = weights[row_base+1][j]
            new_weights[counter+2] = weights[row_base+2][j]
            new_weights[counter+3] = weights[row_base+3][j]
            counter = counter + 4
    return new_weights


# this is the file you should include in your C project
def __generate_header(net, file_name):
    print('Generating file: ' + file_name)
    f = open(file_name, 'w')
    f.write('#ifndef __N%s__\n' % (net.name()))
    f.write('#define __N%s__\n\n' % (net.name()))

    # include all the necessary files
    f.write('#include "arm_math.h"\n')
    f.write('#include "arm_nnfunctions.h"\n\n')
    f.write('#include "n%s_parameters.h"\n' % (net.name()))
    f.write('#include "n%s_weights.h"\n' % (net.name()))
    f.write('#include "local_NN.h"\n\n')

    # prototype of the network function
    # use this function to run an inference on your network
    f.write('void n%s_run(q7_t* input_data, q7_t* output_data);\n\n' % (net.name()))

    f.write('#endif /* __N%s__ */\n' % (net.name()))
    f.close()


# this file contain all the weight and bias in fixed point for your network
def __generate_weights(net, file_name):
    print('Generating file: ' + file_name)
    f = open(file_name, 'w')

    # to detect if the layer before a linear is a convolution
    prev_is_conv = False

    # iterate through every layer that has weight
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight'):
            if isinstance(getattr(net, named_children[0]), conv.Conv2d):
                prev_is_conv = True

                # transpose weight matrix to HWC format (PyTorch is CHW)
                reordered_wts = getattr(net, named_children[0]).quant_weight.permute(0, 2, 3, 1).data.numpy()
            elif isinstance(getattr(net, named_children[0]), linear.Linear):
                if prev_is_conv:
                    # if the last layer was a convolution, we know the shape of the input data
                    # this allow us to optimize the weight matrix for an optimize function of CMSIS-NN
                    [B, C, H, W] = getattr(net, named_children[0]).input_shape
                    [O] = getattr(net, named_children[0]).output_shape
                    reordered_wts = getattr(net, named_children[0]).quant_weight.data.numpy().reshape(O, C, H, W).transpose(0, 2, 3, 1)
                    reordered_wts = __convert_to_x4_weights(reordered_wts)
                else:
                    reordered_wts = getattr(net, named_children[0]).quant_weight.data.numpy()
                prev_is_conv = False

            # once the weight matrix is in the good format, we can flatten it and write it
            f.write('#define %s_%s_WT {' % (net.name(), named_children[0].upper()))
            reordered_wts.tofile(f, sep=",", format="%d")
            f.write('}\n')

            # we can write the bias matrix to (no need to flatten it as the bias are in one dimension
            f.write('#define %s_%s_BIAS {' % (net.name(), named_children[0].upper()))
            getattr(net, named_children[0]).quant_bias.data.numpy().tofile(f, sep=",", format="%d")
            f.write('}\n\n')
    f.close()


# this file contain all the constant use by your network in CMSIS-NN
def __generate_parameters(net, file_name):
    print('Generating file: ' + file_name)
    f = open(file_name, 'w')

    # constant to use to create the net input and output variables in your C project
    # Input buffer
    f.write("#define %s_INPUT_SIZE %d\n" % (net.name(), net.input_nb_element))
    # Output buffer
    f.write("#define %s_OUTPUT_SIZE %d\n\n" % (net.name(), net.output_nb_element))

    # iterate through every layer
    # all constant to used by CMSIS-NN in your C project
    for named_children in net.named_children():
        if isinstance(getattr(net, named_children[0]), linear.Linear):
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_DIM " + str(getattr(net, named_children[0]).in_features) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_DIM " + str(getattr(net, named_children[0]).out_features) + "\n\n")
        elif isinstance(getattr(net, named_children[0]), nn.modules.activation.ReLU):
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_SIZE " + str(getattr(net, named_children[0]).output_nb_element) + "\n\n")
        elif isinstance(getattr(net, named_children[0]), conv.Conv2d):
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_DIM_X " + str(getattr(net, named_children[0]).input_shape[-2]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_DIM_Y " + str(getattr(net, named_children[0]).input_shape[-1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_CH " + str(getattr(net, named_children[0]).in_channels) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_KER_DIM_X " + str(getattr(net, named_children[0]).kernel_size[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_KER_DIM_Y " + str(getattr(net, named_children[0]).kernel_size[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_PAD_X " + str(getattr(net, named_children[0]).padding[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_PAD_Y " + str(getattr(net, named_children[0]).padding[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_STRIDE_X " + str(getattr(net, named_children[0]).stride[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_STRIDE_Y " + str(getattr(net, named_children[0]).stride[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_CH " + str(getattr(net, named_children[0]).out_channels) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_DIM_X " + str(getattr(net, named_children[0]).output_shape[-2]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_DIM_Y " + str(getattr(net, named_children[0]).output_shape[-1]) + "\n\n")
        elif isinstance(getattr(net, named_children[0]), (pooling.MaxPool2d, pooling.AvgPool2d)):
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_DIM_X " + str(getattr(net, named_children[0]).input_shape[-2]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_DIM_Y " + str(getattr(net, named_children[0]).input_shape[-1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_IN_CH " + str(getattr(net, named_children[0]).input_shape[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_KER_DIM_X " + str(getattr(net, named_children[0]).kernel_size[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_KER_DIM_Y " + str(getattr(net, named_children[0]).kernel_size[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_PAD_X " + str(getattr(net, named_children[0]).padding[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_PAD_Y " + str(getattr(net, named_children[0]).padding[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_STRIDE_X " + str(getattr(net, named_children[0]).stride[0]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_STRIDE_Y " + str(getattr(net, named_children[0]).stride[1]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_DIM_X " + str(getattr(net, named_children[0]).output_shape[-2]) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_DIM_Y " + str(getattr(net, named_children[0]).output_shape[-1]) + "\n\n")

    f.write("#define " + net.name() + "_DATA_RSHIFT " + str(net.in_rshift) + "\n")
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight' or 'bias'):
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_BIAS_LSHIFT " + str(getattr(net, named_children[0]).bias_lshift) + "\n")
            f.write("#define " + net.name() + "_" + named_children[0].upper() + "_OUT_RSHIFT " + str(getattr(net, named_children[0]).act_rshift) + "\n")

    # use this constant to scale de value back to its floating point form
    # useful for regression network but not for classification
    f.write("#define " + net.name() + "_OUT_RSHIFT " + str(net.out_rshift) + "\n")
    f.close()


# this file contain the code of the network using CMSIS-NN functions
def __generate_network(net, file_name):
    print('Generating file: ' + file_name)
    f = open(file_name, 'w')
    f.write('#include "n%s.h"\n\n' % net.name().upper())

    # create tabs pointing on the weight and bias constant
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight'):
            f.write("static q7_t const " + net.name() + "_" + named_children[0] + "_wt["
                    + str(getattr(net, named_children[0]).quant_weight.data.numpy().size) + "] = "
                    + net.name() + "_" + named_children[0].upper() + "_WT;\n")
        if hasattr(getattr(net, named_children[0]), 'bias'):
            f.write("static q7_t const " + net.name() + "_" + named_children[0] + "_bias["
                    + str(getattr(net, named_children[0]).quant_bias.data.numpy().size) + "] = "
                    + net.name() + "_" + named_children[0].upper() + "_BIAS;\n\n")

    # example on whate to declare to use the network
    # Input buffer
    f.write("//Add input_data and output_data in top main.c file\n")
    f.write("//q7_t input_data[%s_INPUT_SIZE];\n" % net.name())
    # Output buffer
    f.write("//q7_t output_data[%s_OUTPUT_SIZE];\n" % net.name())
    # Use network
    f.write('//n%s_run(input_data, output_data);\n\n' % net.name())

    # defining the size of the differents buffers used by CMSIS-NN to store the activations data
    max_col_buffer = 0
    max_scratch_buffer = 0
    max_layer_size = 0
    for named_children in net.named_children():
        if isinstance(getattr(net, named_children[0]), conv.Conv2d):
            im2col_buffer_size = 2 * 2 * getattr(net, named_children[0]).out_channels * \
                                 getattr(net, named_children[0]).kernel_size[0] * \
                                 getattr(net, named_children[0]).kernel_size[1]
            max_col_buffer = max(max_col_buffer, im2col_buffer_size)
        if isinstance(getattr(net, named_children[0]), linear.Linear):
            fc_buffer_size = 2 * getattr(net, named_children[0]).in_features
            max_col_buffer = max(max_col_buffer, fc_buffer_size)

        if isinstance(getattr(net, named_children[0]), (pooling.MaxPool2d, pooling.AvgPool2d, conv.Conv2d, linear.Linear)):
            if max_scratch_buffer < getattr(net, named_children[0]).output_nb_element + getattr(net, named_children[0]).input_nb_element:
                max_scratch_buffer = getattr(net, named_children[0]).output_nb_element + getattr(net, named_children[0]).input_nb_element

        if max_layer_size < getattr(net, named_children[0]).output_nb_element:
            max_layer_size = getattr(net, named_children[0]).output_nb_element

    # declaring those buffer
    f.write('static q7_t col_buffer[%d];\n' % max_col_buffer)
    f.write('static q7_t scratch_buffer[%d];\n\n' % max_scratch_buffer)

    f.write('void n%s_run(q7_t* input_data, q7_t* output_data)\n' % net.name())
    f.write('{\n')
    f.write('   q7_t* buffer1 = scratch_buffer;\n')
    f.write('   q7_t* buffer2 = buffer1 + %d;\n\n' % max_layer_size)

    input_buffer = 'input_data'
    output_buffer = 'buffer1'

    # to count the number of layer in our network
    nb_layer = 0

    # to detect if the layer before a linear is a convolution
    prev_is_conv = False

    # iterate through every layer
    for named_children in net.named_children():
        # if this is the first layer, the input of the network becomes the input of the layer
        if nb_layer == 0:
            input_buffer = 'input_data'
            output_buffer = 'buffer1'

        nb_layer += 1

        # if this is the last layer, the output of the layer becomes the output of the network
        if nb_layer == len(list(net.named_children())):
            output_buffer = 'output_data'

        # wrinting all the layer of the network
        if isinstance(getattr(net, named_children[0]), linear.Linear):
            if prev_is_conv:
                # if this is the first linear layer after a convolution layer, we can use the optimized function
                prev_is_conv = False
                conv_func = 'arm_fully_connected_q7_opt'
            else:
                conv_func = 'arm_fully_connected_q7'
            f.write('   ' + conv_func + '(' + input_buffer + ', '
                    + net.name() + "_" + named_children[0] + '_wt, '
                    + net.name() + "_" + named_children[0].upper() + '_IN_DIM, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_DIM, '
                    + net.name() + "_" + named_children[0].upper() + '_BIAS_LSHIFT, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_RSHIFT, '
                    + net.name() + "_" + named_children[0] + '_bias, '
                    + output_buffer + ', (q15_t*)col_buffer);\n')
        elif isinstance(getattr(net, named_children[0]), conv.Conv2d):
            prev_is_conv = True
            if getattr(net, named_children[0]).in_channels % 4 == 0 and getattr(net, named_children[0]).out_channels % 2 == 0:
                # if the layer has a multiple of 4 channel, we can use the optimized function
                conv_func = 'arm_convolve_HWC_q7_fast_nonsquare'
            else:
                conv_func = 'arm_convolve_HWC_q7_basic_nonsquare'
            f.write('   ' + conv_func + '(' + input_buffer + ', '
                    + net.name() + "_" + named_children[0].upper() + '_IN_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_IN_DIM_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_IN_CH, '
                    + net.name() + "_" + named_children[0] + '_wt, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_CH, '
                    + net.name() + "_" + named_children[0].upper() + '_KER_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_KER_DIM_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_PAD_X, '
                    + net.name() + "_" + named_children[0].upper() + '_PAD_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_STRIDE_X, '
                    + net.name() + "_" + named_children[0].upper() + '_STRIDE_Y, '
                    + net.name() + "_" + named_children[0] + '_bias, '
                    + net.name() + "_" + named_children[0].upper() + '_BIAS_LSHIFT, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_RSHIFT, '
                    + output_buffer + ', '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_DIM_Y, (q15_t*)col_buffer, NULL);\n')
        elif isinstance(getattr(net, named_children[0]), nn.modules.activation.ReLU):
            f.write('   arm_relu_q7(' + input_buffer + ', '
                    + net.name() + "_" + named_children[0].upper() + '_SIZE);\n')
        elif isinstance(getattr(net, named_children[0]), (pooling.MaxPool2d, pooling.AvgPool2d)):
            if isinstance(getattr(net, named_children[0]), pooling.MaxPool2d):
                pool_func = 'arm_maxpool_q7_HWC_nonsquare'
            elif isinstance(getattr(net, named_children[0]), pooling.AvgPool2d):
                pool_func = 'arm_avepool_q7_HWC_nonsquare'
            f.write('   ' + pool_func + '(' + input_buffer + ', '
                    + net.name() + "_" + named_children[0].upper() + '_IN_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_IN_DIM_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_IN_CH, '
                    + net.name() + "_" + named_children[0].upper() + '_KER_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_KER_DIM_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_PAD_X, '
                    + net.name() + "_" + named_children[0].upper() + '_PAD_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_STRIDE_X, '
                    + net.name() + "_" + named_children[0].upper() + '_STRIDE_Y, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_DIM_X, '
                    + net.name() + "_" + named_children[0].upper() + '_OUT_DIM_Y, col_buffer, '
                    + output_buffer + ');\n')
        if nb_layer == 1:
            input_buffer = 'buffer2'

        # on each layer (except activation layer such as ReLu) we switch the input and output buffer
        # so output of the current layer becomes input of the next layer
        # and input of the current layer can be overwrite so it becomes the output of the next layer
        if not isinstance(getattr(net, named_children[0]), nn.modules.activation.ReLU):
            input_buffer, output_buffer = output_buffer, input_buffer  # buffer1<->buffer2

    f.write('}\n')
    f.close()


def gen_code(net, folder):
    # Set the model to test mode
    # To deactivate dropout during quantization
    net.eval()

    print("\n----------------------------------------------------------------")
    print("Generate code %s..." % (net.name()))

    if not os.path.exists('/home/miguel/Downloads/myClassifier/source/Network/%s/%s' % (folder, net.name().upper())):
        os.makedirs('/home/miguel/Downloads/myClassifier/source/Network/%s/%s' % (folder, net.name().upper()))

    __generate_header(net, '/home/miguel/Downloads/myClassifier/source/Network/%s/%s/n%s.h' % (folder, net.name().upper(), net.name().upper()))
    __generate_weights(net, '/home/miguel/Downloads/myClassifier/source/Network/%s/%s/n%s_weights.h' % (folder, net.name().upper(), net.name().upper()))
    __generate_parameters(net, '/home/miguel/Downloads/myClassifier/source/Network/%s/%s/n%s_parameters.h' % (folder, net.name().upper(), net.name().upper()))
    __generate_network(net, '/home/miguel/Downloads/myClassifier/source/Network/%s/%s/n%s.c' % (folder, net.name().upper(), net.name().upper()))
