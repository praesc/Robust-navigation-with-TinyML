import torch
import numpy as np
import torch.nn.modules.linear as linear

# --------------------------------------------------------------------
# this code should quantize any classification network
# the only restriction is to load a dataset and create classes attribute
# --------------------------------------------------------------------


# Quantize weights to 8 bits (QM.N)
# Using min and max of weights as nearest power of 2
def __quantize_wt(net):
    print('\nWeight')
    # iterate through every layer that has weight
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight'):
            # Find min/max of weights
            wt_min = getattr(net, named_children[0]).weight.min().item()
            wt_max = getattr(net, named_children[0]).weight.max().item()

            # find number of integer bits and fractional bits to represent this range
            # rounding up to nearest power of 2.
            # new attribute containing the integer bits and fractional bits are created
            getattr(net, named_children[0]).wt_int_bits = int(np.ceil(np.log2(max(abs(wt_min), abs(wt_max)))))  # find M (QM.N)
            getattr(net, named_children[0]).wt_dec_bits = 7 - getattr(net, named_children[0]).wt_int_bits       # find N (QM.N)

            print('Layer: ' + named_children[0] + ' Format: Q' + str(getattr(net, named_children[0]).wt_int_bits) + '.' + str(getattr(net, named_children[0]).wt_dec_bits))

            # floating point weights are scaled and rounded to [-128,127], which are used in
            # the fixed-point operations on the actual hardware (i.e., micro-controller)
            # a new attribute containing the quantized weight is created in each layer
            getattr(net, named_children[0]).quant_weight = (getattr(net, named_children[0]).weight * (2 ** getattr(net, named_children[0]).wt_dec_bits)).round()

            # To quantify the impact of quantized weights, we scale them back to
            # original range to run inference using quantized weights
            getattr(net, named_children[0]).weight = torch.nn.Parameter(getattr(net, named_children[0]).quant_weight / (2 ** getattr(net, named_children[0]).wt_dec_bits))


# Quantize bias to 8 bits (QM.N)
# Using min and max of weights as nearest power of 2
def __quantize_bias(net):
    print('\nBias')
    # iterate through every layer that has bias
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'bias'):
            # Start with min/max of weights
            bias_min = getattr(net, named_children[0]).bias.min().item()
            bias_max = getattr(net, named_children[0]).bias.max().item()

            # find number of integer bits and fractional bits to represent this range
            # rounding up to nearest power of 2.
            # new attribute containing the integer bits and fractional bits are created
            getattr(net, named_children[0]).bias_int_bits = int(np.ceil(np.log2(max(abs(bias_min), abs(bias_max)))))    # find M (QM.N)
            getattr(net, named_children[0]).bias_dec_bits = 7 - getattr(net, named_children[0]).bias_int_bits           # find N (QM.N)

            print('Layer: ' + named_children[0] + ' Format: Q' + str(getattr(net, named_children[0]).bias_int_bits) + '.' + str(getattr(net, named_children[0]).bias_dec_bits))

            # floating point weights are scaled and rounded to [-128,127], which are used in
            # the fixed-point operations on the actual hardware (i.e., micro-controller)
            # a new attribute containing the quantized bias is created in each layer
            getattr(net, named_children[0]).quant_bias = (getattr(net, named_children[0]).bias * (2 ** getattr(net, named_children[0]).bias_dec_bits)).round()

            # To quantify the impact of quantized weights, we scale them back to
            # original range to run inference using quantized weights
            getattr(net, named_children[0]).bias = torch.nn.Parameter(getattr(net, named_children[0]).quant_bias / (2 ** getattr(net, named_children[0]).bias_dec_bits))


# Quantize activations (inter-layer data) to 8 bits (QM.N)
# Using min and max of activations as nearest power of 2
# By iterating trough the entire test dataset
def __quantize_activation(net, validationSet, classes):
    print('\nActivation')
    validationloader = torch.utils.data.DataLoader(validationSet,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=2)
    # create new attribute for all the needed values
    # for each layer
    for named_children in net.named_children():
        getattr(net, named_children[0]).max_in_val = float('-inf')
        getattr(net, named_children[0]).min_in_val = float('inf')
        getattr(net, named_children[0]).max_out_val = float('-inf')
        getattr(net, named_children[0]).min_out_val = float('inf')
        getattr(net, named_children[0]).input_shape = None
        getattr(net, named_children[0]).input_nb_element = None

    # for the network
    net.max_in_val = float('-inf')
    net.min_in_val = float('inf')
    net.max_out_val = float('-inf')
    net.min_out_val = float('inf')
    net.input_shape = None
    net.input_nb_element = None
    net.output_shape = None
    net.output_nb_element = None

    # iterating trough the entire test dataset
    for data in validationloader:
        # get the input and output
        images, labels = data

        # convert the inputs to 1 channel
        images = images[:, 0, :, :].unsqueeze(1)

        # get the net input shape and size (always the same through the dataset)
        net.input_shape = images.shape
        net.input_nb_element = images.nelement()

        # get the min and max value of the net input (very max and very min)
        if images.max().item() > net.max_in_val:
            net.max_in_val = images.max().item()
        if images.min().item() < net.min_in_val:
            net.min_in_val = images.min().item()

        inputs = images

        # quantize layer by layer
        # basically forward function of the network but layer by layer
        for named_children in net.named_children():

            # get the layer input shape and size (always the same through the dataset)
            getattr(net, named_children[0]).input_shape = inputs.shape
            getattr(net, named_children[0]).input_nb_element = inputs.nelement()

            # flatten function if next layer is linear
            if isinstance(getattr(net, named_children[0]), linear.Linear) and len(list(inputs.shape)) > 2:
                inputs = inputs.flatten()

            # we only quantize layer with weight
            if hasattr(getattr(net, named_children[0]), 'weight'):
                # get the min and max value of the layer input (very max and very min)
                if inputs.max().item() > getattr(net, named_children[0]).max_in_val:
                    getattr(net, named_children[0]).max_in_val = inputs.max().item()
                if inputs.min().item() < getattr(net, named_children[0]).min_in_val:
                    getattr(net, named_children[0]).min_in_val = inputs.min().item()

            # forward for one layer
            outputs = getattr(net, named_children[0])(inputs)

            # get the layer output shape and size (always the same through the dataset)
            getattr(net, named_children[0]).output_shape = outputs.shape
            getattr(net, named_children[0]).output_nb_element = outputs.nelement()

            # we only quantize layer with weight
            if hasattr(getattr(net, named_children[0]), 'weight'):
                # get the min and max value of the layer output (very max and very min)
                if outputs.max().item() > getattr(net, named_children[0]).max_out_val:
                    getattr(net, named_children[0]).max_out_val = outputs.max().item()
                if outputs.min().item() < getattr(net, named_children[0]).min_out_val:
                    getattr(net, named_children[0]).min_out_val = outputs.min().item()

            # output of this layer is now input of the next layer
            inputs = outputs

        # get the net output shape and size (always the same through the dataset)
        net.output_shape = outputs.shape
        net.output_nb_element = outputs.nelement()

        # get the min and max value of the net output (very max and very min)
        if outputs.max().item() > net.max_out_val:
            net.max_out_val = outputs.max().item()
        if outputs.min().item() < net.min_out_val:
            net.min_out_val = outputs.min().item()

    # Now that we have all the info (min and max) for each layer,
    # we can easily follow the same flow as the weight and bias to find the format of the data (QM.N)

    # first we find the QM.N format of the input data of the network
    # find number of integer bits and fractional bits to represent this range
    # rounding up to nearest power of 2.
    # new attribute containing the integer bits and fractional bits are created
    net.in_int_bits = int(np.ceil(np.log2(max(abs(net.max_in_val), abs(net.min_in_val)))))
    net.in_dec_bits = 7 - net.in_int_bits
    print('Data format: Q' + str(net.in_int_bits) + '.' + str(net.in_dec_bits))

    # next we find the QM.N format of the input and output data of each layer
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight'):
            # find number of integer bits and fractional bits to represent this range
            # rounding up to nearest power of 2.
            # new attribute containing the integer bits and fractional bits are created
            getattr(net, named_children[0]).in_int_bits = int(np.ceil(np.log2(max(abs(getattr(net, named_children[0]).max_in_val), abs(getattr(net, named_children[0]).min_in_val)))))
            getattr(net, named_children[0]).out_int_bits = int(np.ceil(np.log2(max(abs(getattr(net, named_children[0]).max_out_val), abs(getattr(net, named_children[0]).min_out_val)))))
            getattr(net, named_children[0]).in_dec_bits = 7 - getattr(net, named_children[0]).in_int_bits
            getattr(net, named_children[0]).out_dec_bits = 7 - getattr(net, named_children[0]).out_int_bits
            print(named_children[0] + ' in Format: Q' + str(getattr(net, named_children[0]).in_int_bits) + '.' + str(getattr(net, named_children[0]).in_dec_bits) +
                  ' out Format: Q' + str(getattr(net, named_children[0]).out_int_bits) + '.' + str(getattr(net, named_children[0]).out_dec_bits))

    # then we find the QM.N format of the output data of the network
    # find number of integer bits and fractional bits to represent this range
    # rounding up to nearest power of 2.
    # new attribute containing the integer bits and fractional bits are created
    net.out_int_bits = int(np.ceil(np.log2(max(abs(net.max_out_val), abs(net.min_out_val)))))
    net.out_dec_bits = 7 - net.out_int_bits
    print('Data format: Q' + str(net.out_int_bits) + '.' + str(net.out_dec_bits) + '\n')

    # last but not least we can test the accuracy of the network with the quantized data, weight an bias
    # Testing accuracy with quantized activation
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    nb_image_total = 0
    nb_image_correct = 0

    # iterating trough the entire test dataset
    #'''
    for data in validationloader:
        # get the input and output
        images, labels = data

        # convert the inputs to 1 channel
        images = images[:, 0, :, :].unsqueeze(1)

        inputs = images

        # floating point data are scaled and rounded to [-128,127], which are used in
        # the fixed-point operations on the actual hardware (i.e., micro-controller)
        quant_inputs = (inputs * (2 ** net.in_dec_bits)).round()

        # To quantify the impact of quantized data, we scale them back to
        # original range to run inference using quantized data
        inputs = torch.nn.Parameter(quant_inputs / (2 ** net.in_dec_bits))

        # quantize layer by layer
        # basically forward function of the network but layer by layer
        for named_children in net.named_children():
            # flatten function if next layer is linear
            if isinstance(getattr(net, named_children[0]), linear.Linear) and len(list(inputs.shape)) > 2:
                inputs = inputs.flatten()

            # forward for one layer
            outputs = getattr(net, named_children[0])(inputs)

            # we only quantize layer with weight
            if hasattr(getattr(net, named_children[0]), 'weight'):
                # floating point data are scaled and rounded to [-128,127], which are used in
                # the fixed-point operations on the actual hardware (i.e., micro-controller)
                quant_outputs = (outputs * (2 ** getattr(net, named_children[0]).out_dec_bits)).round()

                # To quantify the impact of quantized data, we scale them back to
                # original range to run inference using quantized data
                outputs = torch.nn.Parameter(quant_outputs / (2 ** getattr(net, named_children[0]).out_dec_bits))

            # output of this layer is now input of the next layer
            inputs = outputs

        # count total true
        if labels.item() == outputs.data.tolist().index(outputs.max().item()):
            nb_image_correct += 1

        nb_image_total += 1

        for i in range(1):
            label = labels[i]
            if labels.item() == outputs.data.tolist().index(outputs.max().item()):
                class_correct[label] += 1
            class_total[label] += 1

    # print accuracy for each classes
    for i in range(len(classes)):
        print('Accuracy of %15s : %3.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # print accuracy for network
    print("%s accuracy after activation quantization : %.2f" % (net.name(), nb_image_correct * 100 / nb_image_total))
    #'''


# in CMSIS-NN, all the weight, bias and activation data are stored in fixed point but are computed with floating point
# so we have to indicate how to shift the point to align them before computing so the result are correct
# the computation flow is : Weight Qa.b * Input Qc.d + Bias Qe.f -> Output Qx.y
# with floating point -> y = b + d
#                        x = 31 - y (31 bits and 1 sign bit)
# so bias has to be shifted form y - f to be aligned
# and since the output data is in floating point, we have to quantize it with the previously found value before storing it
def __quantize_shift(net):
    # find the shift value for the input data
    # in this case the data are already in Int8 format so we don't need to shift it
    net.in_rshift = 0  # net.in_dec_bits

    # find the shift value for the computation of each layer with weight
    for named_children in net.named_children():
        if hasattr(getattr(net, named_children[0]), 'weight'):
            # fint y of Qx.y
            mac_dec_bits = getattr(net, named_children[0]).wt_dec_bits + getattr(net, named_children[0]).in_dec_bits

            # find the bias shift value
            getattr(net, named_children[0]).bias_lshift = mac_dec_bits - getattr(net, named_children[0]).bias_dec_bits

            # find the output shift value
            getattr(net, named_children[0]).act_rshift = mac_dec_bits - getattr(net, named_children[0]).out_dec_bits

            # find the last output shift value so we can scale de value back to its floating point form
            # useful for regression network but not for classification
            net.out_rshift = getattr(net, named_children[0]).act_rshift


def quantize(net, validationSet, classes):
    # Set the model to test mode
    # To deactivate dropout during quantization
    net.eval()

    print("\n----------------------------------------------------------------")
    print("Quantization of %s..." % (net.name()))

    __quantize_wt(net)
    __quantize_bias(net)
    __quantize_activation(net, validationSet, classes)
    __quantize_shift(net)
