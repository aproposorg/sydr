import numpy as np
import sys
import os

def downsampler(filepath, output_path, input_dtype, output_dtype, bit_shift, factor_ds, is_complex, chunck=4096000):
    """
    Downsampling function for raw binary signal file.
    """
    out_file = open(output_path, "wb")
    total_size = os.path.getsize(filepath)
    total_chunck = 0
    # Read data from file
    with open(filepath, 'rb') as fin:
        while True:
            raw = np.fromfile(fin, input_dtype, count=chunck)

            # Downconvert type
            #data = np.array(list(map(lambda x: (x >> bit_shift) & 0xf, list(raw))))
            data = np.array(list(map(lambda x: (x >> bit_shift), list(raw)))).astype(output_dtype)

            # Downsample
            if is_complex:        
                data_real      = data[0::2]
                data_real      = data_real[::factor_ds]
                data_imaginary = data[1::2]
                data_imaginary = data_imaginary[::factor_ds]
                data_ds = np.empty((data_real.size + data_imaginary.size), dtype=output_dtype)
                data_ds[0::2]  = data_real
                data_ds[1::2]  = data_imaginary
            else:
                data = data[::factor_ds]
            data_ds.tofile(out_file)

            total_chunck += chunck
            sys.stdout.write("\r%d%%" % (total_chunck*np.dtype(input_dtype).itemsize/total_size*100))
            sys.stdout.flush()

            # Check EOF
            if len(raw) < chunck:
                print("\nDone.")
                break

    out_file.close()

    return 0

if __name__ == '__main__':

    filepath      = "/mnt/c/Users/vmangr/Documents/Datasets/2021_11_30-TAU_Roof_Antenna_Tallysman/Novatel_20211130_40MHz_16bit_IQ_gain25.bin"
    input_dtype   = np.int16 # Data type of input
    output_dtype  = np.int8  # Data type of output

    # # 40 -> 5MHz, 16 -> 8 bits
    # print("\nConverting file to 5MHz, 8 bits")
    # downsampler(filepath, './_results/Novatel_20211130_resampled_5MHz_8bit_IQ_gain25.bin',\
    #     input_dtype, output_dtype, bit_shift=8, factor_ds=8, is_complex=True)

    # # 40 -> 4MHz, 16 -> 8 bits
    # print("\nConverting file to 4MHz, 8 bits")
    # downsampler(filepath, './_results/Novatel_20211130_resampled_4MHz_8bit_IQ_gain25.bin',\
    #     input_dtype, output_dtype, bit_shift=8, factor_ds=10, is_complex=True)
    
    # # 40 -> 10MHz, 16 -> 4 bits
    # print("\nConverting file to 10MHz, 4 bits")
    # downsampler(filepath, './_results/Novatel_20211130_resampled_10MHz_4bit_IQ_gain25.bin',\
    #     input_dtype, output_dtype, bit_shift=12, factor_ds=4, is_complex=True)

    # # 40 -> 10MHz, 16 -> 2 bits
    # print("\nConverting file to 10MHz, 2 bits")
    # downsampler(filepath, './_results/Novatel_20211130_resampled_10MHz_2bit_IQ_gain25.bin',\
    #     input_dtype, output_dtype, bit_shift=14, factor_ds=4, is_complex=True)

    # # 40 -> 10MHz, 16 -> 1 bits
    # print("\nConverting file to 10MHz, 2 bits")
    # downsampler(filepath, './_results/Novatel_20211130_resampled_10MHz_1bit_IQ_gain25.bin',\
    #     input_dtype, output_dtype, bit_shift=15, factor_ds=4, is_complex=True)