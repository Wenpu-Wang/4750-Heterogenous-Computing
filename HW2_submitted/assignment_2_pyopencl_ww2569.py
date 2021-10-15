import numpy as np
import codecs
import time
from pyopencl import Event
import pyopencl as cl
import matplotlib.pyplot as plt

class clCipher:
    def __init__(self):
        """
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code
        and input variables.
        """
        
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
        	if platform.name == NAME:
        		devs = platform.get_devices()       
        
        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # kernel
        kernel_code ='''
            __kernel void rot13 (__global const char* in, __global char* out ){ 
                const uint index = get_global_id(0); 
                char c=in[index];
            if (c<'a' || c>'z') {
                out[index] = in[index];
            } 
            else {
                if (c>'m') {
                    out[index] = in[index]-13;
                 }                 
            else {
                out[index] = in[index]+13;
                 } 
                }
            }
        '''
        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def devCipher(self, sentence):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables.
        Returns
            decrypted :   decrypted/encrypted result
            time_     :   execution time in milliseconds
        """

        # Text pre-processing/list comprehension (if required)
        # Depends on how you approach the problem
        # device memory allocation
        mem_size=len(sentence)*4
        decrypted=np.empty_like(sentence)
        sentence=np.array(sentence)
        mf=cl.mem_flags
        #-------------------------
        start = time.time()
        d_sentence = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sentence)
        d_decrypted = cl.Buffer(self.ctx, mf.WRITE_ONLY, mem_size)
        # Call the kernel function and time event execution
        evt = self.prg.rot13(self.queue, (mem_size,), None, d_sentence, d_decrypted)#######
        # OpenCL event profiling returns times in nanoseconds. 
        # Hence, 1e-6 will provide the time in milliseconds, 
        # making your plots easier to read.
        evt.wait()
                #time_ = (evt.profile.end - evt.profile.start)*(10**(-6))
        # Copy result to host memory
        cl.enqueue_copy(self.queue, decrypted, d_decrypted)
        end = time.time()
        #----------------------------
        time_ = (end - start)*1000
        decrypted=str(decrypted)
        return decrypted, time_

    
    def pyCipher(self, sentence):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python. (String manipulation and list comprehension
        will prove useful.)

        Returns
            decrypted                  :   decrypted/encrypted result
            time_    :   execution time in milliseconds
        """
        decrypted =""
        start = time.time()
        # decrypted = codecs.encode(sentence, 'rot13')
        for c in sentence:
            if c<'a' or c>'z':   
                decrypted += c
            elif c>'m':
                decrypted += chr(ord(c)-13)
            else:
                decrypted += chr(ord(c)+13)
        end = time.time()
        time_ = (end - start)*1000
        return decrypted, time_


if __name__ == "__main__":
    # Main code
    ins=clCipher()
    file_path = "/home/z821496943/Desktop/HW2/deciphertext.txt"
    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    with open(file_path) as f:
        for line in f:
            # only one line actually, this is a list
            saved_f=line
            sentences = line.split(".")
    #print(len(sentences))
    # Loop over each sentence in the list
    dev_sentences = []
    py_sentences = []
    time_dev = []
    time_py = []
    for sentence in sentences:
        (temp0,temp1)=ins.devCipher(sentence)
        dev_sentences.append(temp0)
        time_dev.append(temp1)
        (temp2,temp3)=ins.pyCipher(sentence)
        py_sentences.append(temp2)
        time_py.append(temp3)
    #print(len(dev_sentences),len(py_sentences))
    # Stitch decrypted sentences together
    join_dev = '.'.join(dev_sentences)
    join_py = '.'.join(py_sentences)
    #print(len(join_dev),len(join_py),len(saved_f))
    tc = sum(time_dev)/len(sentences)
    tp = sum(time_py)/len(sentences)    
    print("OpenCL output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")
    #print(join_dev)

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        # compare outputs
        assert (join_py == join_dev)
    except AssertionError:
        print("Checkpoint failed: Python and OpenCL kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging    
    else:
        print("result match")
        
    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times

    #if #conditions met: 

        # Write cuda output to file
    #with open('data.txt','w') as f:
    #    f.write(join_dev)
    # Scatter plot the  per-sentence execution times
    plt.plot(time_dev, 'r*-')
    plt.plot(time_py,'g^-')
    plt.title("PyOpenCL, Per-sentence execution times (include memory transfer)")
    plt.xlabel("Sentences")
    plt.ylabel("Execution time/ms")
    plt.grid()    
    plt.show()
