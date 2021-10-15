#!/usr/bin/env python

"""
.
.
.
Python Code
.
.
.
"""

import pycuda.gpuarray as gpuarray 
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import codecs
import matplotlib.pyplot as plt 

class cudaCipher:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # If you are using any helper function to make 
        # blocksize or gridsize calculations, you may define them
        # here as lambda functions. 
        # Quick lambda function to calculate grid dimensions
        
        # define block and grid dimensions
        #self.blockdim=(32,1,1)
        
        # kernel code wrapper
        kernelwrapper = """
            __global__ void rot13(char* in, char* out)
            {
                unsigned int idx = threadIdx.x;
                char c=in[idx];
                if (c<'a' || c>'z') {
                out[idx] = in[idx];
            } 
            else {
                if (c>'m') {
                    out[idx] = in[idx]-13;
                 }                 
            else {
                out[idx] = in[idx]+13;
                 } 
                }         
            }              
            """
        # Compile the kernel code when an instance
        # of this class is made.
        self.mod = SourceModule(kernelwrapper)

    
    def devCipher(self, sentence):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables using
        gpuarray.
        Returns
            out                             :   encrypted/decrypted result
            time_ :   execution time in milliseconds
        """
        # create Event
        start = cuda.Event()
        end = cuda.Event()        
        # Get kernel function
        func = self.mod.get_function("rot13")
        # Device memory allocation for input and output array(s)
        mem_size=len(sentence)*4
        decrypted=np.empty_like(sentence)

        #changed here
        #sentence=np.array(list(sentence))
        sentence=np.array(sentence)
        
        start.record()
        a=time.time()
                
        #d_sentence = cuda.mem_alloc(mem_size)
        #d_decrypted = cuda.mem_alloc(mem_size)
        #cuda.memcpy_htod(d_sentence, sentence)
        d_sentence = gpuarray.to_gpu(sentence)
        d_decrypted = gpuarray.to_gpu(decrypted)
        # Record execution time and execute operation.
        func(d_sentence, d_decrypted, block=(mem_size,1,1))
        # Wait for the event to complete
        # Fetch result from device to host
        #cuda.memcpy_dtoh(decrypted, d_decrypted)
        decrypted = d_decrypted.get()

        #b=time.time()       
        end.record()
        end.synchronize()
        b1=time.time()
        
        time_ = start.time_till(end)#milli seconds
        #print("time()before syn:",1000*(b-a),"time()after syn:",1000*(b1-a),"cuda event:",time)
        #print((b1-a)*1000,time_)
        # Convert output array back to string
        decrypted = str(decrypted)
        return decrypted, time_

    
    def pyCipher(self, sentence):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python.

        Returns
            decrypted                       :   encrypted/decrypted result
            time_         :   execution time in milliseconds
        """
        decrypted =""
        start = time.time()
        #decrypted = codecs.encode(sentence, 'rot13')
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
    ins=cudaCipher()
    file_path = "/home/z821496943/Desktop/HW2/deciphertext.txt"
    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    with open(file_path) as f:
        for line in f:
            # only one line actually, this is a list
            saved_f=line
            # Split string into list populated with '.' as delimiter.
            sentences = line.split(".")
    # Empty lists to hold deciphered sentences, execution times
    dev_sentences = []
    py_sentences = []
    time_dev = []
    time_py = []
    # Loop over each sentence in the list
    for sentence in sentences:
        (temp0,temp1)=ins.devCipher(sentence)
        dev_sentences.append(temp0)
        time_dev.append(temp1)
        (temp2,temp3)=ins.pyCipher(sentence)
        py_sentences.append(temp2)
        time_py.append(temp3)
    # post process the string(s) if required
    join_dev = '.'.join(dev_sentences)
    join_py = '.'.join(py_sentences)
    # Execution time
    tc = sum(time_dev)/len(sentences)
    tp = sum(time_py)/len(sentences)     
    print("CUDA output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        assert (join_py == join_dev)
    except AssertionError:
        print("Checkpoint failed: Python and CUDA kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging
    else:
        print("result match")
        
    # Dot plot the  per-sentence execution times
    plt.plot(time_dev, 'r*-')
    plt.plot(time_py,'g^-')
    plt.title("PyCUDA, Per-sentence execution times (include memory transfer)")
    plt.xlabel("Sentences")
    plt.ylabel("Execution time/ms")
    plt.grid()    
    plt.show()
