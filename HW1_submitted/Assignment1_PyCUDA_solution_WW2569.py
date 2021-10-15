"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt 

class deviceAdd:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 3 functions
        # you will call from this class.
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernelwrapper = """
            __global__ void sum(float *c, float *a, float *b, const unsigned int n)
            {
                unsigned int idx = threadIdx.x + threadIdx.y*4;
                if (idx < n) 
                {c[idx] = a[idx] + b[idx];}
                            }              
            """
        return SourceModule(kernelwrapper)

    
    def explicitAdd(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """

        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()
        #start.synchronize()#not sure
        # Device memory allocation for input and output arrays
	#a_gpu = cuda.mem_alloc(a.nbytes)#another way
        a_g = cuda.mem_alloc(a.size * a.dtype.itemsize)
        b_g = cuda.mem_alloc(b.size * b.dtype.itemsize)
        c_g = cuda.mem_alloc(a.size * a.dtype.itemsize)
        # Copy data from host to device
        start.record()          #include memory transfer
        cuda.memcpy_htod(a_g, a)
        cuda.memcpy_htod(b_g, b)
        # Call the kernel function from the compiled module
        func = self.mod.get_function("sum")
        # Get grid and block dim
        blockdim=(16,1,1)
	# Record execution time and call the kernel loaded to the device
        '''
        start.record#exclude memory transfer
        '''
        func(c_g,a_g,b_g,np.int32(length),block=blockdim)
        '''
        end.record()#exclude memory transfer
             
        # Wait for the event to complete(exclude memory transfer)
        end.synchronize()
        '''

        # Copy result from device to the host
        c = np.empty_like(a)
        cuda.memcpy_dtoh(c, c_g)
        end.record()            #include memory transfer
        end.synchronize()
        millis = start.time_till(end)

        # return a tuple of output of addition and time taken to execute the operation.
        return (c, millis)

    
    def implicitAdd(self, a, b, length):
        """S
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # Event objects to mark the start and end points
        start = cuda.Event()
        end = cuda.Event()
        # Get grid and block dim
        blockdim=(16,1,1)
        # Call the kernel function from the compiled module
        func = self.mod.get_function("sum")
        # Record execution time and call the kernel loaded to the device
        c=np.empty_like(a)
        start.record()  #can only include memory transfer
        func(cuda.Out(c),cuda.In(a),cuda.In(b),np.int32(length),block=blockdim)
        end.record()    #can only include memory transfer
        # Wait for the event to complete
        end.synchronize()
        millis = start.time_till(end)
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, millis)


    def gpuarrayAdd_np(self, a, b):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # Event objects to mark start and end points
        start = cuda.Event()
        end = cuda.Event()
        # Allocate device memory using gpuarray class
        start.record()     
        a_g = gpuarray.to_gpu(a)
        b_g = gpuarray.to_gpu(b)
        # Record execution time and execute operation with numpy syntax
        '''
        start.record()
        '''        
        c_g = (a_g+b_g)
        '''
        end.record()
        # Wait for the event to complete
        end.synchronize()
        '''
        # Fetch result from device to host
        c = c_g.get()
        end.record()
        end.synchronize()
        millis = start.time_till(end)
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, millis)
        
    def gpuarrayAdd(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that 
        you call the kernel function.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """

        # Create cuda events to mark the start and end of array.
        start = cuda.Event()
        end = cuda.Event()
        # Get function defined in class defination
        func = self.mod.get_function("sum")
        # Allocate device memory for a, b, output of addition using gpuarray class        
        start.record()     
        a_g = gpuarray.to_gpu(a)
        b_g = gpuarray.to_gpu(b)
        c_g = gpuarray.empty_like(a_g)
        # Get grid and block dim
        blockdim=(16,1,1)
        # Record execution time and execute operation
        '''        
        #start.record()  #exclude memory transfer
        '''        
        func(c_g, a_g, b_g,np.int32(length),block=blockdim)
        '''        
        end.record()    #exclude memory transfer
        # Wait for the event to complete
        end.synchronize()
        '''
        # Fetch result from device to host
        c = c_g.get()
        end.record()
        end.synchronize()
        millis = start.time_till(end)
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, millis)

    def numpyAdd(self, a, b):
        """
        Function to perform on-host vector addition. The operation
        is defined using numpy-like syntax. 
        Returns (addition result, execution time)
        """
        # Initialize empty array on host to store result
        start = time.time()
        c = np.add(a, b)
        end = time.time()
        
        return c, end - start

    def serialAdd(self, a, b):
        c=np.empty_like(a)
        start = time.time()
        for i in range(len(c)):
            c[i]=a[i]+b[i]
        end=time.time()
        return c, end - start


if __name__ == "__main__":
    # Define the number of iterations and starting lengths of vectors
    iter=20
    #length=np.array([10,100,1000]).astype(np.uint32)
    length=np.array([1,10,100,1000,10000,100000,1000000,10000000,100000000]).astype(np.uint32)
    time_mat=np.zeros((6,len(length),iter))
    # Create an instance of the deviceAdd class
    ins=deviceAdd()
    # Perform addition tests for increasing lengths of vectors
    # L = 10, 100, 1000 ..., (You can use np.random.randn to generate two vectors)
    for l in range(len(length)):
        for it in range(iter):    
            a_np=np.random.randn(length[l]).astype(np.float32)
            b_np=np.random.randn(length[l]).astype(np.float32)
            (c1, ex_time1)=ins.explicitAdd(a_np, b_np, length[l])
            (c2, ex_time2)=ins.implicitAdd(a_np, b_np, length[l])
            (c3, ex_time3)=ins.gpuarrayAdd_np(a_np, b_np)
            (c4, ex_time4)=ins.gpuarrayAdd(a_np, b_np, length[l])
            c_np, time_np=ins.numpyAdd(a_np, b_np)
            c_s, time_s=ins.serialAdd(a_np, b_np)
            time_mat[0][l][it]=ex_time1
            time_mat[1][l][it]=ex_time2
            time_mat[2][l][it]=ex_time3
            time_mat[3][l][it]=ex_time4
            time_mat[4][l][it]=time_np
            time_mat[5][l][it]=time_s
                           
    # Compare outputs.
    try:
        print("Checkpoint: Do python and cuda result match? Checking...")
        assert ((c_np-c1).all()==0)        
        assert ((c_np-c2).all()==0)
        assert ((c_np-c3).all()==0)
        assert ((c_np-c4).all()==0)
        assert ((c_np-c_s).all()==0)
    except AssertionError:
        print("Checkpoint failed: Python and cuda kernel result do not match. Try Again!")
    else:
        print("result match")

    # Plot the compute times
    time_mat=np.mean(time_mat, axis=2)
    print(time_mat)
    '''
    plt.plot(np.log10(length), np.log2(time_mat[0]), 'r*-')
    plt.plot(np.log10(length), np.log2(time_mat[1]), 'g^-')
    plt.plot(np.log10(length), np.log2(time_mat[2]), 'bo-')
    plt.plot(np.log10(length), np.log2(time_mat[3]), 'cp-')
    plt.plot(np.log10(length), np.log2(time_mat[4]), 'mh-')
    '''

    plt.plot(np.log10(length[1:]), time_mat[0,1:], 'r*-')
    plt.plot(np.log10(length[1:]), time_mat[1,1:], 'g^-')
    plt.plot(np.log10(length[1:]), time_mat[2,1:], 'bo-')
    plt.plot(np.log10(length[1:]), time_mat[3,1:], 'cp-')
    plt.plot(np.log10(length[1:]), time_mat[4,1:], 'mh-')
    plt.plot(np.log10(length[1:]), time_mat[5,1:], 'ys-')

    
    plt.legend(['explicitAdd', 'implicitAdd', 'gpuarrayAdd_np','gpuarrayAdd','numpyAdd','serialAdd'], loc='upper left')
    plt.ylabel('Average Running Time/ms')
    plt.xlabel('log10(Array length)')
    plt.title("Time-Array length(include memory transfer)")
    plt.grid()
    #plt.axis([1, np.log10(length[-1]), np.min(np.log2(time_mat)), np.max(np.log2(time_mat))])
    plt.show() 
