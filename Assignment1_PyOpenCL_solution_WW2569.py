"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""


# import relevant.libraries
import numpy as np
import pyopencl as cl
import pyopencl.array as ar
import time
from pyopencl import Event
import matplotlib.pyplot as plt 

class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """

        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()       

        # Create Context:
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # kernel - will not be provided for future assignments!
        kernel_code = """
            __kernel void sum(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        """ 

        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceAdd(self, a, b, length):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a       :   1st Vector
            b       :   2nd Vector
            length  :   length of vectors.
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # device memory allocation
        rtime=time.time()#start timer
        a_g = ar.to_device(self.queue, a)
        b_g = ar.to_device(self.queue, b)
        c_g = ar.empty_like(a_g)
        # execute operation.
        #rtime=time.time()#start timer
        ev=self.prg.sum(self.queue, a.shape, None, c_g.data, a_g.data, b_g.data, length)
        # wait for execution to complete.
        ev.wait()
        #ex_time=time.time()-rtime
        # Copy output from GPU to CPU [Use .get() method]
        c=c_g.get()
        # Record execution time.
        ex_time=time.time()-rtime
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, ex_time)

    def bufferAdd(self, a, b, length):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # Create three buffers (plans for areas of memory on the device)
        mf=cl.mem_flags
        rtime=time.time()#start timer 
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, a.nbytes)
        # execute operation.
        #self.prg.sum.set_scalar_arg_dtypes([None, None, None, np.uint32])#type
        #rtime=time.time()#start timer        
        ev=self.prg.sum(self.queue, a.shape, None, c_g, a_g, b_g, length)
        # Wait for execution to complete.
        ev.wait()
        #self.queue.finish()
        #ex_time=time.time()-rtime
        # Copy output from GPU to CPU [Use enqueue_copy]
        c = np.empty_like(a)
        cl.enqueue_copy(self.queue, c, c_g)
        # Record execution time.
        ex_time=time.time()-rtime
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, ex_time)

    def numpyAdd(self, a, b, length):
        """
        Function to perform vector addition on host(CPU).
        Arguments:
            a       :   1st Vector
            b       :   2nd Vector
            length  :   length of vector a or b[since they are of same length] 
        """
        a = np.array(a)
        b = np.array(b)

        start = time.time()
        c = a + b
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
    length=np.array([1, 10, 100, 1000, 10000,100000,1000000,10000000,100000000]).astype(np.uint32)
    #length=np.array([10, 100,1000, 10000]).astype(np.uint32)
    time_mat=np.zeros((4,len(length),iter))
    # Create an instance of the clModule class
    ins=clModule()
    # Perform addition tests for increasing lengths of vectors
    # L = 10, 100, 1000 ..., (You can use np.random.randn to generate two vectors)
    for l in range(len(length)):
        for it in range(iter):
            a_np=np.random.randn(length[l]).astype(np.float32)
            b_np=np.random.randn(length[l]).astype(np.float32)
            (c1, ex_time1)=ins.deviceAdd(a_np, b_np, length[l])
            (c2, ex_time2)=ins.bufferAdd(a_np, b_np, length[l])
            c_np, time_np=ins.numpyAdd(a_np, b_np, length[l])
            c_s, time_s=ins.serialAdd(a_np, b_np)
            #print(length[l])
            time_mat[0][l][it]=ex_time1
            time_mat[1][l][it]=ex_time2
            time_mat[2][l][it]=time_np
            time_mat[3][l][it]=time_s
    # Compare outputs.
    try:
        print("Checkpoint: Do python and opencl result match? Checking...")
        assert ((c_np-c2).all()==0)
        assert ((c_np-c1).all()==0)
        assert ((c_np-c_s).all()==0)
    except AssertionError:
        print("Checkpoint failed: Python and opencl kernel result do not match. Try Again!")
    else:
        print("result match")

    # Plot the compute times
    time_mat=1000*np.mean(time_mat, axis=2)
    print(time_mat)
    plt.plot(np.log10(length[1:]), time_mat[0,1:], 'r*-')
    plt.plot(np.log10(length[1:]), time_mat[1,1:], 'bo-')
    plt.plot(np.log10(length[1:]), time_mat[2,1:], 'cp-')
    plt.plot(np.log10(length[1:]), time_mat[3,1:], 'ys-')
    plt.legend(['deviceAdd', 'bufferAdd', 'numpyAdd','serialAdd'], loc='upper left')
    plt.ylabel('Average Running Time/ms')
    plt.xlabel('log10(Array length)')
    plt.title("Time-Array length(include memory transfer)")
    plt.grid()
    #plt.axis([1, np.log10(length[-1]), np.min(np.log2(time_mat)), np.max(np.log2(time_mat))])
    plt.show()
