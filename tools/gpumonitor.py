import pynvml

def getgpuid():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    emptylist = []
    for gpuid in range(device_count):
        gpu = pynvml.nvmlDeviceGetHandleByIndex(gpuid)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
        if len(processes)==0:
            emptylist.append(gpuid)
    pynvml.nvmlShutdown()
    str_list = list(map(str, emptylist))
    result=",".join(str_list)
    return result


if __name__=="__main__":
    print(getgpuid())