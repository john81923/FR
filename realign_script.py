import os
from multiprocessing import Process, Pool
import sys


def run(splitnumb, runid):
    os.system(f'CUDA_VISIBLE_DEVICES={gpuid} python -m kfr.ssd+onet+fr_021921_fr_realign --splitnumb {splitnumb} --runid {runid}')

if __name__ == '__main__':
    #splitnumb = 56
    #v100 = 28
    #p40 = 28
    splitnumb = 8
    v100 = 0
    p40 = 8
    if sys.argv[1]=='v100':
        trange = range(0,v100)
        gpuid = 0
    elif sys.argv[1]=='p40':
        trange = range(v100,splitnumb)
        gpuid = 1
    print(sys.argv[1], 'run range:', trange  )
    #sys.exit(1)
    Process_list = []
    for i, id in enumerate( trange):
        Process_list.append( Process( target=run, args=(splitnumb, id)))
        Process_list[i].start()


    for i, id in enumerate(trange):
        Process_list[i].join()
    print( 'Realgin + init_cleaning Done')
