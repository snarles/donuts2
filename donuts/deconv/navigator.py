# functions for navigating the index
# see analyses/setup_simulations.py

# use the following code in analyses dir
# 
#   f = open('datapath.txt')
#   datapath = f.read()
#   f.close()
#   f = open(datapath + 'index.txt','r')
#   index = f.read().split('\n')
#   f.close()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# write raw data file format 1 to data directory
# format 1: includes bvals, bvecs, datas
def writeraw1(datapath,fname,bvals,bvecs,datas):


# util function, not directly called
def query0(index,field,value,nameq):
    records = []
    for i in range(len(index)):
        entry = index[i]
        ss = entry.split(' ')
        for ss2 in ss:
            ss3 = ss2.split(':')
            if ss3[0]==field:
                if value != []:
                    if nameq:
                        if ss3[1][:len(value)]==value:
                            records = records + [entry]
                    elif ss3[1]==value:
                        records = records + [entry]
                else:
                    records = records + [entry]
    ans = []    
    for record in records:
        dc = {}
        ss = record.split(' ')
        for ss2 in ss:
            ss3 = ss2.split(':')
            if len(ss3) == 1:
                dc[ss3[0]]=1
            else:
                dc[ss3[0]]=ss3[1]
        ans = ans + [dc]
    return ans

# pass in index and return list of hashes
def query(index, field, value):
    return query0(index, field, value,False)

# for naming files
# eg you want to name a file 'Untitled0'
# use the command
#   name = autonamer(index, 'name', 'Untitled')
# this will check if there is already a file called 'Untitled0'
# if so, it will return the filename 'Untitled1'
# 
def autonamer(index,field,value):
    matches = query0(index, field, value,True)
    names = [m['name'] for m in matches]
    counter = -1
    for nom in names:
        nom2 = nom[len(value):]
        if is_int(nom2):
            counter = max(counter,int(nom2))
    return value+str(counter+1)
