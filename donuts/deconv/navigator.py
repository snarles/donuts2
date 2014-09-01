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

# pass in index and return list of hashes
def query(index,field,value):
    records = []
    for i in range(len(index)):
        entry = index[i]
        ss = entry.split(' ')
        for ss2 in ss:
            ss3 = ss2.split(':')
            if ss3[0]=='field':
                if value != []:
                    if ss3[1]==value:
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
        
# special query for autonaming
