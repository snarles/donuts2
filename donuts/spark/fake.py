
# coding: utf-8

# In[ ]:

class FakeRDD:
    partitions = []
    def __init__(self, partitions):
        self.partitions = partitions
        
    def map(self, func):
        newpartitions = []
        for partition in self.partitions:
            newpartition = [func(element) for element in partition]
            newpartitions.append(newpartition)
        return FakeRDD(newpartitions)
    
    def flatMap(self, func):
        newpartitions = []
        for partition in self.partitions:
            newpartition = []
            for element in partition:
                newpartition = newpartition + func(element)
            newpartitions.append(newpartition)
        return FakeRDD(newpartitions)        
    
    def mapPartitions(self, func):
        newpartitions = []
        for partition in self.partitions:
            newpartitions.append(func(iter(partition)))
        return FakeRDD(newpartitions)   
    
    def reduce(self, func):
        a_list = self.collect()
        n_items = len(a_list)
        if n_items < 2:
            return []
        ans = func(a_list[0], a_list[1])
        for ind in range(2, n_items):
            ans = func(ans, a_list[ind])
        return ans
    
    def collect(self):
        ans = []
        for partition in self.partitions:
            ans = ans + partition
        return ans
    
    def cache(self):
        return
    
class FakeSparkContext:
    name = ''
    def __init__(self, name = 'FakeContext'):
        self.name = name
        
    def textFile(self, tf, n_parts):
        ff = open(tf, 'r')
        a_list = ff.read().split('\n')
        ff.close()
        return self.parallelize(a_list, n_parts)
    
    def parallelize(self, a_list, n_parts):
        n_items = len(a_list)
        sz_part = n_items/n_parts + 1
        count = 0
        partitions = []
        for ind in range(n_parts):
            newcount = min(count + sz_part, n_items)
            newpartition = a_list[count:newcount]
            partitions.append(newpartition)
            count = newcount
        return FakeRDD(partitions)

