from multiprocessing import Pool
class Test:
    A = []
    def myfunc(self, x):
        return [i for i in range(x)]

    def mycallback(self,x):
        print('callback called', x)
        self.A.append(x)
    def do_parallel(self, pool):
        a = pool.apply_async(self.myfunc, (1, 2), callback=self.mycallback)    
        a.wait()
pool = Pool()
t = Test()
t.do_parallel(pool)
# r = pool.map_async(myfunc, (1, 2), callback=mycallback)
# r.wait()
# print(A)