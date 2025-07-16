x=[10,20,30]
y=list(x)
x[0]=100
print(y)

D={'a':10,'b':20,'c':30}
E=dict(D)
D['a']=100
print(E)

D={'I':1,'V':5,'X':10,'L':50}
for d in D:
 print (d),D[d]