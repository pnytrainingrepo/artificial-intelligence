def my_function (x):
    return x+1
    
 
# print(my_function(1))

x= lambda a: a+3
print(x(1))


def addition(n):
    return n*n
n1 = (1,2,3,4)

result = map(addition,n1)
print(list(result))

result1 = map(lambda a : a+2,n1)
print(list(result1))
