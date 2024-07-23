def my_function (x):
    return x+1
    
 
# print(my_function(1))

#x= lambda a: a+3
#print(x(1))


#def addition(n):
 #   return n*n
#n1 = (1,2,3,4)

#result = map(addition,n1)
#print(list(result))

#result1 = map(lambda a : a+2,n1)
#print(list(result1))

seq = [0, 1, 2, 3, 5, 8, 13] 

# result contains odd numbers of the list
result = filter(lambda x: x % 2 != 0, seq)
print(list(result))

# result contains even numbers of the list
result = filter(lambda x: x % 2 == 0, seq)
print(list(result))

def outerFunction(text):
       text = text
       def innerFunction():
        print(text)
       innerFunction()
outerFunction('Hey !') 


def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function 
add_five = outer_function(5)
result = add_five(10) # result is 15
print(result)
