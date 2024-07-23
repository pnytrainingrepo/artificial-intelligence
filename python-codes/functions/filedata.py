#f= open('textfile.txt','r')
#print(f.read())
"""
with open("textfile.txt", "r") as file:
    data = file.readlines()
    for line in data:
        word = line.split()
        print (word)
     


file = open('textfile.txt','a+')
file.write("\n" + "What is the purpose of AI")
file.seek(0, 0)  # Move to the end of file

#file.close()

#file= open('textfile.txt','r')
print(file.read())

  
  
a = [1, 2, 3]
print ("Fourth element = %d" %(a[3]))
#try:
    #print ("Second element = %d" %(a[1]))
    #print ("Fourth element = %d" %(a[3]))

#except:
#    print ("An error occurred")
"""

# define Python user-defined exceptions
class InvalidAgeException(Exception):
    "Raised when the input value is less than 18"
    pass

# you need to guess this number
number = 18

try:
    input_num = int(input("Enter a number: "))
    if input_num < number:
        raise InvalidAgeException
    else:
        print("Eligible to Vote")
        
except InvalidAgeException:
    print("Exception occurred: Invalid Age")