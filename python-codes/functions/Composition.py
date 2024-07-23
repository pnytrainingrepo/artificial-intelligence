# Code to demonstrate Composition 
  
# Class Salary in which we are 
# declaring a public method annual salary 
class Salary: 
    def __init__(self, pay, bonus): 
        self.pay = pay 
        self.bonus = bonus 
  
    def annual_salary(self): 
        return (self.pay*12)+self.bonus 
  
  
# Class EmployeeOne which does not  
# inherit the class Salary yet we will 
# use the method annual salary using 
# Composition 
class EmployeeOne: 
    def __init__(self, name, age, pay, bonus): 
        self.name = name 
        self.age = age 
  
        # Making an object in which we are 
        # calling the Salary class 
        # with proper arguments. 
        self.obj_salary = Salary(pay, bonus)  # composition 
  
    # Method which calculates the total salary 
    # with the help of annual_salary() method 
    # declared in the Salary class 
    def total_sal(self): 
        return self.obj_salary.annual_salary() 
  
# Making an object of the class EmployeeOne 
# and providing necessary arguments 
emp = EmployeeOne('Geek', 25, 10000, 1500) 
  
# calling the total_sal method using  
# the emp object 
print(emp.total_sal()) 