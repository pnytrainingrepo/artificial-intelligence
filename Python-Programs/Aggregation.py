Code to demonstrate Aggregation 
  
# Salary class with the public method  
# annual_salary() 
class Salary: 
    def __init__(self, pay, bonus): 
        self.pay = pay 
        self.bonus = bonus 
  
    def annual_salary(self): 
        return (self.pay*12)+self.bonus 
  
  
# EmployeeOne class with public method 
# total_sal() 
class EmployeeOne: 
  
    # Here the salary parameter reflects 
    # upon the object of Salary class we 
    # will pass as parameter later 
    def __init__(self, name, age, sal): 
        self.name = name 
        self.age = age 
  
        # initializing the sal parameter 
        self.agg_salary = sal   # Aggregation 
  
    def total_sal(self): 
        return self.agg_salary.annual_salary() 
  
# Here we are creating an object  
# of the Salary class 
# in which we are passing the  
# required parameters 
salary = Salary(10000, 1500) 
  
# Now we are passing the same  
# salary object we created 
# earlier as a parameter to  
# EmployeeOne class 
emp = EmployeeOne('Geek', 25, salary) 
  
print(emp.total_sal())