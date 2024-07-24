## Example: Employee, Designer, and Developer
# Here, we define an `Employee` class and two subclasses `Designer` and `Developer`.

### Base Class: Employee

class Employee:
    def __init__(self, name, age, department):
        self.name = name
        self.age = age
        self.department = department
    
    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}, Department: {self.department}")


### Subclass: Designer

class Designer(Employee):
    def __init__(self, name, age, department, tool):
        super().__init__(name, age, department)
        self.tool = tool
    
    def display_info(self):
        super().display_info()
        print(f"Design Tool: {self.tool}")


### Subclass: Developer

class Developer(Employee):
    def __init__(self, name, age, department, language):
        super().__init__(name, age, department)
        self.language = language
    
    def display_info(self):
        super().display_info()
        print(f"Programming Language: {self.language}")


### Usage Example

designer = Designer("Alice", 30, "Design", "Photoshop")
developer = Developer("Bob", 25, "Development", "Python")

designer.display_info()
developer.display_info()