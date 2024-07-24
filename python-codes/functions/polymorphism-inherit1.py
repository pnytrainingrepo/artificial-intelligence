class Shape:
    def __init__(self, name):
        self.name = name

    def area(self):
        pass

    def fact(self):
        return "I am a two-dimensional shape."

    def __str__(self):
        return self.name


class Square(Shape):
    def __init__(self, length):
        super().__init__("Square")
        self.length = length

    def area(self):
        return self.length**2

    def fact(self):
        return "Squares have each angle equal to 90 degrees."


class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius

    def area(self):
        pi=3.14
        return pi*self.radius**2

length = int(input("Enter the length of Square: "))
radius = int(input("Enter the radius of Circle: "))

a = Square(length)
b = Circle(radius)
print(b)
print(b.fact())
print(b.area())
print(a)
print(a.fact())
print(a.area())