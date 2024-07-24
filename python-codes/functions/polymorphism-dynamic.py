class Animal:
    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class Bird(Animal):
    def speak(self):
        return "Chirp!"

def animal_sound(animal: Animal):
    print(animal.speak())

# Creating instances of the subclasses
dog = Dog()
cat = Cat()
bird = Bird()

# Using the function to demonstrate dynamic polymorphism
animal_sound(dog)  # Output: Woof!
animal_sound(cat)  # Output: Meow!
animal_sound(bird)  # Output: Chirp!
