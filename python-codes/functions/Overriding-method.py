class Animal():
    def eating(self,food : str )->None: #same method 
        print(f"Animal is eating {food}")


class Bird(Animal):
    def eating(self, food: str) -> None:
        print(f"Bird is eating {food}")


bird : Bird = Bird()
bird.eating("bread")

animal : Animal = Animal()
animal.eating("grass")