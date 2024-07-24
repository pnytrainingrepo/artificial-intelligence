class Mother:
    def __init__(self,name:str) -> None:
        self.name : str = name
        self.eye_color : str = "blue"
    
    def speaking(self, words : str )->str:
        return f"Monther Speaking function: {words}"

class Father:
    def __init__(self, name:str)->None:
        self.name : str = name
        self.height : str = "6 Feet"

    def speaking(self, words : str )->str:
        return f"Father Speaking function: {words}"

class Child(Mother, Father):
    def __init__(self, mother_name : str, father_name : str , child_name: str)->None:
        Mother.__init__(self, mother_name)
        Father.__init__(self, father_name)
        self.child_name : str = child_name

qasim : Child = Child("Naseem Bano", "Muhammad Aslam","Muhammad Qasim")

print(f"object height {qasim.height}")
print(f"object eye color {qasim.eye_color}")
print(qasim.speaking("Pakistan zinda bad"))