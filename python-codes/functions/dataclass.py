from dataclasses import dataclass


@dataclass
class Exercise:
   name: str
   reps: int
   sets: int
   weight: float


ex1 = Exercise("Bench press", 10, 3, 52.5)

# Verifying Exercise is a regular class
ex1.name
'Bench press'