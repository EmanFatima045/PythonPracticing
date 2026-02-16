class Car:
    def __init__(self, name):
        self.name = name
        self.color ="red"
car1 = Car("Toyota")
print(car1.name)
car2 = Car("Honda")
print (car2.name)
print(car1.color)
print(car2.color)
#polymorphism Example 
# Parent class
class Dresses:
    def __init__(self, name):
        self.name = name

    def wear(self):
        print("This is a dress")

# Child class 1
class EasternDresses(Dresses):
    def wear(self):
        print("Wearing Eastern dresses:")
        print("Frock, Saree, Lehenga")

# Child class 2
class WesternDresses(Dresses):
    def wear(self):
        print("Wearing Western dresses:")
        print("Skirt, Jeans, T-Shirt")

# Polymorphism in action
dresses = [
    EasternDresses("Eastern"),
    WesternDresses("Western")
]

for dress in dresses:
    print(f"Dress type: {dress.name}")
    dress.wear()
    print("-----")

#inheritance Example


     