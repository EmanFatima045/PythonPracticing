# inheritance takes properties and behaviours from parent class
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating")

#Dog class inherits From Animal Class
#Dog uses eat mehtod from Animal class and also has its own bark() mehtod
class Dog(Animal):
    def bark(self):
        print(f"{self.name} is barking")


dog1 = Dog("Buddy")
print(dog1.name)
dog1.eat()
dog1.bark()

