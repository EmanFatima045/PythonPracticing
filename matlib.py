import matplotlib.pyplot as plt

class Circle:
    def __init__(self, radius, color):
        self.radius = radius
        self.color = color

    def add_radius(self, add):
        self.radius = self.radius + add

    def draw(self):
        circle1 = plt.Circle((0, 0), self.radius, color=self.color)
        plt.gca().add_patch(circle1)
        plt.axis('equal')
        plt.show()

# Create object
c1= Circle(5, "blue")
c2 = Circle(9 ,"yellow")
c1.draw()
c2.draw()