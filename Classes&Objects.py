class Faculty:
    def putdata(self):
        self.name = input("Enter Employee name: ")
        self.id = input("Enter Employee id: ")
        self.salary = float(input("Enter salary: "))

    def display(self):
        print("\n--- Faculty Details ---")
        print("Name:", self.name)
        print("ID:", self.id)
        print("Salary:", self.salary)


class Student:
    def putdata(self):
        print("\nEnter the Student Details:")
        self.name = input("Enter Student name: ")
        self.id = input("Enter Lecture Name: ")
        self.marks = float(input("Enter marks: "))

    def display(self):
        print("\n--- Student Details ---")
        print("Name:", self.name)
        print("Lecture Name:", self.id)
        print("Marks:", self.marks)


# Creating objects
f1 = Faculty()
f1.putdata()
f1.display()

s1 = Student()
s1.putdata()
s1.display()
