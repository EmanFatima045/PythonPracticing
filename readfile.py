example=open("training_data.txt", "r")
#read the file and print it
data=example.read()
print("data read from file:", data)
while True:
    file="training_data.txt".readline()
    if not file:
        print("End of file reached")
        break