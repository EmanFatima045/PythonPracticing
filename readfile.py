# Write into file first
with open("training_data.txt", "a") as file:
    file.write("HI this is me , Eman Fatima\n")
    file.write("I am learning how to Create Chatbot using python\n")

# Now read the file
with open("training_data.txt", "r") as file:
    data = file.read()
    print("data read from file:")
    print(data)

