#Smart Message Analyzer 
1. # takes user input
message = input ("Enter your message here : ")
print(message)
2.# count total words
words = message.split()
print("Total words:", len(words))
3.#count total Characters
print("Total characters:", len(message))
4.#count vowels
vowels_count =0
for char in message:
    if char.lower() in "aeiou":
        vowels_count+=1
print("vowels:" ,vowels_count)
5.#digit count
digit_count=0
for char in message:
    if char.isdigit():
     digit_count+=1 
print("Total Digits:" ,digit_count)
6.#reverse message logic
print("reverse message",message[::-1])
7. #sentiment Check analysis
if "good" in message.lower():
   print("positive")
elif "bad"in message.lower():
    print("negative")
else:
    print("neutral")
    
    
    
   

    
