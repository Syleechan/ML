import os
num = os.listdir("./num")
print(num)
f = open("./number.txt","w")
for n in num:
    f.write(n+"\n")
f.close()


