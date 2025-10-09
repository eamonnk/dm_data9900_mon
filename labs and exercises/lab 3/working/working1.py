# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import math

def combinations(n,r):
    num = math.factorial(n)
    den = math.factorial(r) * math.factorial(n-r)
    return num/den

def total_combinations(n):
    Total = 0
    for i in range(1,n+1):
        Total = Total + combinations(n,i) 
    return Total

Num_Variables = 100
for i in range(1,Num_Variables):
    print(i, "has a total of ", total_combinations(i) , " combinations.")
