def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)
def check_exp(n):
    test = n
    out = 0
    while test!=1:
        test = test/2
        out +=1
    return out
# print(check_exp(16))

def fractran(n,fraclist,limit):
    current = n 
    l = limit
    s = 0
    # print(n)
    while s<l:
        s+=1
        for frac in fraclist:
            if current*frac[0] % frac[1] == 0:
                current = int(current * frac[0]/frac[1])
                # print(current)
                if is_power_of_two(current) and current>2:
                    print(check_exp(current))
                break
        else:
            break
    

f = [[17,91],[78,85],[19,51],[23,38],[29,33],[77,29],[95,23],[77,19]\
    ,[1,17],[11,13],[13,11],[15,2],[1,7],[55,1]]
n = 2
limit = 10**6
fractran(n,f,limit)