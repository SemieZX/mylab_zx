nums,price = map(float,input().split(" "))
if nums >= 3:
    price1 = price*nums * 0.7
    if price1 >= 50:
        price1 = price1 - 10
else:
    price1 = price*nums
    if price1 >= 50:
        price1 = price1 - 10

if price*nums >= 10:
    price2 = price*nums - 2
    if price2 >= 99:
        price2 = price2 - 6
else:
    price2 = price*nums


if price1 < price2:
    print(1)
elif price2 < price1:
    print(2)
else:
    print(0)


