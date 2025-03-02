ARRAY_LEN = 5


def vectAdd(a, b, c, arr_len):
    for i in range(arr_len):
        c.append(a[i] + b[i])


def main():
    # define two lists/vectors
    vectA = [1, 2, 3, 4, 5]
    vectB = [4, 5, 3, 3, 5]
    # outputs are appended here
    vectC = []

    # loop through each item, add them and append to vectC
    vectAdd(vectA, vectB, vectC, ARRAY_LEN)

    # print the vector addition result
    print(vectC)


if __name__ == "__main__":
    main()