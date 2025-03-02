#include <stdio.h>

#define ARRAY_LEN 5


// takes in 3 arrays and thenumber of items in the array
void vectAdd(int *a, int *b, int *c, int len) {
    // loop through each item in `a` and `b`, 
    // add them and append to `c`
    for(int i=0; i<len; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    
    // define two arrays/vectors
    int vectA[ARRAY_LEN] = {1, 2, 3, 4, 5};
    int vectB[ARRAY_LEN] = {4, 5, 3, 3, 5};
    // outputs are appended here
    int vectC[ARRAY_LEN];

    // loop through each item, add them and append to vectC
    vectAdd(vectA, vectB, vectC, ARRAY_LEN);
    
    // print the vector addition result
    for(int i=0; i<ARRAY_LEN; i++) {
        printf("%d\n", vectC[i]);
    }

    return 0;
}