import random
def drawSamples(pmf: dict[str, int], n: int) -> list[str]:
    # Program to generate a random number between 0 and 1
    num = list(pmf.keys()) # reading different sample bucket
    sample = list(pmf.values()) # reading no of samples 
    tot = sum(sample)
    # prob = [1 / tot] #computing prob each sample occurence
    cmf = []
    prob=[]
    for _ in sample:
        prob.append(_ / tot)
    c=0
    for i in prob:
        c+=i
        cmf.append(c)

    # importing the random module
    samplesize = n
    sampled_points = []
    for _ in range(samplesize):
        rand_num = random.random()
        print(rand_num)
        for i in cmf:
            if i >= rand_num:
                sampled_points.append(num[cmf.index(i)])
                break
    return sampled_points

# Example usage
samp_data={"abcd":1,"efgh":2,"ijkl":4,"mnop":6,"qrst":11,"uvwxyz":3}
sampled_points = drawSamples(samp_data, 5)  
# Change {} to your dictionary of points if needed
print("Sampled points:", sampled_points)