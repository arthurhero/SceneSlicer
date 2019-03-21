

def calc_padding(ker_size,dilate_rate):
    '''
    calculate how much padding is needed for 'SAME' padding
    assume square square kernel
    assume odd kernel size
    '''
    ker_size=(ker_size-1)*(dilate_rate-1)+ker_size
    margin=(ker_size-1)//2
    return margin
